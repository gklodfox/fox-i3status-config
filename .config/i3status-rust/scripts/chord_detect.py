#!/usr/bin/env python3
"""Toggle continuous guitar chord detection from a microphone.

Sibling to note_detect.py, same daemon/toggle architecture (see that file's
docstring for the click/SIGTERM lifecycle), but a different core algorithm:
note_detect.py assumes one fundamental at a time and uses autocorrelation,
which is the right tool for a monophonic signal but cannot separate several
simultaneous strings. A strummed chord needs a polyphonic approach instead:

  1. FFT the chunk and pick spectral peaks (candidate partials/fundamentals).
  2. Fold each peak's frequency down to a pitch class (chroma bin 0-11),
     discarding octave -- this is what lets "which notes are present" survive
     six strings ringing across three-plus octaves.
  3. Score the resulting chroma vector against every (root, quality)
     template (major, minor, 7th, sus, ...) and take the best match.

A plucked string's overtones don't all land on its own chord tones (the 3rd
harmonic is a fifth *above* the fundamental's pitch class, not the same one),
so a single note's harmonic series can bleed into a genuinely different
chord tone's bin, or occasionally a foreign one. build_chroma and
match_chord below have comments on the specific choices (max over sum,
sqrt-normalized weighting, worst-bin noise penalty) that keep that bleed
from being mistaken for a real note. This is a heuristic build for casual
practice feedback, not a rigorous MIR system -- expect the occasional wrong
guess on murky or heavily-overtone-rich input.
"""
import math
import os
import signal
import sys
import subprocess
import time

import numpy as np

RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
STATE_FILE = os.path.join(RUNTIME_DIR, "i3status-chord-state")
PID_FILE = os.path.join(RUNTIME_DIR, "i3status-chord.pid")

# Same physical mic as note_detect.py. PipeWire/PulseAudio happily serves
# multiple simultaneous recording clients from one source, so this can run
# alongside note_detect.py without contention.
MIC_SOURCE = "alsa_input.usb-SunplusIT_Inc_HP_320_FHD_Webcam_YJGD325HP20211201V0-02.pro-input-0"

SAMPLE_RATE = 48000
# Chords need a much longer analysis window than single-note detection: to
# tell E2 (82.4Hz) apart from F2 (87.3Hz) the FFT bin width must be well
# under their ~5Hz gap. 0.5s gives a true resolution of ~2Hz (with parabolic
# interpolation sharpening that further) while still updating roughly twice
# a second -- slower than note_detect's 0.25s chunks, but a strummed chord
# rings for a while so that's not a responsiveness problem in practice.
CHUNK_DURATION = 0.5
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
CHUNK_BYTES = CHUNK_SAMPLES * 2  # s16le mono: 2 bytes per sample

# Same noisy-mic reasoning as note_detect.py's SILENCE_RMS.
SILENCE_RMS = 1500

# Spectral peak search range: below the low E2 string and above the point
# where guitar fundamentals are still musically ambiguous. Harmonics of low
# strings above MAX_FREQ are lost, but plenty survive under it to reinforce
# the chroma vector.
MIN_FREQ = 70.0
MAX_FREQ = 1300.0

# A peak must clear this fraction of the loudest in-range peak to count --
# rejects noise-floor bins so a handful of real string partials come through.
PEAK_REL_THRESHOLD = 0.12
# Minimum spacing (Hz) between accepted peaks. Zero-padding the FFT gives far
# more bins than the window's true ~2Hz resolution, so without this a single
# spectral lobe's shoulder bins would each look like a separate "peak".
MIN_PEAK_SEP_HZ = 3.0 / CHUNK_DURATION
MAX_PEAKS = 8

FFT_SIZE = 1 << (CHUNK_SAMPLES - 1).bit_length()
BIN_WIDTH = SAMPLE_RATE / FFT_SIZE

# How much a template's score is hurt by energy on pitch classes it doesn't
# include. Without this, a 2-note power-chord template ("5") would match
# almost anything that also has a full triad, since it never looks at the
# 3rd. With it, a strongly-present 3rd counts as unexplained "noise" against
# the power-chord template and correctly loses to the full triad.
NOISE_PENALTY = 0.6
# Tiny nudge towards fewer chord tones so genuine ties (which barely happen
# in practice) resolve towards the simpler reading rather than a richer one.
COMPLEXITY_BIAS = 0.015
# Tiny nudge towards the root matching the lowest detected string. See
# match_chord's docstring: this is what resolves pitch-class-identical ties
# like Csus4 vs Fsus2. Bigger than COMPLEXITY_BIAS so it wins a tie against
# it, but still far smaller than the gap between genuinely different-scoring
# chords.
BASS_BIAS = 0.02
# Below this score the chroma vector doesn't cleanly match any template
# (still tuning, muted strings, transition between chords) -- detect_chord
# returns None, which the caller treats the same as no pitch detected at all
# (falls back to hold/silence), rather than reporting a confident-looking
# wrong guess.
MIN_SCORE = 0.15

# How many consecutive silent/unmatched chunks may still show the last
# recognized chord before falling back. Mirrors note_detect.py's
# MAX_HOLD_CHUNKS; kept short since each chunk here already covers 0.5s.
MAX_HOLD_CHUNKS = 1

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# (intervals from root, display suffix). Order only matters as a tie-break
# nudge alongside COMPLEXITY_BIAS.
QUALITIES = [
    ((0, 4, 7), ""),
    ((0, 3, 7), "m"),
    ((0, 2, 7), "sus2"),
    ((0, 5, 7), "sus4"),
    ((0, 4, 7, 10), "7"),
    ((0, 3, 7, 10), "m7"),
    ((0, 4, 7, 11), "maj7"),
    ((0, 3, 6), "dim"),
    ((0, 4, 8), "aug"),
    ((0, 7), "5"),
]


def write_state(status, chord="-", confidence: object = "-"):
    # chord_status.sh polls this file continuously; write-then-rename keeps
    # each read atomic (see note_detect.py's write_state for the same trick).
    tmp = f"{STATE_FILE}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(f"{status} {chord} {confidence} {int(time.time())}\n")
    os.replace(tmp, STATE_FILE)


def find_peaks(samples: np.ndarray):
    """Return up to MAX_PEAKS (freq_hz, magnitude) spectral peaks, loudest first."""
    windowed = samples.astype(np.float64) * np.hanning(len(samples))
    spectrum = np.fft.rfft(windowed, n=FFT_SIZE)
    mag = np.abs(spectrum)

    min_bin = max(1, int(MIN_FREQ / BIN_WIDTH))
    max_bin = min(len(mag) - 2, int(MAX_FREQ / BIN_WIDTH))
    if max_bin <= min_bin:
        return []

    window_max = mag[min_bin:max_bin + 1].max()
    if window_max <= 0:
        return []
    threshold = PEAK_REL_THRESHOLD * window_max

    candidates = []
    for i in range(min_bin + 1, max_bin):
        m = mag[i]
        if m < threshold or m < mag[i - 1] or m < mag[i + 1]:
            continue
        # Parabolic (log-magnitude) interpolation for sub-bin frequency
        # accuracy -- flat FFT bins alone aren't fine enough to separate
        # adjacent semitones at low guitar frequencies.
        a, b, c = math.log(mag[i - 1] + 1e-12), math.log(m + 1e-12), math.log(mag[i + 1] + 1e-12)
        denom = a - 2 * b + c
        offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
        freq = (i + offset) * BIN_WIDTH
        candidates.append((freq, m))

    candidates.sort(key=lambda p: p[1], reverse=True)
    accepted = []
    for freq, m in candidates:
        if all(abs(freq - af) >= MIN_PEAK_SEP_HZ for af, _ in accepted):
            accepted.append((freq, m))
        if len(accepted) >= MAX_PEAKS:
            break
    return accepted


def build_chroma(peaks):
    # Peaks come in as raw FFT magnitudes, which for a real chunk sit in a
    # huge, mostly-irrelevant absolute range (millions+). Normalizing against
    # the loudest peak first, then compressing with sqrt, keeps the
    # *relative* loudness gap between a true fundamental and a much quieter
    # overtone-of-another-note bleeding into its bin -- compressing the raw
    # magnitude directly (e.g. log1p on values already in the millions)
    # flattens that gap away, making a faint harmonic bleed look almost as
    # important as a real chord tone.
    ref = max((m for _, m in peaks), default=0.0)
    chroma = np.zeros(12)
    if ref <= 0:
        return chroma
    for freq, mag in peaks:
        if freq <= 0:
            continue
        pitch = 69 + 12 * math.log2(freq / 440.0)
        bin_idx = int(round(pitch)) % 12
        # sqrt compresses dynamic range a little so the loudest string
        # doesn't completely drown out quieter ones in the same strum, while
        # still preserving enough of the loudness ordering to tell a real
        # (if quiet) chord tone apart from a faint stray harmonic bleeding in
        # from a different note. Taking the max rather than summing matters
        # just as much: a root doubled across octaves/strings (very common
        # in open guitar chords, e.g. open E major doubles E across three
        # octaves) has each copy's fundamental land at roughly the same
        # magnitude as any other string's fundamental, so summing them would
        # artificially inflate that pitch class relative to a chord tone
        # that's only played once -- max keeps "is this pitch class present"
        # from turning into "how many times, and how loud in total".
        weight = math.sqrt(mag / ref)
        if weight > chroma[bin_idx]:
            chroma[bin_idx] = weight
    peak = chroma.max()
    return chroma / peak if peak > 0 else chroma


def match_chord(chroma, bass_pitch_class=None):
    """Return (root, suffix, confidence_0_1) for the best-scoring template.

    bass_pitch_class, when given, is the pitch class of the lowest detected
    string. Chroma alone can't always pick a root: sus2 and sus4 chords a
    fourth apart are literally the same three pitch classes (Csus4 = C,F,G;
    Fsus2 = F,G,C), so they score identically. A guitarist's actual bass
    note is the real tie-breaker for "which one is it", so a matching root
    gets a small nudge -- too small to override a genuinely better-scoring
    template, just enough to resolve an exact or near tie in its favor.
    """
    best = (0, "", -1e9, 0.0)  # root, suffix, biased_score, raw_confidence -- overwritten by the loop's first (always-better) candidate
    all_bins = set(range(12))
    for intervals, suffix in QUALITIES:
        n = len(intervals)
        for root in range(12):
            tone_bins = {(root + iv) % 12 for iv in intervals}
            noise_bins = all_bins - tone_bins
            tone_score = sum(chroma[b] for b in tone_bins) / n
            # The worst unexplained bin, not the average one: a simpler
            # template (e.g. a "5" power chord) leaves most of the other ten
            # bins near zero anyway, so averaging would dilute away the one
            # bin that actually matters -- a clearly-present 3rd that the
            # richer triad template accounts for but this one doesn't.
            noise_score = max((chroma[b] for b in noise_bins), default=0.0)
            confidence = max(0.0, min(1.0, tone_score - NOISE_PENALTY * noise_score))
            biased = confidence - COMPLEXITY_BIAS * n
            if bass_pitch_class is not None and root == bass_pitch_class:
                biased += BASS_BIAS
            if biased > best[2]:
                best = (root, suffix, biased, confidence)
    return best[0], best[1], best[3]


def detect_chord(samples: np.ndarray):
    rms = math.sqrt(np.mean((samples.astype(np.float64) - samples.mean()) ** 2))
    if rms < SILENCE_RMS:
        return None

    peaks = find_peaks(samples)
    if not peaks:
        return None
    chroma = build_chroma(peaks)
    if chroma.max() <= 0:
        return None

    bass_freq = min(freq for freq, _ in peaks)
    bass_pitch_class = int(round(69 + 12 * math.log2(bass_freq / 440.0))) % 12

    root, suffix, confidence = match_chord(chroma, bass_pitch_class)
    if confidence < MIN_SCORE:
        return None
    return f"{NOTE_NAMES[root]}{suffix}", round(confidence * 100)


def open_recorder() -> subprocess.Popen:
    # See note_detect.py's open_recorder: parecord targets a specific
    # PipeWire/PulseAudio source and is left running for the whole session
    # rather than respawned per chunk.
    return subprocess.Popen(
        [
            "parecord", "--raw",
            f"--device={MIC_SOURCE}",
            "--rate", str(SAMPLE_RATE),
            "--channels", "1",
            "--format", "s16le",
            "--latency-msec", "50",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def read_chunk(proc: subprocess.Popen):
    """Read one CHUNK_DURATION worth of samples from a running recorder.

    Returns None on EOF, meaning the recorder died (e.g. device unplugged).
    """
    assert proc.stdout is not None  # guaranteed by open_recorder's stdout=PIPE
    buf = bytearray()
    while len(buf) < CHUNK_BYTES:
        data = proc.stdout.read(CHUNK_BYTES - len(buf))
        if not data:
            return None
        buf += data
    return np.frombuffer(bytes(buf), dtype=np.int16)


def cleanup_and_exit(signum=None, frame=None):
    try:
        os.remove(PID_FILE)
    except FileNotFoundError:
        pass
    write_state("idle")
    sys.exit(0)


def listen_loop():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    signal.signal(signal.SIGTERM, cleanup_and_exit)
    signal.signal(signal.SIGINT, cleanup_and_exit)

    write_state("listening")
    last = None  # (chord, confidence) of the most recently recognized chord
    hold_remaining = 0
    proc = open_recorder()
    try:
        while True:
            samples = read_chunk(proc)
            if samples is None:
                proc.wait()
                last = None
                write_state("silence")
                time.sleep(1)
                proc = open_recorder()
                continue

            result = detect_chord(samples)
            if result is not None:
                last = result
                hold_remaining = MAX_HOLD_CHUNKS
                write_state("chord", *last)
            elif last is not None and hold_remaining > 0:
                hold_remaining -= 1
                write_state("chord", *last)
            else:
                last = None
                write_state("silence")
    finally:
        proc.terminate()
        proc.wait()


def running_pid():
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None

    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def main():
    pid = running_pid()
    if pid is not None:
        os.kill(pid, signal.SIGTERM)
        return

    try:
        os.remove(PID_FILE)
    except FileNotFoundError:
        pass

    listen_loop()


if __name__ == "__main__":
    main()
