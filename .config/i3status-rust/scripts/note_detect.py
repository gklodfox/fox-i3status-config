#!/usr/bin/env python3
"""Toggle continuous note detection from a microphone.

Invoked by the i3status-rs "note" custom block on click. The first click
starts an infinite capture/detect loop that keeps STATE_FILE updated with
whatever note is currently being played; note_status.sh polls that file to
render the block. The i3bar click handler is run fully detached (see
config_bottom.toml note), so this process is free to run forever as its own
background daemon. The second click sends SIGTERM to that same process
(tracked via PID_FILE), which stops the loop and resets the block to idle.
"""
import math
import os
import signal
import sys
import subprocess
import time

import numpy as np

RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
STATE_FILE = os.path.join(RUNTIME_DIR, "i3status-note-state")
PID_FILE = os.path.join(RUNTIME_DIR, "i3status-note.pid")

# PulseAudio/PipeWire source name to record from. List available sources with
# `pactl list sources short`; this one is the HP 320 FHD Webcam's mic.
MIC_SOURCE = "alsa_input.usb-SunplusIT_Inc_HP_320_FHD_Webcam_YJGD325HP20211201V0-02.pro-input-0"

SAMPLE_RATE = 48000
# Seconds of audio processed per detection cycle while listening. parecord is
# started once per listening session (see open_recorder) and left running, so
# this only controls how many samples we read off its stream before running
# autocorrelation again -- there's no per-chunk process startup cost anymore.
# 0.25s gives ~9 periods of the lowest supported note (45Hz), which is
# comfortably reliable for a clean guitar tone.
CHUNK_DURATION = 0.25
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
CHUNK_BYTES = CHUNK_SAMPLES * 2  # s16le mono: 2 bytes per sample
MIN_FREQ = 45.0    # ~C2, below the fundamental of most instruments
MAX_FREQ = 1500.0  # covers voice and most instruments' fundamental range

# This webcam mic has a noisy analog front end: ambient-room captures measure
# ~900-1400 RMS with occasional transients, well above a quiet built-in mic's
# noise floor. SILENCE_RMS is set with margin above that; raise it if you get
# false "notes" from room noise, lower it if quiet playing isn't detected.
SILENCE_RMS = 1500
# Ratio of the autocorrelation peak to zero-lag energy. Rejects non-tonal
# transients (taps, clicks); this mic's self-noise is tonal enough that this
# alone won't reject it, so SILENCE_RMS above is the primary gate.
MIN_PERIODICITY = 0.4

# How many consecutive no-detection chunks may still show the last recognized
# note before falling back to silence. Bridges brief gaps (decay tail, a pick
# attack that fails MIN_PERIODICITY for one chunk) without freezing the
# display on a stale note once the player has actually moved on.
MAX_HOLD_CHUNKS = 2

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def write_state(status, note="-", freq="-", cents="-"):
    # note_status.sh polls this file continuously; writing in place risks it
    # reading a truncated line mid-write. Write to a sibling temp file and
    # rename instead, which POSIX guarantees is atomic on the same filesystem.
    tmp = f"{STATE_FILE}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(f"{status} {note} {freq} {cents} {int(time.time())}\n")
    os.replace(tmp, STATE_FILE)


def freq_to_note(freq):
    midi = 69 + 12 * math.log2(freq / 440.0)
    midi_round = round(midi)
    cents = round((midi - midi_round) * 100)
    name = NOTE_NAMES[midi_round % 12]
    octave = midi_round // 12 - 1
    return f"{name}{octave}", cents


def detect_pitch(samples: np.ndarray):
    samples = samples.astype(np.float64)
    samples -= samples.mean()
    if np.sqrt(np.mean(samples**2)) < SILENCE_RMS:
        return None

    # Autocorrelation via FFT: O(n log n) instead of numpy.correlate's O(n^2).
    n = len(samples)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(samples, size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    if acf[0] <= 0:
        return None

    # Raw acf[lag] = sum_{i=0}^{n-1-lag} x[i]*x[i+lag] sums over fewer
    # overlapping samples as lag grows, so its magnitude falls off with lag
    # regardless of actual periodicity. Comparing raw values (or ratios to
    # the single acf[0] reference) across the whole lag range is structurally
    # biased toward the shortest lag in range -- i.e. spuriously picks the
    # highest allowed frequency. Normalize each lag by the energy actually
    # overlapping at that lag (the NSDF from the McLeod Pitch Method) so all
    # lags are compared on equal footing; NSDF is 1.0 at lag 0 for any signal
    # and close to 1.0 at a lag with genuine strong periodicity.
    energy = np.concatenate(([0.0], np.cumsum(samples * samples)))
    lags = np.arange(n)
    overlap_energy = energy[n - lags] + energy[n] - energy[lags]
    nsdf = np.divide(2.0 * acf, overlap_energy, out=np.zeros(n), where=overlap_energy > 0)

    min_lag = int(SAMPLE_RATE / MAX_FREQ)
    max_lag = min(int(SAMPLE_RATE / MIN_FREQ), n - 1)
    if max_lag <= min_lag:
        return None

    # A periodic tone's NSDF is tall not just at the fundamental's period but
    # at every integer multiple of it too (harmonics are themselves periodic
    # at 2x, 3x, ... the fundamental's period), often comparably tall once
    # windowing effects are normalized out above. Taking the global max over
    # the whole lag range can therefore lock onto a multiple of the true
    # period -- an octave-down error -- instead of the fundamental. The
    # fundamental is always the first such peak as lag increases from 0, so
    # scan forward and take the first local maximum clearing the threshold.
    segment = nsdf[min_lag:max_lag]
    peak_lag = None
    for i in range(1, len(segment) - 1):
        if segment[i] >= MIN_PERIODICITY and segment[i] >= segment[i - 1] and segment[i] >= segment[i + 1]:
            peak_lag = min_lag + i
            break
    if peak_lag is None:
        return None
    return SAMPLE_RATE / peak_lag


def open_recorder() -> subprocess.Popen:
    # parecord (unlike arecord) talks to PulseAudio/PipeWire, so it can target
    # a specific source like the webcam mic instead of whatever ALSA's
    # "default" device resolves to. Started once and left running for the
    # whole listening session: spawning a fresh parecord per chunk (the old
    # approach) opens a brand new PipeWire client each time, which shows up
    # as a constant stream of new/switching "Recording" devices in
    # pavucontrol. Streaming raw PCM straight off stdout instead of writing
    # to a temp file avoids that churn entirely.
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
    last = None  # (note, freq_str, cents_str) of the most recently recognized note
    hold_remaining = 0  # chunks left where `last` may still be shown unconfirmed
    proc = open_recorder()
    try:
        while True:
            samples = read_chunk(proc)
            if samples is None:
                # Recorder died (device unplugged, PipeWire hiccup): drop it
                # and reconnect after a short backoff instead of busy-looping
                # respawns.
                proc.wait()
                last = None
                write_state("silence")
                time.sleep(1)
                proc = open_recorder()
                continue

            freq = detect_pitch(samples)
            if freq is not None:
                note, cents = freq_to_note(freq)
                last = (note, f"{freq:.1f}", f"{cents:+d}")
                hold_remaining = MAX_HOLD_CHUNKS
                write_state("note", *last)
            elif last is not None and hold_remaining > 0:
                # No note in this chunk (rest between notes, silence, a missed
                # attack transient): keep showing the last recognized note for
                # a bounded number of chunks instead of either flashing "no
                # signal" or freezing on a stale note indefinitely.
                hold_remaining -= 1
                write_state("note", *last)
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
        # Already listening: this click means "stop". The running process's
        # own signal handler resets the state file and removes PID_FILE.
        os.kill(pid, signal.SIGTERM)
        return

    # Stale PID file (process died without cleaning up) shouldn't block a
    # fresh start.
    try:
        os.remove(PID_FILE)
    except FileNotFoundError:
        pass

    listen_loop()


if __name__ == "__main__":
    main()
