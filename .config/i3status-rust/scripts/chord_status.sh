#!/usr/bin/env bash
# Polled by the i3status-rs "chord" custom block (json = true).
# Renders whatever chord_detect.py last wrote to STATE_FILE.
set -u

STATE_FILE="${XDG_RUNTIME_DIR:-/tmp}/i3status-chord-state"
CHORD_TTL=1  # seconds a detected chord stays on screen before falling back to idle
ERROR_TTL=1  # seconds a "no signal" message stays on screen before falling back to idle
# Both are just a safety net for a crashed/killed listener that didn't clean up;
# during normal operation chord_detect.py refreshes STATE_FILE roughly twice a second while listening.

status="idle"
chord="-"
confidence=0
ts=0

if [[ -f "$STATE_FILE" ]]; then
    read -r status chord confidence ts < "$STATE_FILE"
fi

# confidence/ts should always be numeric, but guard against a torn read
# (state file caught mid-write) leaving them short/empty.
[[ "$confidence" =~ ^[0-9]+$ ]] || confidence=0
[[ "$ts" =~ ^[0-9]+$ ]] || ts=0

printf -v now '%(%s)T' -1
age=$(( now - ts ))

case "$status" in
    listening)
        echo '{"icon":"music","state":"Info","text":"listening…"}'
        ;;
    chord)
        if (( age <= CHORD_TTL )); then
            # chord_detect.py only ever reports a chord once its match score
            # clears its own MIN_SCORE floor (15%), and a clean strum of a
            # real chord scores 85-100 in practice -- so these cutoffs split
            # "confident" from "barely cleared the floor", not confidence
            # from 0.
            if (( confidence >= 70 )); then
                state="Good"
            elif (( confidence >= 40 )); then
                state="Warning"
            else
                state="Idle"
            fi
            echo "{\"icon\":\"music\",\"state\":\"${state}\",\"text\":\"${chord} (${confidence}%)\"}"
        else
            echo '{"icon":"music","state":"Idle","text":""}'
        fi
        ;;
    silence)
        if (( age <= ERROR_TTL )); then
            echo '{"icon":"music","state":"Warning","text":"-"}'
        else
            echo '{"icon":"music","state":"Idle","text":"-"}'
        fi
        ;;
    *)
        echo '{"icon":"music","state":"Idle","text":""}'
        ;;
esac
