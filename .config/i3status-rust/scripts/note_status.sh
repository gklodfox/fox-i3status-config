#!/usr/bin/env bash
# Polled by the i3status-rs "note" custom block (json = true).
# Renders whatever note_detect.py last wrote to STATE_FILE.
set -u

STATE_FILE="${XDG_RUNTIME_DIR:-/tmp}/i3status-note-state"
NOTE_TTL=1    # seconds a detected note stays on screen before falling back to idle
ERROR_TTL=1   # seconds a "no signal" message stays on screen before falling back to idle
# Both are just a safety net for a crashed/killed listener that didn't clean up;
# during normal operation note_detect.py refreshes STATE_FILE every ~1s while listening.

status="idle"
note="-"
freq="-"
cents="-"
ts=0

if [[ -f "$STATE_FILE" ]]; then
    read -r status note freq cents ts < "$STATE_FILE"
fi

# ts should always be numeric, but guard against a torn read (state file
# caught mid-write) leaving it short/empty, which would abort the script.
[[ "$ts" =~ ^[0-9]+$ ]] || ts=0

printf -v now '%(%s)T' -1
age=$(( now - ts ))

case "$status" in
    listening)
        echo '{"icon":"music","state":"Info","text":"listening…"}'
        ;;
    note)
        if (( age <= NOTE_TTL )); then
            echo "{\"icon\":\"music\",\"state\":\"Good\",\"text\":\"${note} (${cents}c, ${freq}Hz)\"}"
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
