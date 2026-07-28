#!/usr/bin/env bash
# Polled by the i3status-rs "phone" custom block (json = true).
# Reports Good/green only while adb has PHONE_IP connected AND scrcpy is
# still mirroring it; closing scrcpy (or losing the adb connection) drops
# it back to gray. Otherwise shows the in-progress scan/connect/error state
# written by phone_connect.sh, which fades back to idle/gray after
# PENDING_TTL.
set -u

STATE_FILE="${XDG_RUNTIME_DIR:-/tmp}/i3status-phone-state"
PHONE_IP="192.168.100.31"
PHONE_IP_RE="${PHONE_IP//./\\.}"
PENDING_TTL=180  # seconds a scanning/connecting/error message stays on screen

if adb devices 2>/dev/null | grep -qE "^${PHONE_IP_RE}:[0-9]+[[:space:]]+device$" \
    && pgrep -x scrcpy >/dev/null; then
    echo '{"icon":"phone","state":"Good","text":"connected"}'
    exit 0
fi

status="idle"
ts=0
if [[ -f "$STATE_FILE" ]]; then
    read -r status ts < "$STATE_FILE"
fi

age=$(( $(date +%s) - ts ))
if (( age > PENDING_TTL )); then
    status="idle"
fi

case "$status" in
    scanning)
        echo '{"icon":"phone","state":"Info","text":"scanning…"}'
        ;;
    connecting)
        echo '{"icon":"phone","state":"Info","text":"connecting…"}'
        ;;
    error)
        echo '{"icon":"phone","state":"Warning","text":"error"}'
        ;;
    *)
        echo '{"icon":"phone","state":"Idle","text":""}'
        ;;
esac
