#!/usr/bin/env bash
# Triggered by the i3status-rs "phone" custom block on left-click.
# Scans PHONE_IP for its adb-over-tcpip port, connects, then mirrors the
# screen with scrcpy. That chain can run for a long time (scrcpy lasts until
# its window is closed), so the work is detached with setsid; the click
# itself returns immediately and phone_status.sh polls STATE_FILE to reflect
# progress, falling back to the live adb connection state once connected.
set -u

STATE_FILE="${XDG_RUNTIME_DIR:-/tmp}/i3status-phone-state"
PHONE_IP="192.168.100.31"

write_state() {
    echo "$1 $(date +%s)" > "$STATE_FILE"
}

if [[ "${1:-}" == "--worker" ]]; then
    write_state "scanning"

    port=$(nmap -sV -p 20000-65535 "$PHONE_IP" \
        | awk -F/ '/^[0-9]+\/tcp[[:space:]]+open[[:space:]]/ {print $1; exit}')

    if [[ -z "$port" ]]; then
        write_state "error"
        exit 1
    fi

    write_state "connecting"

    if ! adb connect "${PHONE_IP}:${port}" | grep -q "connected"; then
        write_state "error"
        exit 1
    fi

    scrcpy --select-tcpip -S >/dev/null 2>&1
    write_state "idle"
    exit 0
fi

# Ignore extra clicks while a scan/connect is already in flight.
if [[ -f "$STATE_FILE" ]]; then
    read -r status ts < "$STATE_FILE"
    age=$(( $(date +%s) - ts ))
    if [[ "$status" =~ ^(scanning|connecting)$ ]] && (( age <= 300 )); then
        exit 0
    fi
fi

setsid "$0" --worker >/dev/null 2>&1 &
disown
