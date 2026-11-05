#!/usr/bin/env bash
#
# Control the launchd-managed paper trading bot.
#
#   ./scripts/bot.sh install     install and start the launchd agent
#   ./scripts/bot.sh status      is it running, and what did it last do
#   ./scripts/bot.sh stop        stop it and unload the agent
#   ./scripts/bot.sh start       load and start it again
#   ./scripts/bot.sh restart     bounce it (use after changing symbols)
#   ./scripts/bot.sh logs        follow the log
#   ./scripts/bot.sh uninstall   stop it and remove the agent entirely
#
# Note: while the agent is loaded, `kill <pid>` does NOT stop the bot --
# launchd restarts it within ThrottleInterval. Use `stop`.

set -uo pipefail

LABEL="com.autotrade.bot"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$PROJECT_DIR/deploy/$LABEL.plist"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
LOG="$PROJECT_DIR/logs/bot.log"

loaded() { launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; }

case "${1:-}" in
install)
    [ -x "$PROJECT_DIR/.venv/bin/python" ] || {
        echo "error: no virtualenv at $PROJECT_DIR/.venv -- run: python3 -m venv .venv" >&2
        exit 1
    }
    [ -f "$PROJECT_DIR/.env" ] || {
        echo "error: no .env -- copy .env.example and add your Alpaca keys" >&2
        exit 1
    }
    mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/logs"
    sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$TEMPLATE" >"$PLIST"
    loaded && launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
    launchctl bootstrap "$DOMAIN" "$PLIST" || {
        echo "error: bootstrap failed" >&2
        exit 1
    }
    echo "installed and started: $PLIST"
    echo "it will now start automatically at login and restart if it crashes"
    ;;

start)
    loaded && { echo "already loaded -- use restart to bounce it"; exit 0; }
    launchctl bootstrap "$DOMAIN" "$PLIST" && echo "started"
    ;;

stop)
    loaded || { echo "not loaded"; exit 0; }
    launchctl bootout "$DOMAIN/$LABEL" && echo "stopped and unloaded"
    ;;

restart)
    loaded || { echo "not loaded -- run: $0 install" >&2; exit 1; }
    launchctl kickstart -k "$DOMAIN/$LABEL" && echo "restarted"
    ;;

status)
    if loaded; then
        launchctl print "$DOMAIN/$LABEL" 2>/dev/null |
            grep -E "^\s+(state|pid|last exit code) " | sed 's/^[[:space:]]*/  /'
    else
        echo "  agent not loaded"
    fi
    echo "  --- last 3 log lines ---"
    [ -f "$LOG" ] && tail -3 "$LOG" | sed 's/^/  /' || echo "  (no log yet)"
    ;;

logs)
    tail -f "$LOG"
    ;;

uninstall)
    loaded && launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
    rm -f "$PLIST"
    echo "uninstalled -- the bot will no longer start at login"
    ;;

*)
    sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
