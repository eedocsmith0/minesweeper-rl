#!/bin/bash
# Keepalive: restarts the auto-eval watcher and dashboard web server if
# they die. Runs forever; safe to start at boot via cron @reboot.
cd "$(dirname "$0")" || exit 1
PY="$PWD/.venv-rocm/bin/python"
[ -x "$PY" ] || PY=python3
mkdir -p logs

while true; do
  sleep 60

  if ! pgrep -f "eval/watcher.py" > /dev/null; then
    echo "$(date '+%F %T') starting watcher"
    setsid "$PY" -u eval/watcher.py --models-dir models --games 30 \
      --interval 2 --tta --device cuda \
      >> logs/watcher.log 2>&1 < /dev/null &
  fi

  if ! pgrep -f "viz/dashboard.py" > /dev/null; then
    echo "$(date '+%F %T') starting dashboard"
    setsid "$PY" -u viz/dashboard.py --port 8787 \
      >> logs/dashboard.log 2>&1 < /dev/null &
  fi

  # optional RAM sentinel: only managed here if you have it locally
  # (kept out of the published repo - see PLAN.md)
  if [ -f "$PWD/scripts/memguard.sh" ]; then
    if ! pgrep -f "scripts/memguard.sh" > /dev/null; then
      echo "$(date '+%F %T') starting memguard"
      setsid nice -n 19 "$PWD/scripts/memguard.sh" \
        >> logs/memguard.launch.log 2>&1 < /dev/null &
    fi
  fi
done
