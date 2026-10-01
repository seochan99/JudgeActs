#!/bin/bash
# Sequential, resumable MLX extension runs with a watchdog: relaunch on crash or 20 min without a new record.
# Usage: nohup tools/run_mlx_watchdog.sh qwen4b_mlx qwen8b_mlx >> runs/extension/run_mlx.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
STALL=1200
for NAME in "$@"; do
  OUT="runs/extension/${NAME}_main.jsonl"
  for TRY in $(seq 1 30); do
    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 900 ]; then echo "[watchdog] $NAME complete"; break; fi
    echo "[watchdog] $(date -u +%FT%TZ) launch $NAME try=$TRY"
    .venv/bin/python -u -m src.run_mlx --name "$NAME" 2>&1 | grep --line-buffered -v 'UserWarning\|warnings.warn' &
    PID=$!
    LAST=$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 ); SINCE=$(date +%s)
    while kill -0 $PID 2>/dev/null; do
      sleep 30
      NOW=$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 )
      if [ "$NOW" != "$LAST" ]; then LAST=$NOW; SINCE=$(date +%s); fi
      if [ $(( $(date +%s) - SINCE )) -gt $STALL ]; then
        echo "[watchdog] $(date -u +%FT%TZ) $NAME stalled at $NOW records; killing"
        pkill -f "src.run_mlx --name $NAME"; sleep 10; pkill -9 -f "src.run_mlx --name $NAME"
      fi
    done
    sleep 5
  done
done
echo "[watchdog] $(date -u +%FT%TZ) ALL DONE"
