#!/bin/zsh
# Relaunches src/run_extension.py until all 900 calls are recorded. Restarts on crash (e.g. jetsam SIGKILL under
# memory pressure) and on stalls (no log output for STALL_SECONDS). Usage:
#   nohup src/extension_watchdog.sh [runner args...] >/dev/null 2>&1 &
ROOT=${0:A:h:h}
cd "$ROOT" || exit 1
LOG=runs/extension/run.log
OUT=runs/extension/qwen8b_main.jsonl
STALL_SECONDS=${STALL_SECONDS:-1200}
MAX_RESTARTS=${MAX_RESTARTS:-40}
echo $$ > runs/extension/watchdog.pid
restarts=0
while true; do
  n=$( [ -f $OUT ] && grep -c . $OUT || echo 0 )
  if [ "$n" -ge 900 ]; then echo "[watchdog] $(date -u +%FT%TZ) complete ($n records)" >> $LOG; break; fi
  if [ $restarts -ge $MAX_RESTARTS ]; then echo "[watchdog] $(date -u +%FT%TZ) giving up after $restarts restarts" >> $LOG; break; fi
  echo "[watchdog] $(date -u +%FT%TZ) launch #$restarts records=$n args=$*" >> $LOG
  PYTHONUNBUFFERED=1 uv run python -m src.run_extension "$@" >> $LOG 2>&1 &
  child=$!
  echo $child > runs/extension/runner.pid
  while kill -0 $child 2>/dev/null; do
    sleep 30
    age=$(( $(date +%s) - $(stat -f %m $LOG) ))
    if [ $age -gt $STALL_SECONDS ]; then
      echo "[watchdog] $(date -u +%FT%TZ) stalled ${age}s; killing $child" >> $LOG
      pkill -P $child 2>/dev/null; kill -9 $child 2>/dev/null
    fi
  done
  wait $child; code=$?
  echo "[watchdog] $(date -u +%FT%TZ) runner exited code=$code" >> $LOG
  tail -1 $LOG | grep -q COMPLETE && continue
  restarts=$((restarts+1))
  sleep 30
done
rm -f runs/extension/watchdog.pid runs/extension/runner.pid
