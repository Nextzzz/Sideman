#!/usr/bin/env bash
# Multi-day collection driver: batches of MAX_NEW downloads with rests,
# longer back-off on YouTube's bot-check, stops when nothing is left
# (dead videos are retried once more on the final pass).
#   HOOK_SUBSET=train ./collect_driver.sh
set -u
cd "$(dirname "$0")"
export HOOK_SUBSET="${HOOK_SUBSET:-train}"
export MAX_NEW="${MAX_NEW:-350}"
LOG="collect_${HOOK_SUBSET}.log"
idle_runs=0
for round in $(seq 1 200); do
  echo "=== round $round $(date '+%F %T') ===" | tee -a "$LOG"
  out=$(./.venv/Scripts/python hooktheory_collect.py 2>&1 | tee -a "$LOG" | tail -3)
  echo "$out"
  if echo "$out" | grep -q "BOT-CHECK"; then
    echo "bot-check: resting 45 min" | tee -a "$LOG"; sleep 2700; continue
  fi
  if echo "$out" | grep -q "BATCH LIMIT"; then
    echo "batch done: resting 20 min" | tee -a "$LOG"; sleep 1200; continue
  fi
  new=$(echo "$out" | grep -o "ok=[0-9]*" | tail -1 | cut -d= -f2)
  remaining=$(echo "$out" | grep -o "remaining=[0-9]*" | tail -1 | cut -d= -f2)
  if [ "${new:-0}" -eq 0 ]; then idle_runs=$((idle_runs+1)); else idle_runs=0; fi
  if [ "${remaining:-1}" -eq 0 ] || [ "$idle_runs" -ge 2 ]; then
    echo "FINISHED: remaining=${remaining:-?} (dead videos left)" | tee -a "$LOG"; exit 0
  fi
  sleep 600
done
