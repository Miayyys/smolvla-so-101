#!/bin/bash
# Power the computer off once the overnight twin v2 pipeline (scripts/twin_night_v2.sh) has ended (user's request,
# 2026-11-11). Whether it finished or failed is written to runs/twin/poweroff.log first.
cd "$(dirname "$0")/.."
while pgrep -f twin_night_v2.sh > /dev/null; do sleep 60; done
{
  date
  if grep -q TWIN_V2_DONE runs/twin/night_v2.log; then echo "twin v2 pipeline finished"; else echo "twin v2 pipeline ended WITHOUT finishing:"; tail -5 runs/twin/night_v2.log; fi
  ls -d runs/twin/generated_v2/ep_*/ 2>/dev/null | wc -l
  cat artifacts/datasets/so101_twin_gen_v2_512/meta/shrink_report.json 2>/dev/null
  sync
} >> runs/twin/poweroff.log 2>&1
sleep 30
systemctl poweroff
