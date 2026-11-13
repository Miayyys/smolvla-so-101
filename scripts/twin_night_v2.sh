#!/bin/bash
# Overnight twin data v2 (2026-11-11): random starts, 30% deliberate-miss regrasps, wide placements; 300 episodes
# in two generator processes, then the LeRobot dataset and its upload-size copy (scripts/shrink_videos.py).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/twin/generated_v2
mkdir -p $OUT
for start in 1000 3000; do
  .venv-gs/bin/python scripts/twin_generate.py --episodes 150 --seed $start --random-start --regrasp 0.3 --wide \
      --out $OUT > runs/twin/generated_v2_$start.log 2>&1 &
done
wait
echo "generated: $(ls -d $OUT/ep_*/ | wc -l) episodes"
.venv/bin/python scripts/twin_to_lerobot.py --generated $OUT --out artifacts/datasets/so101_twin_gen_v2 > runs/twin/to_lerobot_v2.log 2>&1
.venv/bin/python scripts/shrink_videos.py --src artifacts/datasets/so101_twin_gen_v2 --out artifacts/datasets/so101_twin_gen_v2_512 --workers 4 2>&1 | grep --line-buffered -v '^Svt' > runs/shrink_twin_v2.log
echo TWIN_V2_DONE
