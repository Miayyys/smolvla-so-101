#!/bin/bash
# Top-up of twin data v2 (2026-11-12): generator seed 3000 of scripts/twin_night_v2.sh stopped after 86 of 150 episodes
# (an object that fell over when put down raised instead of failing the trial; fixed in twin_expert.episode). The 64
# missing episodes, same options, go to a dataset of their own (so101_twin_gen_v2b) so the 236 converted ones stay.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/twin/generated_v2b
mkdir -p $OUT
pids=()
for start in 5000 7000; do
  .venv-gs/bin/python scripts/twin_generate.py --episodes 32 --seed $start --random-start --regrasp 0.3 --wide \
      --out $OUT > runs/twin/generated_v2b_$start.log 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait $pid; done
echo "generated: $(ls -d $OUT/ep_*/ | wc -l) episodes"
.venv/bin/python scripts/twin_to_lerobot.py --generated $OUT --out artifacts/datasets/so101_twin_gen_v2b > runs/twin/to_lerobot_v2b.log 2>&1
.venv/bin/python scripts/shrink_videos.py --src artifacts/datasets/so101_twin_gen_v2b --out artifacts/datasets/so101_twin_gen_v2b_512 --workers 4 2>&1 | grep --line-buffered -v '^Svt' > runs/shrink_twin_v2b.log
echo TWIN_V2B_DONE
