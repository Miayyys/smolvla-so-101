#!/usr/bin/env fish
# Run a trained policy on the real SO-101 once, without recording anything (lerobot-rollout, base strategy).
#
#   fish scripts/try.fish -t "Put the doll in the white box." [-p runs/so101_pilot_v2/best] [-s 40] [--rtc] [--dry-run]
#
# The follower starts from the pose it has when the script starts; after -s seconds (0 = until Ctrl+C) or on Ctrl+C
# it returns there and the script exits. Nothing is saved: no dataset, no CSV. Inference settings match eval.fish.

argparse 't/task=' 'p/policy=' 's/seconds=' 'rtc' 'dry-run' 'h/help' -- $argv
or return 1
if set -q _flag_help; or not set -q _flag_task
    echo 'usage: fish scripts/try.fish -t "<task text>" [-p policy_dir=runs/so101_pilot_v2/best] [-s seconds=40, 0 = until Ctrl+C] [--rtc] [--dry-run]'
    return 1
end
set -q _flag_policy; or set _flag_policy runs/so101_pilot_v2/best
set -q _flag_seconds; or set _flag_seconds 40

cd (status dirname)/..
if not test -f $_flag_policy/model.safetensors
    echo "try: no model.safetensors in $_flag_policy" >&2
    return 1
end
source scripts/so101_env.fish
for v in FOLLOWER TOP WRIST
    if not test -e $$v
        echo "try: $v device $$v not found; check cables" >&2
        return 1
    end
end

set inference --inference.type=sync
set mode sync
set -q _flag_rtc; and set mode rtc; and set inference --inference.type=rtc --inference.rtc.execution_horizon=20 --inference.rtc.max_guidance_weight=10.0 --inference.queue_threshold=30

set cmd uv run --frozen lerobot-rollout \
    --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower --robot.cameras=$CAMERAS \
    --policy.path=$_flag_policy \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --strategy.type=base $inference \
    --task="$_flag_task" --duration=$_flag_seconds --display_data=true

echo "policy: $_flag_policy  task: \"$_flag_task\"  seconds: $_flag_seconds  inference: $mode  (not recorded; Ctrl+C stops and returns to the start pose)"
if set -q _flag_dry_run
    printf '%s\n' $cmd
    return 0
end
$cmd
