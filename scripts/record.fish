#!/usr/bin/env fish
# Record SO-101 teleoperation demos with both cameras, refusing to start if any device/camera config is missing.
#
#   fish scripts/record.fish -n so101_key_pilot -t "Put the key in the white box." [-e 5] [-s 40] [-r 15] [--resume]
#
# Always sources scripts/so101_env.fish itself (so a stale shell cannot drop the cameras), records into
# artifacts/datasets/<name> without pushing to the Hub, and checks afterwards that the dataset has both image streams.

argparse 'n/name=' 't/task=' 'e/episodes=' 's/episode-time=' 'r/reset-time=' 'resume' 'h/help' -- $argv
or return 1
if set -q _flag_help; or not set -q _flag_name; or not set -q _flag_task
    echo 'usage: fish scripts/record.fish -n <dataset_name> -t "<task text>" [-e episodes=5] [-s episode_s=40] [-r reset_s=15] [--resume]'
    return 1
end
set -q _flag_episodes; or set _flag_episodes 5
set -q _flag_episode_time; or set _flag_episode_time 40
set -q _flag_reset_time; or set _flag_reset_time 15

cd (status dirname)/..
source scripts/so101_env.fish
for v in FOLLOWER LEADER TOP WRIST
    if not test -e $$v
        echo "record: $v device $$v not found; check cables" >&2
        return 1
    end
end
if test -z "$CAMERAS"
    echo 'record: $CAMERAS is empty' >&2
    return 1
end

set root artifacts/datasets/$_flag_name
set extra
if set -q _flag_resume
    if not test -d $root
        echo "record: --resume given but $root does not exist" >&2
        return 1
    end
    set extra --resume=true
else if test -e $root
    echo "record: $root already exists; add --resume to append episodes, or use a new name" >&2
    return 1
end

echo "dataset: $root  task: \"$_flag_task\"  episodes: $_flag_episodes x $_flag_episode_time s (reset $_flag_reset_time s)"
echo "cameras: top=$TOP (1280x720) wrist=$WRIST (640x480)"
uv run --frozen lerobot-record \
    --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower --robot.cameras=$CAMERAS \
    --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=leader \
    --display_data=true \
    --dataset.repo_id=local/$_flag_name --dataset.root=$root \
    --dataset.single_task="$_flag_task" \
    --dataset.num_episodes=$_flag_episodes --dataset.episode_time_s=$_flag_episode_time --dataset.reset_time_s=$_flag_reset_time \
    --dataset.push_to_hub=false $extra
set rc $status

if test -f $root/meta/info.json
    uv run --frozen python -c "
import json, sys
info = json.load(open(sys.argv[1]))
images = [k for k in info['features'] if k.startswith('observation.images.')]
print(f\"check: {info['total_episodes']} episodes, {info['total_frames']} frames, image streams: {images}\")
if sorted(images) != ['observation.images.top', 'observation.images.wrist']:
    print('check: WARNING - expected observation.images.top and observation.images.wrist', file=sys.stderr)
    sys.exit(1)
" $root/meta/info.json
end
return $rc
