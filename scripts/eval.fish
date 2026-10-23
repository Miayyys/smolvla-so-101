#!/usr/bin/env fish
# Run a trained policy on the real SO-101 for a few episodes (lerobot-rollout, episodic strategy), record the rollouts,
# then ask for the success of each new episode and append it to runs/real_eval/<name>.csv.
#
#   fish scripts/eval.fish -n rollout_key_v1_best -t "Put the key in the white box." [-p runs/so101_key_v1/best] [-e 10] [-s 40] [-r 600] [--smooth sigma=2, 0 = off] [--rtc] [--resume] [--dry-run]
#
# No leader arm: between episodes the follower returns (1 s) to the pose it had when the script started and holds it
# while you rearrange the objects, so start with the arm in the same rest pose used when recording demos.
# Keys as in record.fish: n ends the episode / reset early, r re-runs the episode, q stops (finished episodes are saved).
# The policy was trained with top -> camera1 and wrist -> camera2, hence the rename map.
# Default inference is sync: the arm waits ~0.18 s for each new 50-step chunk. --rtc uses LeRobot's Real-Time Chunking:
# a background thread replans once 30 actions are left and guides the new chunk to match the next 20 old actions
# (horizon 20 > the ~11-step real inference delay, so the first executed actions are still blended).
# Each predicted chunk is Gaussian-smoothed along time (sigma 2 steps, so101vla/rollout.py) unless --smooth 0.

argparse 'n/name=' 't/task=' 'p/policy=' 'e/episodes=' 's/episode-time=' 'r/reset-time=' 'smooth=' 'rtc' 'resume' 'dry-run' 'h/help' -- $argv
or return 1
if set -q _flag_help; or not set -q _flag_name; or not set -q _flag_task
    echo 'usage: fish scripts/eval.fish -n rollout_<name> -t "<task text>" [-p policy_dir=runs/so101_key_v1/best] [-e episodes=10] [-s episode_s=40] [-r reset_s=600] [--smooth sigma=2, 0 = off] [--rtc] [--resume] [--dry-run]'
    return 1
end
set -q _flag_policy; or set _flag_policy runs/so101_key_v1/best
set -q _flag_episodes; or set _flag_episodes 10
set -q _flag_episode_time; or set _flag_episode_time 40
set -q _flag_reset_time; or set _flag_reset_time 600

cd (status dirname)/..
if not string match -q 'rollout_*' -- $_flag_name
    echo 'eval: LeRobot requires rollout dataset names to start with rollout_' >&2
    return 1
end
if not test -f $_flag_policy/model.safetensors
    echo "eval: no model.safetensors in $_flag_policy" >&2
    return 1
end
source scripts/so101_env.fish
for v in FOLLOWER TOP WRIST
    if not test -e $$v
        echo "eval: $v device $$v not found; check cables" >&2
        return 1
    end
end
if test -z "$CAMERAS"
    echo 'eval: $CAMERAS is empty' >&2
    return 1
end

set root artifacts/datasets/$_flag_name
set extra
if set -q _flag_resume
    if not test -d $root
        echo "eval: --resume given but $root does not exist" >&2
        return 1
    end
    set extra --resume=true
else if test -e $root
    echo "eval: $root already exists; add --resume to append episodes, or use a new name" >&2
    return 1
end

set inference --inference.type=sync
set mode sync
set -q _flag_rtc; and set mode rtc; and set inference --inference.type=rtc --inference.rtc.execution_horizon=20 --inference.rtc.max_guidance_weight=10.0 --inference.queue_threshold=30
set -q _flag_smooth; or set _flag_smooth 2
test $_flag_smooth != 0; and set mode $mode+smooth$_flag_smooth

set cmd uv run --frozen python -m so101vla.rollout --smooth_sigma=$_flag_smooth \
    --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower --robot.cameras=$CAMERAS \
    --policy.path=$_flag_policy \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --strategy.type=episodic $inference \
    --task="$_flag_task" --display_data=true \
    --dataset.repo_id=local/$_flag_name --dataset.root=$root --dataset.no_stamp=true \
    --dataset.single_task="$_flag_task" \
    --dataset.num_episodes=$_flag_episodes --dataset.episode_time_s=$_flag_episode_time --dataset.reset_time_s=$_flag_reset_time \
    --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false $extra

echo "policy: $_flag_policy  dataset: $root  task: \"$_flag_task\"  episodes: $_flag_episodes x $_flag_episode_time s  inference: $mode"
if set -q _flag_dry_run
    printf '%s\n' $cmd
    return 0
end

set before 0
test -f $root/meta/info.json; and set before (uv run --frozen python -c "import json,sys; print(json.load(open(sys.argv[1]))['total_episodes'])" $root/meta/info.json)
$cmd
set rc $status
test -f $root/meta/info.json; or return $rc
set after (uv run --frozen python -c "import json,sys; print(json.load(open(sys.argv[1]))['total_episodes'])" $root/meta/info.json)

set csv runs/real_eval/$_flag_name.csv
mkdir -p runs/real_eval
test -f $csv; or echo 'episode,success,policy,task,time,note' >$csv
# Keys typed while the recording shut down (space, n, ...) would otherwise be read as the first answer.
python3 -c 'import termios; termios.tcflush(0, termios.TCIFLUSH)' 2>/dev/null
echo "saved episodes $before.."(math $after - 1)"; mark each: y = success, n = failure, s = skip (not counted)"
for ep in (seq $before (math $after - 1))
    while true
        read -P "episode $ep success? [y/n/s] " answer
        set answer (string trim -- $answer)
        contains -- $answer y n s; and break
    end
    test $answer = s; and continue
    echo "$ep,"(test $answer = y; and echo 1; or echo 0)",$_flag_policy,$_flag_task,"(date -Iseconds)",$mode" >>$csv
end
uv run --frozen python -c "
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ok = sum(r['success'] == '1' for r in rows)
print(f'{sys.argv[1]}: {ok}/{len(rows)} successful')
" $csv
return $rc
