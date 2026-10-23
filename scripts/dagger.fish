#!/usr/bin/env fish
# Collect human corrections on the real SO-101: the policy runs one task attempt per episode, you take over with the
# leader arm where it goes wrong and hand back once the current step is done (so101vla/correction.py). Every frame is
# tagged intervention = True/False; afterwards each new episode is asked for success and appended, with its correction
# statistics, to runs/dagger/<name>.csv.
#
#   fish scripts/dagger.fish -n rollout_dagger_<name> -t "Put the doll in the white box." [-p runs/so101_pilot_v2/best] [-e 10] [-s 0] [-r 600] [--smooth sigma=2, 0 = off] [--rtc] [--resume] [--dry-run]
#
# Keys (this terminal must stay focused):
#   space  model -> pause: the policy stops and the leader is driven (2 s) to the follower's pose and holds it;
#          grab the leader, then
#   tab    start correcting: the leader goes limp and the follower follows it
#   space  hand back to the model (the leader holds where you leave it); tab instead pauses with both arms holding
#   n      end the attempt (let go of the leader: it is driven back to the rest pose), r redo it, q stop
# Rule of thumb: once you take over, finish the current step (e.g. grasp and lift) before handing back; if the model
# fails at the same spot again in this attempt, finish the attempt yourself and press n.
# Between attempts the follower returns to its startup pose (start with both arms in the demo rest pose); rearrange
# the objects, then n. Attempts end only with n unless -s sets a limit on recorded seconds (paused time and the 2 s
# leader moves do not count, and a correction is never cut short). Inference flags as in eval.fish.

argparse 'n/name=' 't/task=' 'p/policy=' 'e/episodes=' 's/episode-time=' 'r/reset-time=' 'smooth=' 'rtc' 'resume' 'dry-run' 'h/help' -- $argv
or return 1
if set -q _flag_help; or not set -q _flag_name; or not set -q _flag_task
    echo 'usage: fish scripts/dagger.fish -n rollout_dagger_<name> -t "<task text>" [-p policy_dir=runs/so101_pilot_v2/best] [-e episodes=10] [-s max recorded s per attempt, 0 = no limit] [-r reset_s=600] [--smooth sigma=2, 0 = off] [--rtc] [--resume] [--dry-run]'
    return 1
end
set -q _flag_policy; or set _flag_policy runs/so101_pilot_v2/best
set -q _flag_episodes; or set _flag_episodes 10
set -q _flag_episode_time; or set _flag_episode_time 0
set -q _flag_reset_time; or set _flag_reset_time 600

cd (status dirname)/..
if not string match -q 'rollout_*' -- $_flag_name
    echo 'dagger: LeRobot requires rollout dataset names to start with rollout_' >&2
    return 1
end
if not test -f $_flag_policy/model.safetensors
    echo "dagger: no model.safetensors in $_flag_policy" >&2
    return 1
end
source scripts/so101_env.fish
for v in FOLLOWER LEADER TOP WRIST
    if not test -e $$v
        echo "dagger: $v device $$v not found; check cables" >&2
        return 1
    end
end

set root artifacts/datasets/$_flag_name
set extra
if set -q _flag_resume
    if not test -d $root
        echo "dagger: --resume given but $root does not exist" >&2
        return 1
    end
    set extra --resume=true
else if test -e $root
    echo "dagger: $root already exists; add --resume to append episodes, or use a new name" >&2
    return 1
end

set inference --inference.type=sync
set mode sync
set -q _flag_rtc; and set mode rtc; and set inference --inference.type=rtc --inference.rtc.execution_horizon=20 --inference.rtc.max_guidance_weight=10.0 --inference.queue_threshold=30
set -q _flag_smooth; or set _flag_smooth 2
test $_flag_smooth != 0; and set mode $mode+smooth$_flag_smooth

set cmd uv run --frozen python -m so101vla.rollout --smooth_sigma=$_flag_smooth \
    --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower --robot.cameras=$CAMERAS \
    --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=leader \
    --policy.path=$_flag_policy \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --strategy.type=correction $inference \
    --task="$_flag_task" --display_data=true \
    --dataset.repo_id=local/$_flag_name --dataset.root=$root --dataset.no_stamp=true \
    --dataset.single_task="$_flag_task" \
    --dataset.num_episodes=$_flag_episodes --dataset.episode_time_s=$_flag_episode_time --dataset.reset_time_s=$_flag_reset_time \
    --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false $extra

set limit none
test $_flag_episode_time != 0; and set limit $_flag_episode_time s
echo "policy: $_flag_policy  dataset: $root  task: \"$_flag_task\"  attempts: $_flag_episodes, time limit: $limit  inference: $mode"
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
test $after -gt $before; or return $rc

set csv runs/dagger/$_flag_name.csv
mkdir -p runs/dagger
test -f $csv; or echo 'episode,success,policy,task,time,note,frames,corrections,human_fraction,human_finished' >$csv
set stats (uv run --frozen python -m so101vla.correction $root | uv run --frozen python -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(r['episode'], r['frames'], r['corrections'], r['human_fraction'], int(r['human_finished']))
")
# Keys typed while the recording shut down (space, n, ...) would otherwise be read as the first answer.
python3 -c 'import termios; termios.tcflush(0, termios.TCIFLUSH)' 2>/dev/null
echo "saved episodes $before.."(math $after - 1)"; mark each: y = task completed (with or without help), n = failed, s = skip (not counted)"
for ep in (seq $before (math $after - 1))
    set row (string split ' ' -- (string match -r "^$ep .*" -- $stats))
    echo "episode $ep: $row[2] frames, $row[3] corrections, human $row[4] of frames"(test "$row[5]" = 1; and echo ', finished by human')
    while true
        read -P "episode $ep success? [y/n/s] " answer
        set answer (string trim -- $answer)
        contains -- $answer y n s; and break
    end
    test $answer = s; and continue
    echo "$ep,"(test $answer = y; and echo 1; or echo 0)",$_flag_policy,$_flag_task,"(date -Iseconds)",$mode,$row[2],$row[3],$row[4],$row[5]" >>$csv
end
uv run --frozen python -c "
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
done = sum(r['success'] == '1' for r in rows)
alone = sum(r['success'] == '1' and r['corrections'] == '0' for r in rows)
print(f'{sys.argv[1]}: {done}/{len(rows)} completed, {alone}/{len(rows)} without corrections, '
      f'{sum(int(r[\"corrections\"]) for r in rows)} corrections')
" $csv
return $rc
