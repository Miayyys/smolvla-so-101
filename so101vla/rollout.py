"""``lerobot-rollout`` with optional Gaussian smoothing of every predicted SmolVLA action chunk along time.

  python -m so101vla.rollout [--smooth_sigma=2] <lerobot-rollout arguments>

Also adds ``--strategy.type=correction`` (per-attempt human corrections with the leader arm, so101vla/correction.py).

RTC reset fix (always on): LeRobot's ``RTCInferenceEngine.reset`` clears the action queue but keeps the last published
observation, and the background thread may run inference on it as soon as it is resumed, before the loop publishes a
fresh one; an inference already running during the reset also lands in the cleared queue. A new episode then starts
with a chunk planned from the previous episode's last frame (key RTC eval episode 4: shoulder_lift commanded to ~40 deg
from the -103 deg rest pose for ~0.7 s), and a correction hand-back with one planned from before the takeover. The
patched reset also drops the observation (the thread waits for a fresh one) and swaps in a new queue, so a stale
in-flight chunk is merged into the discarded one.

RTC skip fix (always on): when a new chunk replaces the queue, LeRobot drops its first ``ceil(latency / dt)`` actions
(~11) as "executed during inference", even when nothing was executed (episode start, hand-back after a correction:
the arm holds still). The arm then jumps to where the plan is ~0.37 s later (up to 30 deg at hand-back). The patch
drops the number of actions actually consumed during the inference instead (LeRobot already computes it and only
warns "Indexes diff is not equal to real delay" when it differs).

The model's chunks zigzag step to step (several direction reversals per second per joint, against ~0.01/s in the
teleoperated demos). Smoothing the whole chunk once it is predicted uses only actions the model already planned, so
it adds no lag; ends are edge-padded. It wraps ``SmolVLAPolicy._get_action_chunk``, which both sync
(``select_action``) and RTC (``predict_action_chunk``) inference go through, and acts on the normalized output; the
filter is linear with weights summing to 1, so this equals filtering in joint degrees. Offline check:
``scripts/smoothing_offline.py``; record: docs/notes/action-smoothing.md.
"""

import sys

import torch


def smooth_chunk(actions, sigma):
    """Gaussian filter along time for a [batch, steps, action_dim] tensor (edge padding, radius 3 sigma)."""
    radius = int(3 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=actions.device)
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    b, t, a = actions.shape
    y = actions.float().transpose(1, 2).reshape(b * a, 1, t)
    y = torch.nn.functional.conv1d(torch.nn.functional.pad(y, (radius, radius), mode='replicate'), kernel)
    return y.reshape(b, a, t).transpose(1, 2).to(actions.dtype)


def install(sigma):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    original = SmolVLAPolicy._get_action_chunk

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        return smooth_chunk(original(self, batch, noise, **kwargs), sigma)

    SmolVLAPolicy._get_action_chunk = _get_action_chunk


def install_rtc_fixes():
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.rollout.inference.rtc import RTCInferenceEngine

    original = RTCInferenceEngine.reset

    def reset(self):
        original(self)
        with self._obs_lock:
            self._obs_holder['obs'] = None
        if self._action_queue is not None:
            self._action_queue = ActionQueue(self._rtc_config)

    RTCInferenceEngine.reset = reset

    def resolve_delay(self, real_delay, action_index_before_inference=None):
        if action_index_before_inference is None:
            return max(0, real_delay)
        return max(0, self.last_index - action_index_before_inference)

    ActionQueue._check_and_resolve_delays = resolve_delay


def install_episode_banner(total):
    """Print a visible "episode k / total" line when each episode starts and ends (LeRobot only logs
    "Recording episode <0-based dataset index>" among many INFO lines)."""
    import re

    from lerobot.rollout.strategies import episodic
    from lerobot.utils import utils

    first = []

    def log_say(text, play_sounds=True, blocking=False):
        original(text, play_sounds, blocking)
        m = re.fullmatch(r'Recording episode (\d+)', text)
        if m:
            index = int(m.group(1))
            first[:] = first or [index]
            print(f'\n========== 第 {index - first[0] + 1} / {total} 段（数据集第 {index} 段）：开始 ==========\n', flush=True)
        elif text == 'Reset the environment':
            print('\n---------- 本段结束：摆放物体后按 n 开始下一段 ----------\n', flush=True)
        elif text == 'Re-record episode':
            print('\n---------- 重录本段 ----------\n', flush=True)

    original = utils.log_say
    # episodic.py imported log_say at module level; correction.py imports it from lerobot.utils.utils when it runs
    episodic.log_say = utils.log_say = log_say


def main():
    sigma, args = 0.0, []
    for arg in sys.argv[1:]:
        if arg.startswith('--smooth_sigma='):
            sigma = float(arg.split('=', 1)[1])
        else:
            args.append(arg)
    sys.argv = [sys.argv[0], *args]
    if sigma > 0:
        install(sigma)
        print(f'so101vla.rollout: Gaussian smoothing of action chunks, sigma={sigma} steps', flush=True)
    install_rtc_fixes()
    from so101vla import correction
    correction.install()
    total = next((a.split('=', 1)[1] for a in args if a.startswith('--dataset.num_episodes=')), None)
    if total:
        install_episode_banner(total)
    from lerobot.scripts.lerobot_rollout import main as rollout_main
    rollout_main()


if __name__ == '__main__':
    main()
