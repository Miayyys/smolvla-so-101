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


def install_rtc_reset_fix():
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
    install_rtc_reset_fix()
    from so101vla import correction
    correction.install()
    from lerobot.scripts.lerobot_rollout import main as rollout_main
    rollout_main()


if __name__ == '__main__':
    main()
