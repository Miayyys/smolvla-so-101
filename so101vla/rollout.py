"""``lerobot-rollout`` with optional Gaussian smoothing of every predicted SmolVLA action chunk along time.

  python -m so101vla.rollout [--smooth_sigma=2] <lerobot-rollout arguments>

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
    from lerobot.scripts.lerobot_rollout import main as rollout_main
    rollout_main()


if __name__ == '__main__':
    main()
