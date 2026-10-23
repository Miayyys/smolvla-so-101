"""Offline check of action-chunk smoothing on held-out frames: jitter vs error against the recorded actions.

  uv run --frozen python scripts/smoothing_offline.py [--policy runs/so101_pilot_v2/best] [--config configs/so101_pilot_v2.json]

For every validation frame of the run (fixed noise per frame), predicts one 50-step chunk with 10 and 20 denoising
steps, applies each smoother along time, and reports per-joint-averaged:
  mae        |prediction - recorded action| over unpadded steps (degrees; gripper percent), normalized by action std
  reversal   direction reversals per second inside the chunk (steps of opposite sign, both > 0.2)
  hf_rms     RMS of chunk - 9-frame centred moving average (degrees)
  first_dev  |smoothed - raw| at chunk step 0 (how much the smoother moves the first executed action)
Smoothing is linear (edge-padded FIR) with weights summing to 1, so filtering after unnormalization equals filtering the model output.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from so101vla.policy import load_policy
from so101vla.train import Frames


def filter_time(a, kernel):
    # Centred FIR along time with edge padding (each end repeats its value).
    r = len(kernel) // 2
    padded = np.pad(a, ((r, r), (0, 0)), mode='edge')
    return np.stack([np.convolve(padded[:, j], kernel[::-1], mode='valid') for j in range(a.shape[1])], 1)


def gaussian_kernel(sigma):
    x = np.arange(-int(3 * sigma + 0.5), int(3 * sigma + 0.5) + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def savgol_kernel(window, order):
    # Least-squares polynomial fit over the window, evaluated at its centre.
    x = np.arange(window) - window // 2
    return np.linalg.pinv(np.vander(x, order + 1, increasing=True))[0]


SMOOTHERS = {
    'none': lambda a: a,
    'gauss_s1': lambda a: filter_time(a, gaussian_kernel(1.0)),
    'gauss_s2': lambda a: filter_time(a, gaussian_kernel(2.0)),
    'gauss_s3': lambda a: filter_time(a, gaussian_kernel(3.0)),
    'savgol_9_2': lambda a: filter_time(a, savgol_kernel(9, 2)),
    'savgol_15_2': lambda a: filter_time(a, savgol_kernel(15, 2)),
}


def jitter(a):
    d = np.diff(a, axis=0)
    flips = (d[1:] * d[:-1] < 0) & (np.abs(d[1:]) > 0.2) & (np.abs(d[:-1]) > 0.2)
    k = np.ones(9) / 9
    ma = np.stack([np.convolve(a[:, j], k, mode='valid') for j in range(a.shape[1])], 1)
    return flips.sum(0) / (len(a) / 30.0), np.sqrt(((a[4:-4] - ma) ** 2).mean(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default='runs/so101_pilot_v2/best')
    parser.add_argument('--config', default='configs/so101_pilot_v2.json')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg['output'])
    stats = json.loads((out / 'stats.json').read_text())
    std = np.asarray(stats['action']['std'])
    policy, pre, post = load_policy(SimpleNamespace(model_dir=Path(args.policy).resolve(),
        vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(), device='cuda'))
    frames = Frames(out, cfg)
    rows = frames.data['validation_rows']
    results = {}
    for steps in (10, 20):
        policy.config.num_steps = steps
        acc = {name: {'mae': [], 'rev': [], 'hf': [], 'first': []} for name in SMOOTHERS}
        truth_rev = []
        for row in rows:
            raw = frames.batch([int(row)])
            target = raw['action'].numpy()[0]; mask = ~raw['action_is_pad'].numpy()[0]
            torch.manual_seed(100000 + int(row))
            with torch.inference_mode():
                policy.reset()
                chunk = post(policy.predict_action_chunk(pre(raw))).float().cpu().numpy()[0]
            if mask.all():
                truth_rev.append(jitter(target)[0])
            for name, smooth in SMOOTHERS.items():
                a = smooth(chunk)
                rev, hf = jitter(a)
                acc[name]['mae'].append(np.abs(a[mask] - target[mask]).mean(0) / std)
                acc[name]['rev'].append(rev); acc[name]['hf'].append(hf)
                acc[name]['first'].append(np.abs(a[0] - chunk[0]))
        for name, v in acc.items():
            results[f'steps{steps}/{name}'] = {
                'mae': round(float(np.mean(v['mae'])), 4),
                'reversal_per_s_arm': round(float(np.mean(np.array(v['rev'])[:, :5])), 2),
                'hf_rms_arm': round(float(np.mean(np.array(v['hf'])[:, :5])), 3),
                'first_dev_max_joint_median': round(float(np.median(np.max(v['first'], axis=1))), 2)}
        results[f'steps{steps}/recorded_actions_reversal_per_s_arm'] = round(float(np.mean(np.array(truth_rev)[:, :5])), 2)
    for k, v in results.items():
        print(k, json.dumps(v))


if __name__ == '__main__':
    main()
