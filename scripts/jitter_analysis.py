"""Measure high-frequency jitter of commanded actions and measured joint positions in LeRobot datasets.

  uv run --frozen python scripts/jitter_analysis.py artifacts/datasets/<name> [...]

Per joint (degrees; gripper in percent), over all episodes:
  hf_rms    RMS of x - centred moving average over 9 frames (0.3 s at 30 Hz): the part a 0.3 s smoother would remove.
  reversal  direction reversals per second: consecutive steps of opposite sign, both larger than 0.2.
Columns are reported for ``action`` (what was commanded) and ``observation.state`` (what the servos reached).
"""

import glob
import json
import sys

import numpy as np
import pandas as pd

WINDOW, MIN_STEP, FPS = 9, 0.2, 30.0


def episodes(root):
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f'{root}/data/*/*.parquet')))
    for _, ep in df.sort_values('index').groupby('episode_index'):
        yield np.stack(ep['action'].to_numpy()), np.stack(ep['observation.state'].to_numpy())


def metrics(x):
    if len(x) <= WINDOW:
        return None
    kernel = np.ones(WINDOW) / WINDOW
    smooth = np.stack([np.convolve(x[:, j], kernel, mode='valid') for j in range(x.shape[1])], 1)
    residual = x[WINDOW // 2: len(x) - WINDOW // 2] - smooth
    d = np.diff(x, axis=0)
    flips = (d[1:] * d[:-1] < 0) & (np.abs(d[1:]) > MIN_STEP) & (np.abs(d[:-1]) > MIN_STEP)
    return (residual ** 2).sum(0), len(residual), flips.sum(0), len(x) / FPS


def summarize(root):
    out = {}
    for key, pick in (('action', 0), ('state', 1)):
        sq = n = 0; flips = 0; seconds = 0.0
        for pair in episodes(root):
            m = metrics(pair[pick].astype(np.float64))
            if m is None:
                continue
            sq, n, flips, seconds = sq + m[0], n + m[1], flips + m[2], seconds + m[3]
        out[key] = {'hf_rms': np.sqrt(sq / n).round(2).tolist(), 'reversal_per_s': (flips / seconds).round(2).tolist()}
    return out


if __name__ == '__main__':
    names = json.load(open(f'{sys.argv[1]}/meta/info.json'))['features']['action']['names']
    print('joints:', [n.split('.')[0] for n in names])
    for root in sys.argv[1:]:
        print(root.rstrip('/').split('/')[-1], json.dumps(summarize(root)))
