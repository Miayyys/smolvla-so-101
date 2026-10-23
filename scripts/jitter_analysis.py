"""Measure high-frequency jitter of commanded actions and measured joint positions in LeRobot datasets.

  uv run --frozen python scripts/jitter_analysis.py artifacts/datasets/<name> [...]

Per joint (degrees; gripper in percent), over all episodes:
  hf_rms    RMS of x - centred moving average over 9 frames (0.3 s at 30 Hz): the part a 0.3 s smoother would remove.
  reversal  direction reversals per second: consecutive steps of opposite sign, both larger than 0.2.
  hf_ratio  share of velocity (frame-to-frame step) spectral energy at >= 3 Hz; teleoperated demos keep ~98% below 3 Hz.
  jerk_rms  RMS of the third difference, in deg/s^3. Positions are quantized to 0.088 deg (4096 ticks), which alone
            gives measured positions a floor of several thousand deg/s^3 even for smooth teleoperation.
Columns are reported for ``action`` (what was commanded) and ``observation.state`` (what the servos reached).
"""

import glob
import json
import sys

import numpy as np
import pandas as pd

WINDOW, MIN_STEP, FPS, HF_CUT = 9, 0.2, 30.0, 3.0


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
    power = np.abs(np.fft.rfft(d - d.mean(0), axis=0)) ** 2
    high = power[np.fft.rfftfreq(len(d), 1 / FPS) >= HF_CUT].sum(0)
    jerk = np.diff(x, 3, axis=0) * FPS ** 3
    return {'sq': (residual ** 2).sum(0), 'n': len(residual), 'flips': flips.sum(0), 'seconds': len(x) / FPS,
            'high': high, 'power': power.sum(0), 'jerk_sq': (jerk ** 2).sum(0), 'jerk_n': len(jerk)}


def summarize(root):
    out = {}
    for key, pick in (('action', 0), ('state', 1)):
        total = {}
        for pair in episodes(root):
            m = metrics(pair[pick].astype(np.float64))
            if m is None:
                continue
            total = {k: total.get(k, 0) + v for k, v in m.items()}
        out[key] = {'hf_rms': np.sqrt(total['sq'] / total['n']).round(2).tolist(),
                    'reversal_per_s': (total['flips'] / total['seconds']).round(2).tolist(),
                    'hf_ratio': (total['high'] / total['power']).round(3).tolist(),
                    'jerk_rms': np.sqrt(total['jerk_sq'] / total['jerk_n']).round(-1).astype(int).tolist()}
    return out


if __name__ == '__main__':
    names = json.load(open(f'{sys.argv[1]}/meta/info.json'))['features']['action']['names']
    print('joints:', [n.split('.')[0] for n in names])
    for root in sys.argv[1:]:
        print(root.rstrip('/').split('/')[-1], json.dumps(summarize(root)))
