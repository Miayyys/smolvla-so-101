"""Figure for docs/notes/action-smoothing.md: real-arm jitter before vs after action-chunk smoothing.

  uv run --frozen --with matplotlib python scripts/smoothing_plot.py [--out docs/notes/figures/action-smoothing.png]

Top row: elbow_flex velocity over the 3 s window of episode 0 whose 9-frame-averaged commanded position moves most
(demo, RTC, RTC + sigma 2), on a shared y-axis; jitter shows as the velocity flipping sign from frame to frame.
Bottom row: velocity power spectrum of the 5 arm joints, Welch-averaged (64-frame Hann segments, 50% overlap) over
all episodes of both tasks, each curve normalized to unit total energy so its area above 3 Hz equals hf_ratio.
"""

import argparse
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

FPS, SEG, WINDOW, JOINT = 30.0, 64, 90, 2  # elbow_flex
ROOT = 'artifacts/datasets'
DEMO = ['so101_pilot_v2']
BEFORE = ['rollout_pilot_v2_final_doll_rtc', 'rollout_pilot_v2_final_key_rtc']
AFTER = ['rollout_pilot_v2_final_doll_rtc_s2', 'rollout_pilot_v2_final_key_rtc_s2']
STYLE = {'示教': ('#8a8a85', '--'), 'RTC': ('#2a78d6', '-'), 'RTC + 平滑 σ=2': ('#eb6834', '-')}
INK, MUTED, GRID = '#1f1f1e', '#6b6b66', '#e6e5e0'


def episodes(name):
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f'{ROOT}/{name}/data/*/*.parquet')))
    for _, ep in df.sort_values('index').groupby('episode_index'):
        yield np.stack(ep['action'].to_numpy()).astype(float), np.stack(ep['observation.state'].to_numpy()).astype(float)


def spectrum(names, pick):
    win = np.hanning(SEG)
    total, count = 0.0, 0
    for name in names:
        for pair in episodes(name):
            v = np.diff(pair[pick][:, :5], axis=0)
            for s in range(0, len(v) - SEG + 1, SEG // 2):
                seg = v[s:s + SEG] - v[s:s + SEG].mean(0)
                total = total + (np.abs(np.fft.rfft(seg * win[:, None], axis=0)) ** 2).sum(1)
                count += 1
    f = np.fft.rfftfreq(SEG, 1 / FPS)
    p = total / count
    return f, p / (p.sum() * (f[1] - f[0]))


def busiest_window(name):
    action, state = next(episodes(name))
    x = np.convolve(action[:, JOINT], np.ones(9) / 9, mode='same')
    start = max(range(4, len(x) - WINDOW - 5), key=lambda s: np.ptp(x[s:s + WINDOW]))
    cut = slice(start, start + WINDOW + 1)
    return np.diff(action[cut, JOINT]) * FPS, np.diff(state[cut, JOINT]) * FPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='docs/notes/figures/action-smoothing.png')
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'font.size': 10, 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(11, 7), facecolor='#fcfcfb')
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.1], hspace=0.45, wspace=0.6)
    t = np.arange(WINDOW) / FPS
    top = None
    groups = {'示教': DEMO, 'RTC': BEFORE, 'RTC + 平滑 σ=2': AFTER}
    for i, (label, names) in enumerate(groups.items()):
        ax = top = fig.add_subplot(grid[0, 2 * i:2 * i + 2], sharey=top)
        action, state = busiest_window(names[0])
        color = STYLE[label][0]
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.plot(t, action, color=color, lw=1.4, label='下发动作')
        ax.plot(t, state, color=INK, lw=1.0, alpha=0.55, label='实测位置')
        ax.set_title(label, color=color, fontsize=11, loc='left')
        ax.set_xlabel('时间 (s)')
        ax.grid(color=GRID, lw=0.6)
        if i == 0:
            ax.set_ylabel('肘关节速度 (°/s)')
            ax.legend(frameon=False, fontsize=9)
    for i, (pick, title) in enumerate(((0, '下发动作的速度谱'), (1, '实测位置的速度谱'))):
        ax = fig.add_subplot(grid[1, 3 * i:3 * i + 3])
        for label, names in groups.items():
            f, p = spectrum(names, pick)
            color, ls = STYLE[label]
            ax.semilogy(f, p, color=color, ls=ls, lw=1.8, label=label)
        ax.axvline(3, color=MUTED, lw=0.8, ls=':')
        ax.text(3.2, ax.get_ylim()[1] * 0.5, '3 Hz', color=MUTED, fontsize=9, va='top')
        ax.set_title(title, loc='left', fontsize=11)
        ax.set_xlabel('频率 (Hz)')
        ax.set_xlim(0, 15)
        ax.grid(color=GRID, lw=0.6)
        if i == 0:
            ax.set_ylabel('归一化能量密度 (1/Hz)')
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle('动作块平滑前后的抖动（第 15000 步模型，真机，玩偶与钥匙各 10 段）', x=0.07, ha='left', fontsize=12)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print('saved', args.out)


if __name__ == '__main__':
    main()
