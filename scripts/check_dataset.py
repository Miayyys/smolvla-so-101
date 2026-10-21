"""Sanity-check a recorded LeRobot dataset: per-episode length, video/table frame agreement, duplicate camera frames,
state/action ranges and timestamp regularity; optionally save a contact sheet of sampled frames.

    python scripts/check_dataset.py artifacts/datasets/so101_key_pilot [--sheet out.jpg]

Duplicate frames: consecutive decoded frames that are (almost) identical while the arm is moving (some joint changed by
more than 0.5 deg between the two steps), i.e. the camera delivered no new image for that 30 Hz step (the top camera
runs at ~26.5 fps, so ~12% is expected; see docs/hardware.md). Steps where the arm is still are excluded because the
scene itself does not change there.
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd


def decode(path, start_s, end_s):
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(start_s - 0.5, 0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start_s - 1e-3:
                continue
            if t > end_s + 1e-3:
                break
            frames.append(frame.to_ndarray(format='rgb24'))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('--sheet', type=Path, help='save sampled frames (rows: episodes; start/middle/end per camera)')
    args = parser.parse_args()

    info = json.loads((args.root / 'meta/info.json').read_text())
    fps = info['fps']
    cams = [k for k in info['features'] if k.startswith('observation.images.')]
    data = pd.concat([pd.read_parquet(p) for p in sorted((args.root / 'data').rglob('*.parquet'))])
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((args.root / 'meta/episodes').rglob('*.parquet'))])
    tasks = pd.read_parquet(args.root / 'meta/tasks.parquet')
    print(f"{args.root}: {info['total_episodes']} episodes, {info['total_frames']} frames, fps {fps}, cameras {cams}")
    print('tasks:', list(tasks.index) if 'task' not in tasks.columns else list(tasks['task']))

    names = info['features']['observation.state'].get('names')
    state = np.stack(data['observation.state'].to_numpy())
    action = np.stack(data['action'].to_numpy())
    print('state  min/max:', np.round(state.min(0), 1).tolist(), np.round(state.max(0), 1).tolist())
    print('action min/max:', np.round(action.min(0), 1).tolist(), np.round(action.max(0), 1).tolist())
    if names:
        print('joints:', names)
    print('max |action - state| per joint:', np.round(np.abs(action - state).max(0), 1).tolist())

    sheet_rows = []
    for _, ep in episodes.sort_values('episode_index').iterrows():
        idx = int(ep['episode_index'])
        rows = data[data['episode_index'] == idx].sort_values('frame_index')
        ts = rows['timestamp'].to_numpy()
        dt = np.diff(ts)
        ep_state = np.stack(rows['observation.state'].to_numpy())
        moving = np.abs(np.diff(ep_state, axis=0)).max(1) > 0.5
        line = (f'ep{idx}: {len(rows)} frames ({len(rows) / fps:.1f} s), dt {dt.min() * 1000:.1f}-{dt.max() * 1000:.1f} ms, '
                f'moving {moving.mean():.0%}')
        sheet = []
        for cam in cams:
            chunk, file = int(ep[f'videos/{cam}/chunk_index']), int(ep[f'videos/{cam}/file_index'])
            start = float(ep[f'videos/{cam}/from_timestamp'])
            path = args.root / info['video_path'].format(video_key=cam, chunk_index=chunk, file_index=file)
            frames = decode(path, start + ts[0], start + ts[-1])
            diffs = np.array([np.abs(a.astype(np.int16) - b.astype(np.int16)).mean() for a, b in zip(frames, frames[1:])])
            n = min(len(diffs), len(moving))
            dup = float((diffs[:n][moving[:n]] < 0.5).mean()) if moving[:n].any() else float('nan')
            line += f' | {cam.split(".")[-1]}: {len(frames)} decoded, {dup:.0%} duplicate while moving'
            if args.sheet and frames:
                picks = [frames[0], frames[len(frames) // 2], frames[-1]]
                sheet += [cv2.resize(f, (320, 240 if f.shape[0] * 4 == f.shape[1] * 3 else 180)) for f in picks]
        print(line)
        if args.sheet:
            h = max(im.shape[0] for im in sheet)
            sheet_rows.append(np.hstack([np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0))) for im in sheet]))
    if args.sheet and sheet_rows:
        cv2.imwrite(str(args.sheet), cv2.cvtColor(np.vstack(sheet_rows), cv2.COLOR_RGB2BGR))
        print('sheet:', args.sheet)


if __name__ == '__main__':
    main()
