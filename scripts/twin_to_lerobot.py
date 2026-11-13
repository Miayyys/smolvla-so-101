"""Write the episodes of twin_generate.py as a LeRobot dataset with the real recordings' features. Runs in .venv.

    .venv/bin/python scripts/twin_to_lerobot.py --generated runs/twin/generated_v1 --out artifacts/datasets/so101_twin_gen_v1

Features (names, shapes, camera keys, fps, robot type) are copied from a real dataset (--like), so the two can be
trained on together. Episodes go in seed order; the task text is the one twin_generate.py stored.

Light: the twin is rendered in the afternoon scan's light, while the recordings range from that to the evening's (the
top view's table from about 60 to 130 in 8-bit values). Each episode is therefore scaled by a random brightness
(LIGHT_RANGE, log-uniform) and tint (TINT per channel), both cameras alike with a smaller CAMERA_SPREAD each, seeded by
the episode's name so that a rerun gives the same dataset. --no-light keeps the renders as they are.
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_KEYS = {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
LIGHT_RANGE = (0.5, 1.1)
TINT = 0.08
CAMERA_SPREAD = 0.1


def light_gains(name, cameras):
    """Per camera an RGB gain for the episode."""
    rng = np.random.default_rng(int(name.split('_')[-1]))
    common = np.exp(rng.uniform(*np.log(LIGHT_RANGE))) * (1 + rng.uniform(-TINT, TINT, 3))
    return {camera: (common * np.exp(rng.uniform(-CAMERA_SPREAD, CAMERA_SPREAD))).astype(np.float32) for camera in cameras}


def frames(path):
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format='rgb24')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generated', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--like', default='artifacts/datasets/so101_twin_keytied_day')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--no-light', action='store_true', help='no random brightness and tint')
    args = parser.parse_args()

    info = json.loads((Path(args.like) / 'meta/info.json').read_text())
    features = {k: v for k, v in info['features'].items() if k not in DEFAULT_KEYS}
    for spec in features.values():
        spec.pop('info', None)  # the video settings are the writer's own
    cameras = [k for k, v in features.items() if v['dtype'] == 'video']
    episodes = sorted(p for p in Path(args.generated).glob('ep_*') if (p / 'meta.json').exists())[:args.limit]
    dataset = LeRobotDataset.create(repo_id=f'local/{Path(args.out).name}', fps=info['fps'], features=features,
                                    root=args.out, robot_type=info.get('robot_type'), use_videos=True)
    for number, directory in enumerate(episodes):
        meta = json.loads((directory / 'meta.json').read_text())
        data = np.load(directory / 'frames.npz')
        streams = {key: frames(directory / f'{key.split(".")[-1]}.mp4') for key in cameras}
        gains = None if args.no_light else light_gains(directory.name, cameras)
        for action, state in zip(data['action'], data['state']):
            frame = {'action': action, 'observation.state': state, 'task': meta['task']}
            for key, stream in streams.items():
                image = next(stream)
                frame[key] = image if gains is None else np.clip(image * gains[key], 0, 255).astype(np.uint8)
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f'{number + 1}/{len(episodes)}: {directory.name}, {len(data["action"])} frames', flush=True)
    dataset.finalize()


if __name__ == '__main__':
    main()
