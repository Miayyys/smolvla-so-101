"""How often each recorded episode, replayed open loop in the twin, ends with the key ring in the box.

    .venv/bin/python scripts/twin_trials.py --replay runs/twin/replay_v2_soft0.010 --key-softness 0.010

One replay is not a measurement: whether the jaws catch the ring turns on fractions of a millimetre, so even the
rounding of a different numpy build flips some episodes. Each episode is therefore run --trials times with the objects
moved at random within the localisation error (--jitter-mm, --jitter-deg; the first trial unmoved), and the share of
trials with the ring in the box is reported. Placements come from the summary.json of a twin_replay.py run.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import twin_replay
import twin_scene

SETUP = {}


def setup(args):
    SETUP['scene'] = twin_scene.Scene(args.scan, {'box': args.box, 'key': args.key}, key_softness_m=args.key_softness)
    SETUP['frames'] = twin_replay.Episodes(args.dataset).frames


def trial(job):
    episode, placements = job
    scene = SETUP['scene']
    actions, states = SETUP['frames'][episode]
    scene.reset(states[0], placements)
    for action in actions:
        scene.step(action)
    return twin_replay.key_in_box(scene, scene.object_poses())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--replay', required=True, help='a twin_replay.py output directory (its placements are used)')
    parser.add_argument('--dataset', default='artifacts/datasets/so101_twin_keytied_day')
    parser.add_argument('--scan', default='runs/scan/table_curtain')
    parser.add_argument('--box', default='runs/scan/box')
    parser.add_argument('--key', default='runs/scan/key_tied')
    parser.add_argument('--key-softness', type=float, default=0.0)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--jitter-mm', type=float, default=2.0, help='standard deviation of the shift of each object')
    parser.add_argument('--jitter-deg', type=float, default=3.0, help='standard deviation of the turn of each object')
    parser.add_argument('--shift-mm', type=float, nargs=2, default=(0.0, 0.0),
                        help='a correction (base x, y) added to every top-view placement')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--out', help='json file for the results')
    args = parser.parse_args()

    rows = json.loads((Path(args.replay) / 'summary.json').read_text())
    rng = np.random.default_rng(0)
    jobs = []
    for row in rows:
        for k in range(args.trials):
            moved = {}
            for name, (x, y, yaw) in row['placements'].items():
                x, y = x + args.shift_mm[0] / 1000, y + args.shift_mm[1] / 1000
                if k:
                    dx, dy = rng.normal(0, args.jitter_mm / 1000, 2)
                    x, y, yaw = x + dx, y + dy, yaw + np.radians(rng.normal(0, args.jitter_deg))
                moved[name] = (x, y, yaw)
            jobs.append((row['episode'], moved))
    with Pool(args.workers, initializer=setup, initargs=(args,)) as pool:
        results = pool.map(trial, jobs, chunksize=1)

    shares = {}
    for row in rows:
        mine = [r for (episode, _), r in zip(jobs, results) if episode == row['episode']]
        shares[row['episode']] = float(np.mean(mine))
    print(f'shift {args.shift_mm} mm, key softness {args.key_softness * 1000:.0f} mm, {args.trials} trials per episode, jitter {args.jitter_mm} mm / '
          f'{args.jitter_deg} deg: ' + ' '.join(f'{e}:{s:.0%}' for e, s in shares.items()))
    print(f'mean share in the box {np.mean(list(shares.values())):.0%}; episodes mostly in the box (>50%): '
          f'{sum(s > 0.5 for s in shares.values())}/{len(shares)}')
    if args.out:
        Path(args.out).write_text(json.dumps({'args': vars(args), 'shares': shares}, indent=1) + '\n')


if __name__ == '__main__':
    main()
