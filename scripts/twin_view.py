"""Watch a recorded episode replayed in the twin's physics, in MuJoCo's viewer.

    .venv/bin/python scripts/twin_view.py --replay runs/twin/replay_refined_render --episode 0

Object placements come from the summary.json of a twin_replay.py run (its arguments are reused). The episode plays at
recorded speed and starts over at the end. Space pauses, Backspace restarts, the arrow keys step back and forward a
second; the viewer's own mouse and keys work as usual. Grey shapes are the collision geometry (the box, the key ring's
pieces, the jaw slices).
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import twin_replay
import twin_scene

KEY_SPACE, KEY_BACKSPACE, KEY_LEFT, KEY_RIGHT = 32, 259, 263, 262


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--replay', required=True, help='a twin_replay.py output directory')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--dataset', default='artifacts/datasets/so101_twin_keytied_day')
    parser.add_argument('--scan', default='runs/scan/table_curtain')
    parser.add_argument('--box', default='runs/scan/box')
    parser.add_argument('--key', default='runs/scan/key_tied')
    parser.add_argument('--key-softness', type=float, default=0.010)
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()

    rows = {r['episode']: r for r in json.loads((Path(args.replay) / 'summary.json').read_text())}
    if args.episode not in rows:
        raise SystemExit(f'episode {args.episode} is not in {args.replay}/summary.json')
    placements = {k: tuple(v) for k, v in rows[args.episode]['placements'].items()}
    actions, states = twin_replay.Episodes(args.dataset).frames[args.episode]
    scene = twin_scene.Scene(args.scan, {'box': args.box, 'key': args.key}, key_softness_m=args.key_softness)

    state = {'paused': False, 'goto': 0}

    def on_key(key):
        if key == KEY_SPACE:
            state['paused'] = not state['paused']
        elif key == KEY_BACKSPACE:
            state['goto'] = 0
        elif key in (KEY_LEFT, KEY_RIGHT):
            state['goto'] = max(0, state['frame'] + (30 if key == KEY_RIGHT else -30))

    def replay_to(frame):
        """The physics is not reversible: jumping replays from the start up to the frame."""
        scene.reset(states[0], placements)
        for action in actions[:frame]:
            scene.step(action)
        return frame

    with mujoco.viewer.launch_passive(scene.model, scene.data, key_callback=on_key) as viewer:
        viewer.opt.geomgroup[3] = 1  # collision shapes: the objects have no other geometry
        viewer.cam.lookat[:] = [0.25, 0.0, 0.0]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.6, 180, -35
        state['frame'] = 0
        while viewer.is_running():
            started = time.time()
            if state['goto'] is not None:
                with viewer.lock():
                    state['frame'] = replay_to(state['goto'])
                state['goto'] = None
            elif not state['paused']:
                with viewer.lock():
                    scene.step(actions[state['frame']])
                state['frame'] += 1
                if state['frame'] >= len(actions):
                    state['goto'] = 0
            frame = min(state['frame'], len(actions) - 1)
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              f"episode {args.episode}  frame {frame}/{len(actions)}" + ('  paused' if state['paused'] else ''),
                              f"jaw {scene.opening_percent():.0f}% (recorded {states[frame, 5]:.0f}%)"))
            viewer.sync()
            time.sleep(max(0.0, twin_scene.FRAME_S / args.speed - (time.time() - started)))


if __name__ == '__main__':
    main()
