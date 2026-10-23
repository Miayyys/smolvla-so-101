"""Hardware-free check of the correction pipeline: strategy -> dataset -> merge with demos -> train.py prepare.

  uv run --frozen python scripts/correction_selftest.py [--demo artifacts/datasets/rollout_pilot_v2_final_key_rtc_s2]

1. Runs so101vla.correction's strategy against fake follower/leader/inference engine with scripted key presses
   (counted in follower observations, so independent of timing): attempt 0 = model, take over, correct, hand back,
   model, n; attempt 1 = r (discarded); attempt 2 = model, take over, correct, n while correcting. Real LeRobot dataset
   writing with the same features and video encoding as the real-arm rollouts (frames are flat gray).
2. Checks the saved intervention flags, the leader torque sequence (goal = present position at every torque enable)
   and that correction frames record the leader's positions.
3. Merges a small real dataset (stand-in for the demos) with the fake corrections via scripts/merge_corrections.py and
   runs train.py's prepare on it: policy frames must be excluded and correction chunks padded after their run.
Everything is written under artifacts/selftest_correction/ and removed afterwards unless --keep.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Event
import time
from types import SimpleNamespace

import numpy as np
import torch

WORK = Path('artifacts/selftest_correction')
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
KEYS = {(0, 20): 'space', (0, 32): 'tab', (0, 62): 'space', (0, 92): 'n',
        (1, 15): 'r',
        (2, 25): 'space', (2, 37): 'tab', (2, 70): 'n'}


class Follower:
    def __init__(self):
        self.pos = {f'{j}.pos': 10.0 * i for i, j in enumerate(JOINTS)}
        self.attempt, self.count, self.strategy = -1, 0, None
        self.is_connected = True
        self.inner = self

    def get_observation(self):
        self.count += 1
        key = KEYS.get((self.attempt, self.count))
        if key:
            self.strategy._on_key(key)
        shade = np.uint8(self.count % 200)
        return {**self.pos, 'top': np.full((720, 1280, 3), shade, np.uint8), 'wrist': np.full((480, 640, 3), shade, np.uint8)}

    def send_action(self, action):
        self.pos.update({k: float(v) for k, v in action.items() if k in self.pos})
        return action

    def disconnect(self):
        self.is_connected = False


class Leader:
    def __init__(self):
        self.pos = {f'{j}.pos': 0.0 for j in JOINTS}
        self.goal, self.torque, self.log = dict(self.pos), False, []
        self.is_connected = True

    def get_action(self):
        if not self.torque:  # limp: the operator moves the shoulder
            self.pos['shoulder_pan.pos'] += 0.3
        return dict(self.pos)

    def send_feedback(self, goal):
        self.goal = dict(goal)
        if self.torque:
            self.pos = dict(goal)

    def enable_torque(self):
        assert all(abs(self.goal[k] - self.pos[k]) < 1e-9 for k in self.pos), 'torque enabled with a stale goal'
        self.torque = True
        self.log.append('on')

    def disable_torque(self):
        self.torque = False
        self.log.append('off')

    def disconnect(self):
        self.is_connected = False


class Engine:
    ready = True

    def __init__(self):
        self.step, self.log = 0, []

    def get_action(self, obs_frame):
        self.step += 1
        return torch.tensor([10.0 * i + 0.01 * self.step for i in range(len(JOINTS))])

    def __getattr__(self, name):
        if name in ('reset', 'start', 'stop', 'pause', 'resume', 'notify_observation'):
            return lambda *a, **k: self.log.append(name)
        raise AttributeError(name)


def record():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from so101vla import correction

    strategy_cls = correction.install()
    reference = json.loads(Path(ARGS.demo, 'meta/info.json').read_text())['features']
    features = {k: {f: v for f, v in reference[k].items() if f != 'info'}
                for k in ('action', 'observation.state', 'observation.images.top', 'observation.images.wrist')}
    features['intervention'] = {'dtype': 'bool', 'shape': (1,), 'names': None}
    root = WORK / 'rollout_selftest'
    dataset = LeRobotDataset.create('local/rollout_selftest', 30, features, root=root, robot_type='so_follower',
                                    streaming_encoding=True, encoder_threads=2)
    follower, leader, engine = Follower(), Leader(), Engine()
    cfg = SimpleNamespace(fps=30, task='selftest', play_sounds=False, use_torch_compile=False, interpolation_multiplier=1,
                          display_data=False, return_to_initial_position=True,
                          dataset=SimpleNamespace(num_episodes=2, episode_time_s=30, reset_time_s=1.0,
                                                  single_task='Put the key in the white box.', push_to_hub=False))
    identity = SimpleNamespace(robot_observation_processor=lambda obs: obs, teleop_action_processor=lambda x: x[0],
                               robot_action_processor=lambda x: x[0])
    ctx = SimpleNamespace(runtime=SimpleNamespace(cfg=cfg, shutdown_event=Event()),
                          hardware=SimpleNamespace(robot_wrapper=follower, teleop=leader, initial_position=dict(follower.pos)),
                          data=SimpleNamespace(dataset=dataset, dataset_features={**features}, ordered_action_keys=[f'{j}.pos' for j in JOINTS]),
                          processors=identity, policy=SimpleNamespace(inference=engine))
    strategy = strategy_cls(SimpleNamespace(type='correction'))
    follower.strategy = strategy
    attempt = strategy._attempt

    def counted_attempt(*args):
        follower.attempt += 1
        follower.count = 0
        attempt(*args)

    strategy._attempt = counted_attempt
    wait, waits = strategy._wait, []

    def timed_wait(*args):
        start = time.perf_counter()
        wait(*args)
        waits.append(time.perf_counter() - start)

    strategy._wait = timed_wait
    strategy.setup(ctx)
    strategy.run(ctx)
    strategy.teardown(ctx)
    # The n/r that ended an attempt must not also end the reset phase after it (no n is pressed during resets here).
    print('reset phases (s)', [round(w, 2) for w in waits])
    assert len(waits) == 2 and min(waits) > 0.9, waits
    return root, leader


def check_recording(root, leader):
    import pandas as pd
    from so101vla import correction

    stats = correction.summary(root)
    print('summary', json.dumps(stats))
    assert list(stats) == [0, 1], stats
    assert stats[0]['corrections'] == 1 and not stats[0]['human_finished'], stats
    assert stats[1]['corrections'] == 1 and stats[1]['human_finished'], stats
    table = pd.concat(pd.read_parquet(p) for p in sorted((root / 'data').rglob('*.parquet'))).sort_values('index')
    for ep, rows in table.groupby('episode_index'):
        human = np.stack(rows['intervention'].to_numpy()).reshape(-1).astype(bool)
        pan = np.stack(rows['action'].to_numpy())[:, 0]
        # Leader-driven frames carry the leader's shoulder positions, which rise 0.3 per read while limp.
        assert np.all(np.diff(pan[human]) > 0.25), (ep, np.diff(pan[human])[:5])
        print(f'episode {ep}: frames {len(human)}, pattern', ''.join('H' if h else 'm' for h in human))
    print('leader torque sequence', leader.log)
    # take over (on: hold, on: smooth move), correct (off), hand back (on), park (on, on, off), redo park, attempt 2 ...
    assert leader.log[:4] == ['on', 'on', 'off', 'on'], leader.log
    assert leader.log[-1] == 'off', leader.log


def check_merge_and_prepare(root):
    from so101vla.train import prepare

    demo = Path(ARGS.demo)
    n_demo = json.loads((demo / 'meta/info.json').read_text())['total_episodes']
    demo_partition = WORK / 'demo_partition.json'
    demo_partition.write_text(json.dumps({'splits': {'train': list(range(n_demo - 2)), 'validation': [n_demo - 2, n_demo - 1], 'test': []}}))
    tagged = demo.with_name(demo.name + '_intervention')
    assert not tagged.exists(), f'{tagged} exists; remove it or pick another --demo'
    try:
        subprocess.run([sys.executable, 'scripts/merge_corrections.py', '--demo', str(demo), '--demo-partition', str(demo_partition),
                        '--out', 'selftest_correction_merged', '--partition-out', str(WORK / 'partition.json'), str(root)], check=True)
        merged = demo.parent / 'selftest_correction_merged'
        cfg = {'dataset_root': str(merged), 'partition': str(WORK / 'partition.json'), 'validation_frames_per_episode': 2,
               'camera_map': {'observation.images.camera1': 'observation.images.top', 'observation.images.camera2': 'observation.images.wrist'},
               'trim_static_tail': {'tolerance_deg': 2.0, 'keep_frames': 15}}
        out = WORK / 'prepared'; out.mkdir()
        prepare(cfg, out)
        data = np.load(out / 'data.npz')
        manifest = json.loads((out / 'prepared.json').read_text())
        eps, train_rows, end_rows = data['episodes'], set(data['train_rows'].tolist()), data['end_rows']
        import pandas as pd
        table = pd.concat(pd.read_parquet(p) for p in sorted((merged / 'data').rglob('*.parquet'))).sort_values('index')
        human = np.stack(table['intervention'].to_numpy()).reshape(-1).astype(bool)
        assert human[eps < n_demo].all() and not human[eps >= n_demo].all()
        assert not train_rows & set(np.flatnonzero(~human).tolist()), 'policy frames sampled'
        from so101vla.correction import runs
        for ep in (n_demo, n_demo + 1):
            rows = np.flatnonzero(eps == ep)
            for start, stop in runs(human[rows]):
                run = rows[start:stop]
                assert (end_rows[run] <= run[-1]).all(), (ep, end_rows[run], run[-1])
                print(f'merged episode {ep}: correction rows {run[0]}..{run[-1]}, chunk ends {sorted(set(end_rows[run].tolist()))}')
        print('prepare manifest: train_frames', manifest['train_frames'], 'policy_frames_excluded', manifest['policy_frames_excluded'])
    finally:
        if not ARGS.keep:
            shutil.rmtree(tagged, ignore_errors=True)
            shutil.rmtree(demo.parent / 'selftest_correction_merged', ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', default='artifacts/datasets/rollout_pilot_v2_final_key_rtc_s2')
    parser.add_argument('--keep', action='store_true')
    ARGS = parser.parse_args()
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    try:
        root, leader = record()
        check_recording(root, leader)
        check_merge_and_prepare(root)
        print('correction selftest: OK')
    finally:
        if not ARGS.keep:
            shutil.rmtree(WORK, ignore_errors=True)
