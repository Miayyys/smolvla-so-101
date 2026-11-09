"""Fit the MuJoCo actuator model of the SO-101 to recorded teleoperation (no robot time needed).

Each episode's recorded actions are replayed open loop from its first state, and the simulated joints are compared with
the recorded joints. One set of parameters is shared by all servos (they are the same STS3215 with the same gains):
position gain, velocity gain, joint damping, friction loss, armature, a command delay, and a first-order low-pass on
the command. The low-pass fits to zero on this arm; only the short delay is needed. The force limit stays at the MJCF
value: free motion never reaches it, so these recordings say nothing about it.

The MJCF joint limits are replaced by the arm's own calibrated range. The real rest pose is 6 degrees beyond the MJCF
limit of shoulder_lift, so with the MJCF limits the simulated arm cannot reach the pose the demonstrations start from.

    python scripts/sysid_actuators.py --dataset artifacts/datasets/so101_pilot_v3 \
        --extrinsics runs/calib/top_extrinsics_2/extrinsics.json --out runs/sysid/v4

Only the five arm joints are scored: the gripper closes on objects that are not in this scene, and the errors exclude
each episode's start and end. 'working' errors also exclude frames with the arm folded at its rest pose. Angles are
converted with the joint zero offsets from the calibration (docs/calibration.md).
"""

import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from calib_top_extrinsics import TICKS_PER_TURN, load_calibration

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
FRAME_S = 1 / 30
SUBSTEPS = 16
SKIP_START, SKIP_END = 30, 45
TIP_IN_GRIPPER = np.array([-0.0089, -0.0002, -0.1044])
NAMES = ['kp', 'kv', 'damping', 'frictionloss', 'armature', 'forcerange', 'delay_s', 'filter_s']
LOWER = np.array([10.0, 0.05, 0.005, 0.001, 0.001, 0.5, 0.0, 0.0])
UPPER = np.array([5000.0, 60.0, 8.0, 2.0, 1.0, 30.0, 0.25, 0.25])
START = np.array([100.0, 2.0, 0.8, 0.04, 0.14, 3.35, 0.02, 0.09])
MJCF_DEFAULT = np.array([998.22, 2.731, 0.60, 0.052, 0.028, 3.35, 0.0, 0.0])
FORCE_LIMIT = 3.35
WORKING_LIFT_RAD = np.radians(-90)  # shoulder_lift below this means the arm is folded at its rest pose


def load_episodes(dataset, offsets_deg):
    files = sorted(glob.glob(str(Path(dataset) / 'data/chunk-*/*.parquet')))
    columns = ['action', 'observation.state', 'episode_index', 'intervention']
    frame = pq.read_table(files, columns=columns).to_pandas()
    episodes = {}
    offsets = np.concatenate([offsets_deg, [0.0]])
    for index, group in frame.groupby('episode_index'):
        if not np.all(np.stack(group['intervention'].values)):  # keep demonstrations, not policy rollouts
            continue
        episodes[int(index)] = (np.radians(np.stack(group['action'].values) + offsets),
                                np.radians(np.stack(group['observation.state'].values) + offsets))
    return episodes


def gripper_percent_to_rad(values, low=-0.17453, high=1.74533):
    return low + np.degrees(values) / 100 * (high - low)


def calibrated_limits(offsets_deg):
    """The arm's calibrated range of each of the five joints, in MJCF radians."""
    calibration = load_calibration()
    half = np.array([(calibration[j]['range_max'] - calibration[j]['range_min']) / 2 for j in JOINTS]) * 360 / TICKS_PER_TURN
    return np.radians(np.column_stack([offsets_deg - half, offsets_deg + half]))


class Sim:
    def __init__(self, mjcf, limits):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.model.jnt_range[:5] = limits
        self.model.actuator_ctrlrange[:5] = limits
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = FRAME_S / SUBSTEPS
        self.gripper = self.model.body('gripper').id

    def apply(self, p):
        kp, kv, damping, friction, armature, force = p[:6]
        m = self.model
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
        m.actuator_biasprm[:, 2] = -kv
        m.actuator_forcerange[:] = [-force, force]
        m.dof_damping[:] = damping
        m.dof_frictionloss[:] = friction
        m.dof_armature[:] = armature

    def replay(self, p, action, state):
        self.apply(p)
        delay, filter_s, m, d = p[6], p[7], self.model, self.data
        mujoco.mj_resetData(m, d)
        target = np.column_stack([action[:, :5], gripper_percent_to_rad(action[:, 5])])
        d.qpos[:5] = state[0, :5]
        d.qpos[5] = gripper_percent_to_rad(state[0, 5])
        mujoco.mj_forward(m, d)
        times = np.arange(len(target)) * FRAME_S
        command = np.append(state[0, :5], gripper_percent_to_rad(state[0, 5]))
        alpha = 1.0 if filter_s <= 0 else 1 - np.exp(-m.opt.timestep / filter_s)
        out = np.zeros((len(target), 5))
        for frame in range(len(target)):
            out[frame] = d.qpos[:5]
            for sub in range(SUBSTEPS):
                t = (frame + sub / SUBSTEPS) * FRAME_S - delay
                for j in range(6):
                    command[j] += alpha * (np.interp(t, times, target[:, j]) - command[j])
                d.ctrl[:] = command
                mujoco.mj_step(m, d)
        return out

    def tip(self, q):
        self.data.qpos[:5] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.gripper] + self.data.xmat[self.gripper].reshape(3, 3) @ TIP_IN_GRIPPER


SIM = None


def init_worker(mjcf, limits):
    global SIM
    SIM = Sim(mjcf, limits)


def with_working(errors, state):
    """Per-frame joint errors plus a last column marking frames away from the rest pose; start and end are dropped."""
    return np.column_stack([errors, state[-len(errors):, 1] > WORKING_LIFT_RAD])[SKIP_START:-SKIP_END]


def score_episode(job):
    params, action, state = job
    simulated = SIM.replay(params, action, state)
    return with_working(np.degrees(np.abs(simulated - state[:, :5])), state)


def evaluate(pool, params, episodes):
    jobs = [(params, a, s) for a, s in episodes]
    return np.concatenate(pool.map(score_episode, jobs))


def to_params(x):
    """Search space: logs of the five positive parameters, delay and filter as is; clipped to the bounds."""
    return np.clip(np.concatenate([np.exp(x[:5]), [FORCE_LIMIT], x[5:]]), LOWER, UPPER)


def nelder_mead(cost, x0, step, max_evals):
    """Plain Nelder-Mead; numpy only (scipy is not in the project environment)."""
    n = len(x0)
    simplex = [x0] + [x0 + step[i] * np.eye(n)[i] for i in range(n)]
    values = [cost(x) for x in simplex]
    evals = n + 1
    while evals < max_evals:
        order = np.argsort(values)
        simplex, values = [simplex[i] for i in order], [values[i] for i in order]
        if values[-1] - values[0] < 1e-4:
            break
        centre = np.mean(simplex[:-1], axis=0)
        reflected = centre + (centre - simplex[-1])
        f_reflected = cost(reflected)
        evals += 1
        if f_reflected < values[0]:
            expanded = centre + 2 * (centre - simplex[-1])
            f_expanded = cost(expanded)
            evals += 1
            simplex[-1], values[-1] = (expanded, f_expanded) if f_expanded < f_reflected else (reflected, f_reflected)
        elif f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
        else:
            contracted = centre + 0.5 * (simplex[-1] - centre)
            f_contracted = cost(contracted)
            evals += 1
            if f_contracted < values[-1]:
                simplex[-1], values[-1] = contracted, f_contracted
            else:
                simplex = [simplex[0]] + [simplex[0] + 0.5 * (x - simplex[0]) for x in simplex[1:]]
                values = [values[0]] + [cost(x) for x in simplex[1:]]
                evals += n
        if evals % 50 < 3:
            print(f'  {evals} evals, best {min(values):.4f} deg', flush=True)
    best = int(np.argmin(values))
    return simplex[best], values[best]


def lag_baseline(episodes, frames):
    """Error of 'the joint simply follows the command, delayed': how much a dynamics model has to beat."""
    return np.concatenate([with_working(np.degrees(np.abs(s[frames:, :5] - a[:-frames, :5])), s) for a, s in episodes])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--extrinsics', required=True, help='for the MJCF path and joint zero offsets')
    parser.add_argument('--out', required=True)
    parser.add_argument('--stride', type=int, default=16, help='take every Nth demonstration for training')
    parser.add_argument('--max-evals', type=int, default=500)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()

    extrinsics = json.loads(Path(args.extrinsics).read_text())
    offsets = np.array([extrinsics['joint_offset_deg'][j] for j in JOINTS])
    episodes = load_episodes(args.dataset, offsets)
    indices = sorted(episodes)
    train_ids = indices[::args.stride]
    val_ids = indices[args.stride // 2::args.stride]
    train = [episodes[i] for i in train_ids]
    val = [episodes[i] for i in val_ids]
    print(f'{len(indices)} demonstrations; train {len(train)}, validation {len(val)}', flush=True)

    def table(name, scored):
        errors, working = scored[:, :5], scored[:, 5] > 0.5
        print(f'{name:34s} all {errors.mean():.3f} deg ' + ' '.join(f'{v:.2f}' for v in errors.mean(0))
              + f' | working {errors[working].mean():.3f} deg ' + ' '.join(f'{v:.2f}' for v in errors[working].mean(0)), flush=True)

    limits = calibrated_limits(offsets)
    with Pool(args.workers, init_worker, (extrinsics['mjcf'], limits)) as pool:
        table('MJCF defaults, no delay (val)', evaluate(pool, MJCF_DEFAULT, val))
        for lag in (3, 4):
            table(f'command delayed {lag} frames (val)', lag_baseline(val, lag))

        def cost(x):
            return float(evaluate(pool, to_params(x), train)[:, :5].mean())

        x0 = np.concatenate([np.log(START[:5]), START[6:]])
        step = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.03, 0.03])
        best_x, best_value = nelder_mead(cost, x0, step, args.max_evals)
        fitted = to_params(best_x)
        print('fitted', {k: float(f'{v:.4g}') for k, v in zip(NAMES, fitted)}, f'train {best_value:.3f} deg', flush=True)
        errors_val = evaluate(pool, fitted, val)
        table('fitted (train)', evaluate(pool, fitted, train))
        table('fitted (val)', errors_val)

        mjcf_delayed = MJCF_DEFAULT.copy()
        mjcf_delayed[6:] = fitted[6:]
        table('MJCF defaults + fitted lag (val)', evaluate(pool, mjcf_delayed, val))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    working = errors_val[:, 5] > 0.5
    errors_val = errors_val[:, :5]
    result = {
        'params': dict(zip(NAMES, fitted.tolist())), 'mjcf': extrinsics['mjcf'], 'joint_offset_deg': extrinsics['joint_offset_deg'],
        'joint_limits_rad': dict(zip(JOINTS, limits.tolist())),
        'train_episodes': train_ids, 'val_episodes': val_ids, 'dataset': args.dataset,
        'val_mae_deg_per_joint': dict(zip(JOINTS, errors_val.mean(0).tolist())), 'val_mae_deg': float(errors_val.mean()),
        'val_mae_deg_working': float(errors_val[working].mean()),
    }
    (out / 'actuators.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'wrote {out / "actuators.json"}')


if __name__ == '__main__':
    main()
