"""Scripted expert of the twin: picks up an elongated object across its narrowest place and puts it in the box.

    .venv/bin/python scripts/twin_expert.py --episodes 200                 # success by side and by shape family
    .venv/bin/python scripts/twin_expert.py --episodes 1 --seed 7 --view   # watch one in MuJoCo's viewer

The grasp rule to be learnt:
  * across: the jaws close across the object, at the place where it is narrowest within the jaws' finger width,
    not too far from its centre of mass (narrowest first, COM_WEIGHT trades the two);
  * the moving jaw on the far side, the fixed jaw towards the arm's base. With the gripper pointing down the jaws
    close along the gripper's x axis, the moving jaw coming from +x; of the two wrist roll angles that put the jaws
    across the object, the one turning +x away from the base is taken. On the left and the right of the arm this is a
    different roll, which the real demonstrations got wrong on one side.
The object's near side is put PUSH_M from the fixed jaw, so closing pushes it a little towards the base, onto the
fixed jaw. Then: lift, carry over the box, open, back to the start pose. Joint targets come from damped least squares
inverse kinematics on the twin's model and go through the identified servos (twin_scene.Scene.step) like a recorded
action would.

The object is a random twin_shapes.Shape (or the scanned key ring with --key); object and box are put at random on
the table, the object mostly within the demonstrations' area.
"""

import argparse
import copy
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

import twin_scene
import twin_shapes

FIXED_FACE_X = -0.0079  # inner face of the fixed jaw, gripper frame (jaws point along -z, the moving one opens to +x)
TIP_Z = -0.1044  # tip of the fixed jaw
FINGER_HALF_M = 0.005  # half the jaws' width at the tips
PUSH_M = 0.003
OPEN_MARGIN_M = 0.012  # room between the object's far side and the open moving jaw
MAX_WIDTH_M = 0.045
COM_WEIGHT = 0.3  # metres of width worth one metre of distance from the centre of mass
ANGLE_STEP = np.radians(5)
OFFSET_STEP_M = 0.002
APPROACH_M = 0.06
MIN_TIP_M = 0.001  # lowest the fixed jaw's tip goes over the table
LIFT_M = 0.08
DROP_ABOVE_RIM_M = 0.04
BOX_HEIGHT_M = 0.022
START = np.array([0.0, -103.6, 96.8, 60.0, 15.0, 1.0])  # dataset units: the rest pose the recordings start from
SPEED_RAD_S = 1.2
HOVER_FRAMES = 9
SAG_SETTLE_S = 0.4
SAG_ITERATIONS = 3
ROTATION_WEIGHT_M = 0.05
TILTS = np.radians([0, 5, -5, 10, 15, 20, 25, 30, 35, 40, 45, 50])  # tried in this order
# Start poses (dataset units) for --random-start: the rest pose of the teleoperated demos (pilot v2/v3 median), the
# raised pose v5 was tested from (experiment 5), START (the 2026-11-10 recordings); weight and spread per joint.
START_POSES = [
    (0.4, np.array([-2.0, -104.0, 97.0, 78.0, -94.0, 1.0]), np.array([4.0, 1.5, 2.0, 6.0, 20.0, 0.0])),
    (0.4, np.array([36.0, -73.0, 58.0, 79.0, -101.0, 1.0]), np.array([15.0, 10.0, 10.0, 10.0, 20.0, 0.0])),
    (0.2, START, np.array([4.0, 1.5, 2.0, 6.0, 20.0, 0.0])),
]
MISS_OFFSET_M = (0.012, 0.022)  # a first attempt shifted across the object by this much beyond the open jaws' reach
MISS_ROLL_DEG = (30.0, 45.0)  # or turned about the gripper axis by this much


@dataclass
class Grasp:
    centre: np.ndarray  # (x, y) in the object frame
    angle: float  # closing direction in the object frame
    width: float
    height: float  # of the object under the jaws


def clip_strip(polygon, direction, low, high):
    """The part of a convex polygon with low <= p . direction <= high (Sutherland-Hodgman, two half planes)."""
    for sign, bound in ((1, low), (-1, -high)):
        out = []
        for k in range(len(polygon)):
            a, b = polygon[k], polygon[(k + 1) % len(polygon)]
            da, db = sign * a @ direction - bound, sign * b @ direction - bound
            if da >= 0:
                out.append(a)
            if da * db < 0:
                out.append(a + (b - a) * da / (da - db))
        polygon = np.array(out)
        if len(polygon) == 0:
            return polygon
    return polygon


def plan_grasp(shape):
    """The narrowest crossing of the footprint by a strip as wide as the jaws, scored against the distance to the COM."""
    polygons = [twin_shapes.convex_hull_2d(p[:, :2]) for p in shape.parts]
    heights = [float(p[:, 2].max()) for p in shape.parts]
    com = shape.centre_of_mass()[:2]
    flat = np.vstack(polygons)
    best = None
    for angle in np.arange(0, np.pi, ANGLE_STEP):
        across = np.array([np.cos(angle), np.sin(angle)])  # the jaws close along this
        along = np.array([-across[1], across[0]])
        reach = flat @ along
        offsets = np.arange(reach.min() + FINGER_HALF_M, reach.max() - FINGER_HALF_M + 1e-9, OFFSET_STEP_M)
        for offset in offsets if len(offsets) else [reach.mean()]:
            spans, height = [], 0.0
            for polygon, h in zip(polygons, heights):
                cut = clip_strip(polygon, along, offset - FINGER_HALF_M, offset + FINGER_HALF_M)
                if len(cut):
                    spans.extend(cut @ across)
                    height = max(height, h)
            if not spans:
                continue
            width = max(spans) - min(spans)
            if width > MAX_WIDTH_M:
                continue
            centre = offset * along + (max(spans) + min(spans)) / 2 * across
            score = width + COM_WEIGHT * np.linalg.norm(centre - com)
            if best is None or score < best[0]:
                best = (score, Grasp(centre, angle, width, height))
    if best is None:
        raise ValueError('no crossing narrower than the jaws open')
    return best[1]


class Kinematics:
    """Inverse kinematics of the gripper body on the scene's model, and the jaw opening per jaw angle."""

    def __init__(self, scene):
        self.model, self.data = scene.model, mujoco.MjData(scene.model)
        self.gripper = scene.model.body('gripper').id
        self.free_model = copy.deepcopy(scene.model)
        self.free_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.free_data = mujoco.MjData(self.free_model)
        mujoco.mj_kinematics(self.model, self.data)
        self.pan_axis = self.data.xanchor[scene.model.joint('shoulder_pan').id].copy()
        self.low, self.high = scene.model.jnt_range[:5, 0], scene.model.jnt_range[:5, 1]
        pad = scene.model.geom('moving_finger_pad').id
        self.jaw_angles = np.linspace(scene.jaw_low, scene.jaw_high, 60)
        faces = []
        for q in self.jaw_angles:
            self.data.qpos[5] = q
            mujoco.mj_kinematics(self.model, self.data)
            local = self.data.xmat[self.gripper].reshape(3, 3).T @ (self.data.geom_xpos[pad] - self.data.xpos[self.gripper])
            faces.append(local[0] - 0.0009)  # pad centre to its inner face
        self.faces = np.array(faces)

    def jaw_for_face(self, x):
        """Jaw angle that puts the moving jaw's inner face at gripper x."""
        return float(np.interp(x, self.faces, self.jaw_angles))

    def solve(self, start, held, target, up, across=None, tilt=0.0, iterations=200):
        """Arm joints (5) putting the gripper-frame point held at target, with the gripper's x axis (the jaws' closing
        direction) square to the object's long side (up x across) when across is given, and the gripper tilted by
        tilt from pointing straight down, its tips away from the arm for tilt > 0.
        Returns (joints, position error m, misalignment rad, tilt error rad)."""
        d, q = self.data, np.array(start[:5], float)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        long_side = None if across is None else np.cross(up, across) / np.linalg.norm(np.cross(up, across))
        outward = target - self.pan_axis
        outward -= (outward @ up) * up
        outward /= np.linalg.norm(outward)
        for _ in range(iterations):
            d.qpos[:5] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            rotation = d.xmat[self.gripper].reshape(3, 3)
            point = d.xpos[self.gripper] + rotation @ held
            mujoco.mj_jac(self.model, d, jacp, jacr, point, self.gripper)
            z = rotation[:, 2]
            rows = [jacp[:, :5], ROTATION_WEIGHT_M * (np.cross(z, outward) @ jacr[:, :5])[None]]
            error = [target - point, [ROTATION_WEIGHT_M * (-np.sin(tilt) - z @ outward)]]
            if long_side is not None:
                x = rotation[:, 0]
                rows.append(ROTATION_WEIGHT_M * (np.cross(x, long_side) @ jacr[:, :5])[None])
                error.append([-ROTATION_WEIGHT_M * (x @ long_side)])
            jacobian, e = np.vstack(rows), np.concatenate(error)
            if np.linalg.norm(e) < 1e-5:
                break
            free = np.ones(5, bool)
            for _ in range(5):  # joints at a limit that the step pushes further are left out
                j = jacobian * free
                step = j.T @ np.linalg.solve(j @ j.T + 1e-7 * np.eye(len(j)), e)
                blocked = free & (((q <= self.low + 1e-4) & (step < 0)) | ((q >= self.high - 1e-4) & (step > 0)))
                if not blocked.any():
                    break
                free &= ~blocked
            q = np.clip(q + np.clip(step, -0.1, 0.1), self.low, self.high)
        d.qpos[:5] = q
        mujoco.mj_kinematics(self.model, d)
        rotation = d.xmat[self.gripper].reshape(3, 3)
        position = float(np.linalg.norm(target - d.xpos[self.gripper] - rotation @ held))
        misaligned = 0.0 if long_side is None else float(abs(np.arcsin(np.clip(rotation[:, 0] @ long_side, -1, 1))))
        tilted = float(abs(np.arcsin(np.clip(-rotation[:, 2] @ outward, -1, 1)) - tilt))
        return q, position, misaligned, tilted

    def solve_tilted(self, start, held, target, up, across=None):
        """solve() at the smallest tilt that reaches; returns (joints, tilt) or None."""
        guess = np.array(start[:5], float)
        for tilt in TILTS:
            q, position, misaligned, tilted = self.solve(guess, held, target, up, across, tilt)
            if position < 0.001 and misaligned < np.radians(2) and tilted < np.radians(2):
                return q, tilt
        return None

    def compensate(self, q):
        """The command (6) that brings the arm to rest at q (6) with no contact: the servos' P control stops short
        under gravity, by up to about a centimetre at the jaws; the leader arm a person moves is offset the same way."""
        m, d = self.free_model, self.free_data
        command = np.array(q, float)
        for _ in range(SAG_ITERATIONS):
            mujoco.mj_resetData(m, d)
            d.qpos[:6], d.ctrl[:6] = q, command
            for _ in range(int(SAG_SETTLE_S / m.opt.timestep)):
                mujoco.mj_step(m, d)
            command[:5] += q[:5] - d.qpos[:5]
        return command

    def pose(self, q):
        self.data.qpos[:5] = q[:5]
        mujoco.mj_kinematics(self.model, self.data)
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = self.data.xmat[self.gripper].reshape(3, 3), self.data.xpos[self.gripper]
        return out


def minimum_jerk(a, b, frames):
    s = np.linspace(0, 1, max(frames, 1) + 1)[1:, None]
    return a + (b - a) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


class Failed(Exception):
    pass


def plan_motion(scene, kinematics, grasp, start, shift=np.zeros(3), approach_s=1.2, home=None):
    """Joint targets (rad, 6 per frame) for the whole pick and place, from the scene's current object poses.
    shift is added to the targets around the object (a deliberate miss, see plan_miss). The arm returns to ``home``
    (default ``start``)."""
    poses = scene.object_poses()
    obj, box = poses['object'], poses['box']
    up = scene.table_rotation[:, 2]
    centre = obj[:3, :3] @ np.array([*grasp.centre, 0]) + obj[:3, 3]
    across = obj[:3, :3] @ np.array([np.cos(grasp.angle), np.sin(grasp.angle), 0])
    across -= (across @ up) * up
    across /= np.linalg.norm(across)
    radial = np.array([centre[0], centre[1], 0]) / np.linalg.norm(centre[:2])
    if across @ radial < 0:  # the moving jaw (+x) goes on the far side
        across = -across
    held = np.array([FIXED_FACE_X + PUSH_M + grasp.width / 2, 0, TIP_Z])  # the grasp centre in the gripper frame
    jaw_open = kinematics.jaw_for_face(FIXED_FACE_X + PUSH_M + grasp.width + OPEN_MARGIN_M)
    jaw_closed = scene.jaw_low
    a, b, c = scene.plane
    # the fixed jaw's tip (the moving one ends 1 mm lower when closed and may brush the table, as on the real arm)
    tips = np.array([[x, y, TIP_Z] for x in (FIXED_FACE_X, -0.015) for y in (-0.005, 0.005)])

    def clearance(q):
        pose = kinematics.pose(q)
        points = tips @ pose[:3, :3].T + pose[:3, 3]
        return float((points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c)).min())

    def solve(target, guess, what, along=True):
        guesses = [np.array(guess, float)]
        if along:  # other rolls, in case the first lands with the moving jaw on the near side
            for turn in (np.pi / 2, -np.pi / 2, np.pi):
                other = guesses[0].copy()
                other[4] = (other[4] + turn + np.pi) % (2 * np.pi) - np.pi
                guesses.append(other)
        for g in guesses:
            found = kinematics.solve_tilted(g, held, target, up, across if along else None)
            if found is None:
                continue
            if along and kinematics.pose(found[0])[:3, 0] @ across < 0:
                continue
            return found[0]
        raise Failed(f'ik {what}')

    tip = np.clip(grasp.height / 2 - 0.003, MIN_TIP_M, 0.012)  # the jaw tips' 6 mm around the object's middle
    at = solve(centre + up * tip, start, 'grasp')
    low = clearance(at)
    if low < MIN_TIP_M:  # a tilted gripper puts one jaw lower than the grasp height
        tip += MIN_TIP_M - low
        at = solve(centre + up * tip, at, 'grasp')
    if shift.any():
        at = solve(centre + up * tip + shift, at, 'grasp')
    above = solve(centre + up * (tip + APPROACH_M) + shift, at, 'above')
    down_path = [above]
    for k in (1, 2, 3):
        down_path.append(solve(centre + up * (tip + APPROACH_M * (1 - k / 4)) + shift, down_path[-1], 'down'))
    down_path.append(at)
    lift = solve(centre + up * (tip + LIFT_M) + shift, at, 'lift')
    # over the box: the held grasp centre above the box centre, the gripper turned as it was
    drop = box[:3, 3] + up * (BOX_HEIGHT_M + DROP_ABOVE_RIM_M + grasp.height)
    carry = solve(drop, lift, 'carry', along=False)
    info = {'roll_deg': float(np.degrees(at[4])), 'tilt_deg': float(np.degrees(np.arccos(kinematics.pose(at)[:3, 2] @ up))),
            'across': across.tolist(), 'hover': centre + up * (tip + APPROACH_M), 'held': held,
            'grasp_width': grasp.width, 'jaw_open': jaw_open}

    def frames_for(a, b, least):
        return int(max(least, np.abs(a - b).max() / SPEED_RAD_S) * 30)

    with_jaw = lambda q, jaw: np.append(q, jaw)
    path, phases = [], {}
    q0 = np.asarray(start, float)

    def go(target, least, name):
        a = path[-1] if path else q0
        path.extend(minimum_jerk(a, target, frames_for(a, target, least)))
        phases[name] = len(path)

    compensated = {}
    with_jaw_plain = with_jaw
    def with_jaw(q, jaw):  # commands for the arm compensated for the sag; the jaw as is
        key = tuple(np.round(q, 9))
        if key not in compensated:
            compensated[key] = kinematics.compensate(np.append(q, jaw))[:5]
        return with_jaw_plain(compensated[key], jaw)

    go(with_jaw(above, jaw_open), approach_s, 'approach')
    path.extend([path[-1]] * HOVER_FRAMES)  # the servos settle above the object
    for k, q in enumerate(down_path[1:]):
        go(with_jaw(q, jaw_open), 0.25, 'descend')
    path.extend([path[-1]] * 9)
    go(with_jaw(down_path[-1], jaw_closed), 0.6, 'close')
    path.extend([path[-1]] * 9)
    go(with_jaw(lift, jaw_closed), 0.8, 'lift')
    go(with_jaw(carry, jaw_closed), 1.2, 'carry')
    path.extend([path[-1]] * 6)
    go(with_jaw(carry, jaw_open), 0.4, 'open')
    path.extend([path[-1]] * 12)
    go(q0 if home is None else np.asarray(home, float), 1.5, 'return')
    return np.array(path), phases, info


def plan_miss(scene, kinematics, grasp, start, rng):
    """A first attempt that misses: the normal approach, either shifted across the object so that it ends up
    outside the jaws or with the wrist turned off the grasp angle; the jaws close, open again and the gripper rises
    back to the hover height. Returns (joint targets, phases, description)."""
    path, phases, info = plan_motion(scene, kinematics, grasp, start)
    across = np.array(info['across'])
    end = phases['close'] + 9  # closed and held
    if rng.random() < 0.6:
        side = rng.choice([-1.0, 1.0])
        offset = side * (grasp.width / 2 + rng.uniform(*MISS_OFFSET_M) + 0.01)
        path, phases, _ = plan_motion(scene, kinematics, grasp, start, shift=across * offset)
        what = {'miss': 'offset', 'offset_mm': float(offset * 1000)}
    else:
        turn = np.radians(rng.choice([-1.0, 1.0]) * rng.uniform(*MISS_ROLL_DEG))
        path = path.copy()
        ramp = np.clip(np.arange(len(path)) / max(phases['approach'], 1), 0, 1)
        path[:, 4] += turn * ramp  # the wrist turns off the grasp angle during the approach
        what = {'miss': 'roll', 'roll_error_deg': float(np.degrees(turn))}
    path = list(path[:end])
    opened = path[-1].copy()
    opened[5] = info['jaw_open']
    path.extend(minimum_jerk(path[-1], opened, 12))
    above = path[phases['approach'] - 1].copy()
    path.extend(minimum_jerk(path[-1], above, 18))
    path.extend([path[-1]] * HOVER_FRAMES)
    return np.array(path), {'miss_approach': phases['approach'], 'miss_close': phases['close'], 'miss_end': len(path)}, what


def sample_start(rng):
    """A start pose in dataset units near one of START_POSES."""
    weights = np.array([w for w, _, _ in START_POSES])
    _, centre, spread = START_POSES[rng.choice(len(START_POSES), p=weights / weights.sum())]
    return centre + rng.uniform(-1, 1, 6) * spread


def object_in_box(poses):
    local = np.linalg.inv(poses['box']) @ poses['object'][:, 3]
    return bool(abs(local[0]) < 0.045 and abs(local[1]) < 0.045 and -0.005 < local[2] < 0.03)


def sample_placements(rng, wide=False):
    """Object and box (x, y, yaw) on the table; ``wide`` also near the base and further out."""
    obj_x, box_x, near = ((0.11, 0.34), (0.16, 0.35), 0.12) if wide else ((0.14, 0.32), (0.20, 0.35), 0.15)
    while True:
        obj = (rng.uniform(*obj_x), rng.uniform(-0.22, 0.22), rng.uniform(-np.pi, np.pi))
        box = (rng.uniform(*box_x), rng.uniform(-0.18, 0.20), rng.uniform(-np.pi, np.pi))
        if np.hypot(obj[0] - box[0], obj[1] - box[1]) > 0.12 and np.hypot(*obj[:2]) > near and np.hypot(*box[:2]) > near:
            return {'object': obj, 'box': box}


def episode(job):
    """One pick and place; returns the outcome and, if asked, the frames.

    Options in args: ``random_start`` (start near one of START_POSES), ``regrasp`` (share of episodes whose first
    attempt misses on purpose, plan_miss, before the real grasp is planned from where the object then lies), ``wide``
    (sample_placements). They draw from a second generator of the seed, so episodes without them are unchanged."""
    seed, args = job
    rng = np.random.default_rng(seed)
    extra = np.random.default_rng([seed, 1])
    shape = twin_shapes.random_shape(rng, args.get('family'))
    obj = args['key'] if args.get('key') else shape
    scene = twin_scene.Scene(args['scan'], {'box': args['box'], 'object': obj})
    placements = sample_placements(extra if args.get('wide') else rng, wide=bool(args.get('wide')))
    start = sample_start(extra) if args.get('random_start') else START
    regrasp = extra.random() < args.get('regrasp', 0.0)
    scene.reset(start, placements)
    out = {'seed': seed, 'family': 'key' if args.get('key') else shape.family, 'placements': placements,
           'side': 'left' if placements['object'][1] > 0 else 'right', 'start': np.asarray(start).tolist(),
           'regrasp': bool(regrasp)}
    if scene.object_poses()['object'][2, 2] < np.cos(np.radians(15)):  # some shapes do not stand at every placement
        out.update(outcome='fell over when put down', success=False)
        return out
    actions, states, held, all_phases, heights = [], [], [], {}, {}

    def run(path, phases):
        offset = len(actions)
        all_phases.update({k: v + offset for k, v in phases.items()})
        for k, target in enumerate(path):
            actions.append(scene.from_rad(target))
            scene.step(actions[-1])
            states.append(scene.from_rad(scene.data.qpos[:6].copy()))
            if args.get('watch'):
                args['watch'](scene, offset + k, all_phases)
            if args.get('keep'):
                held.append(np.linalg.inv(scene.pose(scene.link_ids[5])) @ scene.object_poses()['object'])
            if offset + k + 1 == all_phases.get('lift'):
                heights['lift'] = scene.object_poses()['object'][2, 3]

    try:
        grasp = plan_grasp(key_shape(args['key']) if args.get('key') else shape)
        out['width_mm'] = grasp.width * 1000
        kinematics = Kinematics(scene)
        home = scene.to_rad(start)
        if regrasp:
            miss_path, miss_phases, what = plan_miss(scene, kinematics, grasp, home, extra)
            out.update(what)
            run(miss_path, miss_phases)
            if scene.object_poses()['object'][2, 2] < np.cos(np.radians(15)):
                raise Failed('object knocked over by the miss')
            path, phases, info = plan_motion(scene, kinematics, grasp, miss_path[-1], home=home)
        else:
            path, phases, info = plan_motion(scene, kinematics, grasp, home)
        out.update({k: v for k, v in info.items() if k not in ('hover', 'held', 'jaw_open', 'grasp_width')})
    except (Failed, ValueError) as error:
        out.update(outcome='no plan', reason=str(error), success=False)
        return out
    start_height = scene.object_poses()['object'][2, 3]
    run(path, phases)
    lifted = bool(heights['lift'] - start_height > 0.03)
    for _ in range(15):  # let it settle
        scene.step(actions[-1])
    actions = np.array(actions)
    success = object_in_box(scene.object_poses())
    out.update(success=success, lifted=lifted, outcome='in box' if success else ('dropped' if lifted else 'not lifted'),
               frames=len(actions), phases=all_phases)
    if args.get('keep'):
        out['actions'], out['states'], out['held'] = actions, np.array(states), np.array(held)
    return out


def key_shape(directory):
    """A scanned object as a Shape for grasp planning: its collision pieces in the object frame."""
    pieces = twin_scene.object_pieces(directory)
    return twin_shapes.Shape('key', pieces, twin_scene.KEY_MASS_KG, 1.0)


def view(job):
    """Plays one episode in MuJoCo's viewer, over and over."""
    import mujoco.viewer

    seed, args = job
    result = episode((seed, {**args, 'keep': True}))
    print({k: v for k, v in result.items() if k not in ('actions', 'states')})
    if 'actions' not in result:
        return
    rng = np.random.default_rng(seed)
    shape = twin_shapes.random_shape(rng, args.get('family'))
    scene = twin_scene.Scene(args['scan'], {'box': args['box'], 'object': args['key'] or shape})
    placements = sample_placements(rng)
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        viewer.opt.geomgroup[3] = 1
        viewer.cam.lookat[:] = [0.25, 0.0, 0.0]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.7, 180, -40
        while viewer.is_running():
            scene.reset(START, placements)
            for action in result['actions']:
                started = time.time()
                with viewer.lock():
                    scene.step(action)
                viewer.sync()
                if not viewer.is_running():
                    return
                time.sleep(max(0.0, twin_scene.FRAME_S - (time.time() - started)))
            time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--family', choices=twin_shapes.FAMILIES)
    parser.add_argument('--key', help='use this scanned object (e.g. runs/scan/key_tied) instead of random shapes')
    parser.add_argument('--scan', default='runs/scan/table_curtain')
    parser.add_argument('--box', default='runs/scan/box')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--view', action='store_true', help='watch the first episode in the viewer')
    parser.add_argument('--out', help='json file for the per-episode outcomes')
    parser.add_argument('--random-start', action='store_true', help='start near one of START_POSES')
    parser.add_argument('--regrasp', type=float, default=0.0, help='share of episodes whose first attempt misses')
    parser.add_argument('--wide', action='store_true', help='placements also near the base and further out')
    args = parser.parse_args()
    settings = {k: getattr(args, k) for k in ('family', 'key', 'scan', 'box', 'random_start', 'regrasp', 'wide')}
    if args.view:
        return view((args.seed, settings))

    jobs = [(args.seed + k, settings) for k in range(args.episodes)]
    started = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(episode, jobs, chunksize=1)
    print(f'{len(results)} episodes in {time.time() - started:.0f} s')
    groups = defaultdict(list)
    for r in results:
        groups[('side', r['side'])].append(r)
        groups[('family', r['family'])].append(r)
        groups[('all', '')].append(r)
        groups[('regrasp', r.get('miss', 'no') if r.get('regrasp') else 'no')].append(r)
        if r.get('start'):
            groups[('start', 'raised' if r['start'][1] > -90 else 'rest')].append(r)
    for (kind, name), rows in sorted(groups.items()):
        outcomes = defaultdict(int)
        for r in rows:
            outcomes[r['outcome']] += 1
        print(f'{kind:6s} {name:9s} {np.mean([r["success"] for r in rows]):4.0%} of {len(rows):3d}   ' +
              ', '.join(f'{k} {v}' for k, v in sorted(outcomes.items())))
    reasons = defaultdict(int)
    for r in results:
        if r['outcome'] == 'no plan':
            reasons[r['reason'].split(':')[0]] += 1
    if reasons:
        print('no plan:', dict(reasons))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
