"""Top camera to arm base extrinsics, by touching known points with the fixed jaw tip.

The tablet lies in the workspace showing artifacts/calibration/charuco_7x5_dict6x6_250_2880x1800_targets.png. The top
camera gives the board pose, so each crosshair (A-D, in the white margins) has a known position in the camera frame.
Touching it with the fixed jaw tip gives the same point in the arm base frame through forward kinematics. Several
tablet placements spread over the workspace give the rigid transform, and also show how good the kinematics are.

Nothing here commands the motors: joints are only read, with torque off, and the arm is moved by hand.

    source scripts/so101_env.fish
    python scripts/calib_top_extrinsics.py collect --camera $TOP --port $FOLLOWER \
        --intrinsics runs/calib/top/intrinsics_standard.json --out runs/calib/top_extrinsics
    python scripts/calib_top_extrinsics.py solve --out runs/calib/top_extrinsics --square-mm <7-square width / 7>
    python scripts/calib_top_extrinsics.py check --camera $TOP --port $FOLLOWER --out runs/calib/top_extrinsics

Joint angles are LeRobot degrees (zero at the middle of the calibrated range), the same values stored in the datasets,
so the fitted offsets convert dataset states to MJCF joint angles: q_mjcf_deg = q_lerobot_deg + offset_deg.
"""

import argparse
import json
import os
import select
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np

from calib_intrinsics import PREVIEW_COMMAND

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
ALL_MOTORS = JOINTS + ['gripper']
BOARD_SQUARES = (7, 5)
SQUARE_PX, MARKER_PX = 340, 256
BOARD_LEFT_PX, BOARD_TOP_PX = 250, 50
TARGETS_PX = {'A': (125, 300), 'B': (2755, 300), 'C': (2755, 1500), 'D': (125, 1500)}
N_CORNERS = (BOARD_SQUARES[0] - 1) * (BOARD_SQUARES[1] - 1)
MIN_CORNERS = 16
# Sharp tip of the fixed jaw in the MJCF `gripper` body frame (lowest vertex of wrist_roll_follower_so101_v1).
TIP_IN_GRIPPER = np.array([-0.0089, -0.0002, -0.1044])
DEFAULT_MJCF = Path(os.environ.get('SO101_NEXUS_SOURCE', '/home/loser/Study/so101-nexus')) / \
    'src/so101_nexus/assets/SO101/so101_new_calib.xml'
TICKS_PER_TURN = 4095


def make_board(square_mm=1.0):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(BOARD_SQUARES, square_mm, square_mm * MARKER_PX / SQUARE_PX, dictionary)
    return board, cv2.aruco.CharucoDetector(board)


def load_calibration():
    files = list((Path.home() / '.cache/huggingface/lerobot/calibration/robots').glob('*/follower.json'))
    if len(files) != 1:
        raise SystemExit(f'expected one follower.json, found {files}')
    return json.loads(files[0].read_text())


def raw_to_lerobot_deg(raw, calibration):
    """Same formula as LeRobot's DEGREES normalization; returns the five arm joints."""
    out = []
    for name in JOINTS:
        mid = (calibration[name]['range_min'] + calibration[name]['range_max']) / 2
        out.append((raw[name] - mid) * 360 / TICKS_PER_TURN)
    return np.array(out)


class Arm:
    """Read-only access to the follower: no torque, position or configuration register is written."""

    def __init__(self, port):
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        motors = {n: Motor(i + 1, 'sts3215', MotorNormMode.DEGREES) for i, n in enumerate(ALL_MOTORS)}
        self.bus = FeetechMotorsBus(port=port, motors=motors)
        self.bus.connect()
        torque = self.bus.sync_read('Torque_Enable', normalize=False)
        if any(torque.values()):
            self.close()
            raise SystemExit(f'torque is on ({torque}); this script never changes it. Power-cycle the arm while '
                             'holding it, then run again.')

    def read(self, samples=8):
        """Median raw position over a short window; warns if the arm was moving."""
        rows = []
        for _ in range(samples):
            positions = self.bus.sync_read('Present_Position', normalize=False)
            rows.append([positions[n] for n in ALL_MOTORS])
            time.sleep(0.03)
        rows = np.array(rows)
        spread = int((rows.max(0) - rows.min(0)).max())
        if spread > 4:
            print(f'  warning: arm moved during the reading ({spread} ticks); keep it still')
        return {n: float(v) for n, v in zip(ALL_MOTORS, np.median(rows, 0))}

    def read_once(self):
        positions = self.bus.sync_read('Present_Position', normalize=False)
        return np.array([positions[n] for n in ALL_MOTORS], float)

    def close(self):
        self.bus.disconnect(disable_torque=False)


def open_camera(path):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 31)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {path}')
    return cap


def grab(cap, drain=8):
    frame = None
    for _ in range(drain):  # drop buffered frames so the image is current
        ok, frame = cap.read()
        if not ok:
            raise SystemExit('camera read failed')
    return frame


def observe_board(cap, detector, frames=15):
    """Mean pixel position of each board corner seen in most of the frames."""
    seen = {}
    for _ in range(frames):
        corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(grab(cap, 2), cv2.COLOR_BGR2GRAY))
        if ids is not None:
            for i, c in zip(ids.ravel(), corners.reshape(-1, 2)):
                seen.setdefault(int(i), []).append(c)
    return {i: np.mean(c, 0).tolist() for i, c in seen.items() if len(c) > frames / 2}


def poll_line():
    """A line typed in the terminal, or None if there is none yet; never blocks."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().lower()
    return None


def target_pixels(board):
    """Where the crosshairs appear in the image, from the locked board view (homography on the board plane)."""
    template = make_board(float(SQUARE_PX))[0].getChessboardCorners()
    ids = sorted(int(i) for i in board)
    H, _ = cv2.findHomography(template[ids][:, :2], np.array([board[i] for i in ids]))
    points = np.array([[(u - BOARD_LEFT_PX, v - BOARD_TOP_PX)] for u, v in TARGETS_PX.values()], np.float64)
    return {n: p for n, p in zip(TARGETS_PX, cv2.perspectiveTransform(points, H).reshape(-1, 2))}


def render(frame, corners, title, detail, target=None):
    """Preview image: corners seen now (green), the crosshair to touch (yellow ring), and the current instruction."""
    view = frame.copy()
    for u, v in corners.values():
        cv2.circle(view, (round(float(u)), round(float(v))), 5, (0, 255, 0), 2)
    if target is not None:
        centre = (round(float(target[0])), round(float(target[1])))
        cv2.circle(view, centre, 30, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(view, centre, 3, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(view, (0, 0), (view.shape[1], 92), (0, 0, 0), -1)
    cv2.putText(view, title, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(view, detail, (14, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def collect(args):
    """Driven from the preview window. Keys (type in the terminal, then Enter) are only needed while the hands are free;
    a touch is recorded by itself once the arm has moved to the target and stays still."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'touches.json'
    data = json.loads(path.read_text()) if path.exists() else {'intrinsics': args.intrinsics, 'placements': []}
    _, detector = make_board()
    cap = open_camera(args.camera)
    arm = Arm(args.port)
    preview = subprocess.Popen(PREVIEW_COMMAND, stdin=subprocess.PIPE)
    names = list(TARGETS_PX)
    state, board, touches, index, pixels, reference = 'place', None, {}, 0, {}, None
    window, held, still_since, flash, flash_until = deque(), [], None, '', 0.0
    print('Follow the preview window. Open the jaw a little so the fixed jaw tip touches first; touch lightly.')
    print('Terminal keys, followed by Enter: (empty) confirm step, u undo last touch, s skip target, q finish.')
    try:
        while True:
            frame = grab(cap, 1)
            found, ids = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[:2]
            corners = {} if ids is None else {int(i): c for i, c in zip(ids.ravel(), found.reshape(-1, 2))}
            raw_now, now, line = arm.read_once(), time.time(), poll_line()

            # Stillness: arm joints within a few ticks over the last 0.4 s.
            window.append((now, raw_now))
            while now - window[0][0] > 0.4:
                window.popleft()
            recent = np.array([r[:5] for _, r in window])
            if (recent.max(0) - recent.min(0)).max() <= args.still_ticks and now - window[0][0] > 0.2:
                still_since = still_since or now
                held.append(raw_now)
            else:
                still_since, held = None, []
            held_s = now - still_since if still_since else 0.0

            target = None
            if state == 'place':
                title = f'Placement {len(data["placements"]) + 1}: put tablet in a new area, arm clear'
                detail = f'{len(corners)}/{N_CORNERS} corners, need {MIN_CORNERS}. Enter = lock, q = finish'
                if line == 'q':
                    break
                if line == '':
                    locked = observe_board(cap, detector)
                    if len(locked) < MIN_CORNERS:
                        flash, flash_until = f'only {len(locked)} steady corners, adjust and retry', time.time() + 3.0
                        print(f'  {flash}', flush=True)
                    else:
                        board, touches, index = locked, {}, 0
                        pixels, reference = target_pixels(board), raw_now.copy()
                        still_since, held, state = None, [], 'touch'
                        print(f'placement {len(data["placements"]) + 1}: board locked, {len(locked)} corners', flush=True)
            elif state == 'touch':
                name = names[index]
                target = pixels[name]
                moved = reference is None or np.abs(raw_now[:5] - reference[:5]).max() >= args.min_move_ticks
                title = f'Touch {name} with the fixed jaw tip'
                detail = (f'hold still {min(held_s, args.hold_s):.1f}/{args.hold_s:.1f} s' if moved
                          else 'move the tip to the yellow ring') + '   (u = undo, s = skip)'
                if line == 'u' and index > 0:
                    index -= 1
                    touches.pop(names[index], None)
                    reference, still_since, held = None, None, []
                    print(f'  undid {names[index]}', flush=True)
                elif line == 's':
                    print(f'  skipped {name}', flush=True)
                    index += 1
                    reference, still_since, held = raw_now.copy(), None, []
                elif moved and held_s >= args.hold_s:
                    value = np.median(np.array(held), 0)
                    touches[name] = {n: float(v) for n, v in zip(ALL_MOTORS, value)}
                    flash, flash_until = f'SAVED {name}', now + 1.2
                    print(f'  saved {name}\a', flush=True)
                    index += 1
                    reference, still_since, held = value, None, []
                if index == len(names):
                    state = 'clear'
            else:
                title = 'Done. Move the arm clear of the tablet'
                detail = 'do not bump the tablet, then Enter here'
                if line == '':
                    after = observe_board(cap, detector)
                    common = board.keys() & after.keys()
                    shift = max((float(np.linalg.norm(np.subtract(board[i], after[i]))) for i in common),
                                default=float('inf'))
                    if len(common) < MIN_CORNERS or shift > args.max_shift_px:
                        message = f'discarded: tablet moved or hidden (shift {shift:.1f} px, {len(common)} corners)'
                    else:
                        data['placements'].append({'board': board, 'touches': touches})
                        path.write_text(json.dumps(data, indent=2) + '\n')
                        message = f'saved placement {len(data["placements"])}, tablet shift {shift:.2f} px'
                    print(f'  {message}', flush=True)
                    flash, flash_until, state = message, now + 3.0, 'place'

            if preview:
                shown = render(frame, corners, title, flash if now < flash_until else detail, target)
                try:
                    preview.stdin.write(cv2.imencode('.jpg', shown, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
                    preview.stdin.flush()
                except BrokenPipeError:  # window closed; the terminal still reports each step
                    preview = None
    finally:
        arm.close()
        cap.release()
        if preview:
            preview.stdin.close()
    print(f'{len(data["placements"])} placements in {path}')


class Kinematics:
    def __init__(self, mjcf):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)
        self.gripper = self.model.body('gripper').id

    def forward(self, q_deg, jaw_rad=0.0):
        self.data.qpos[:5] = np.radians(q_deg)
        self.data.qpos[5] = jaw_rad
        mujoco.mj_kinematics(self.model, self.data)

    def tip(self, q_deg):
        self.forward(q_deg)
        return self.data.xpos[self.gripper] + self.data.xmat[self.gripper].reshape(3, 3) @ TIP_IN_GRIPPER


def kabsch(base, cam):
    """R, t with cam = R base + t."""
    base_mean, cam_mean = base.mean(0), cam.mean(0)
    u, _, vt = np.linalg.svd((base - base_mean).T @ (cam - cam_mean))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    R = vt.T @ np.diag([1, 1, d]) @ u.T
    return R, cam_mean - R @ base_mean


def residuals(kin, params, q, cam):
    R = cv2.Rodrigues(params[:3])[0]
    base = np.array([kin.tip(row + params[6:]) for row in q])
    return ((R @ base.T).T + params[3:6] - cam).ravel()


def fit(kin, q, cam, with_offsets, prior_m_per_deg=2e-4):
    """Rigid transform, plus optionally one zero offset per joint with a weak prior (Levenberg-Marquardt)."""
    base = np.array([kin.tip(row) for row in q])
    R, t = kabsch(base, cam)
    params = np.concatenate([cv2.Rodrigues(R)[0].ravel(), t, np.zeros(5)])
    if not with_offsets:
        return params

    def cost(p):
        return np.concatenate([residuals(kin, p, q, cam), prior_m_per_deg * p[6:]])

    damping, current = 1e-3, cost(params)
    for _ in range(60):
        J = np.zeros((current.size, params.size))
        for j in range(params.size):
            step = 1e-3 if j >= 6 else 1e-5
            shifted = params.copy()
            shifted[j] += step
            J[:, j] = (cost(shifted) - current) / step
        A = J.T @ J
        delta = np.linalg.solve(A + damping * np.diag(np.diag(A) + 1e-12), -J.T @ current)
        trial = params + delta
        trial_cost = cost(trial)
        if trial_cost @ trial_cost < current @ current:
            params, current, damping = trial, trial_cost, max(damping * 0.5, 1e-9)
            if np.abs(delta).max() < 1e-7:
                break
        else:
            damping *= 4
    return params


def errors_mm(kin, params, q, cam):
    return np.linalg.norm(residuals(kin, params, q, cam).reshape(-1, 3), axis=1) * 1000


def solve(args):
    out = Path(args.out)
    data = json.loads((out / 'touches.json').read_text())
    intrinsics = json.loads(Path(data['intrinsics']).read_text())
    K, dist = np.array(intrinsics['K']), np.array(intrinsics['dist'])
    board, _ = make_board(args.square_mm)
    template = board.getChessboardCorners()
    calibration = load_calibration()
    kin = Kinematics(args.mjcf)

    q, cam, group, labels = [], [], [], []
    for index, placement in enumerate(data['placements']):
        if index + 1 in args.skip:
            continue
        ids = sorted(int(i) for i in placement['board'])
        pixels = np.array([placement['board'][str(i)] for i in ids], np.float32)
        ok, rvec, tvec = cv2.solvePnP(template[ids].astype(np.float32), pixels, K, dist)
        projected = cv2.projectPoints(template[ids].astype(np.float32), rvec, tvec, K, dist)[0].reshape(-1, 2)
        print(f'placement {index + 1}: board RMS {np.sqrt(((projected - pixels) ** 2).sum(1).mean()):.3f} px, '
              f'{len(placement["touches"])} touches')
        R = cv2.Rodrigues(rvec)[0]
        for name, raw in placement['touches'].items():
            u, v = TARGETS_PX[name]
            point_mm = np.array([(u - BOARD_LEFT_PX) / SQUARE_PX, (v - BOARD_TOP_PX) / SQUARE_PX, 0.0]) * args.square_mm
            cam.append((R @ point_mm + tvec.ravel()) / 1000)
            q.append(raw_to_lerobot_deg(raw, calibration))
            group.append(index)
            labels.append(f'{index + 1}{name}')
    q, cam, group = np.array(q), np.array(cam), np.array(group)
    if len(set(group.tolist())) < 2:
        raise SystemExit('need at least 2 placements with touches')

    def report(name, with_offsets):
        params = fit(kin, q, cam, with_offsets)
        err = errors_mm(kin, params, q, cam)
        held = []
        for g in sorted(set(group.tolist())):
            train = group != g
            held.append(errors_mm(kin, fit(kin, q[train], cam[train], with_offsets), q[~train], cam[~train]))
        held = np.concatenate(held)
        print(f'{name}: fit mean {err.mean():.1f} mm, max {err.max():.1f} mm; '
              f'leave-one-placement-out mean {held.mean():.1f} mm, max {held.max():.1f} mm')
        return params, err, held

    rigid, rigid_err, rigid_held = report('rigid only          ', False)
    offset, offset_err, offset_held = report('rigid + joint zeros ', True)
    use_offsets = offset_held.mean() < 0.8 * rigid_held.mean()
    chosen, err, held = (offset, offset_err, offset_held) if use_offsets else (rigid, rigid_err, rigid_held)
    print('joint zero offsets (deg):', dict(zip(JOINTS, np.round(offset[6:], 2).tolist())))
    print('per-point error (mm):', {l: round(float(e), 1) for l, e in zip(labels, err)})
    print(f'chosen: {"rigid + joint zeros" if use_offsets else "rigid only"}')

    R = cv2.Rodrigues(chosen[:3])[0]
    result = {
        'R_cam_base': R.tolist(), 't_cam_base_m': chosen[3:6].tolist(),
        'camera_position_in_base_m': (-R.T @ chosen[3:6]).tolist(),
        'joint_offset_deg': dict(zip(JOINTS, (chosen[6:] if use_offsets else np.zeros(5)).tolist())),
        'fitted_joint_offset_deg_for_reference': dict(zip(JOINTS, offset[6:].tolist())),
        'intrinsics': data['intrinsics'], 'square_mm': args.square_mm, 'mjcf': str(args.mjcf),
        'tip_in_gripper_m': TIP_IN_GRIPPER.tolist(), 'n_points': len(q), 'n_placements': len(set(group.tolist())),
        'fit_error_mm': {'mean': float(err.mean()), 'max': float(err.max())},
        'leave_one_placement_out_error_mm': {'mean': float(held.mean()), 'max': float(held.max())},
        'rigid_only_leave_one_out_mean_mm': float(rigid_held.mean()),
    }
    (out / 'extrinsics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'camera height above base plane {result["camera_position_in_base_m"][2] * 1000:.0f} mm; '
          f'wrote {out / "extrinsics.json"}')


def check(args):
    """Project the MJCF arm onto a live frame: the dots should sit on the real arm."""
    out = Path(args.out)
    result = json.loads((out / 'extrinsics.json').read_text())
    intrinsics = json.loads(Path(result['intrinsics']).read_text())
    K, dist = np.array(intrinsics['K']), np.array(intrinsics['dist'])
    R, t = np.array(result['R_cam_base']), np.array(result['t_cam_base_m'])
    offsets = np.array([result['joint_offset_deg'][j] for j in JOINTS])
    cap = open_camera(args.camera)
    arm = Arm(args.port)
    try:
        frame, raw = grab(cap, 15), arm.read()
    finally:
        arm.close()
        cap.release()

    kin = Kinematics(result['mjcf'])
    calibration = load_calibration()
    jaw = calibration['gripper']  # same linear map as recon_table.py, so a closed jaw is drawn closed
    low, high = kin.model.jnt_range[5]
    opening = (raw['gripper'] - jaw['range_min']) / (jaw['range_max'] - jaw['range_min'])
    kin.forward(raw_to_lerobot_deg(raw, calibration) + offsets, low + opening * (high - low))
    model, state = kin.model, kin.data
    colours = {}
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 2:
            continue
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count][::args.stride]
        world = (state.geom_xmat[geom].reshape(3, 3) @ vertices.T).T + state.geom_xpos[geom]
        points = cv2.projectPoints(world, cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1, 2)
        body = int(model.geom_bodyid[geom])
        colour = colours.setdefault(body, tuple(int(c) for c in cv2.applyColorMap(
            np.uint8([[(40 * len(colours) + 20) % 255]]), cv2.COLORMAP_HSV)[0, 0]))
        for u, v in points:
            if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                cv2.circle(frame, (int(u), int(v)), 1, colour, -1)
    cv2.imwrite(str(out / 'overlay.jpg'), frame)
    print(f'wrote {out / "overlay.jpg"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    c = commands.add_parser('collect')
    c.add_argument('--camera', required=True)
    c.add_argument('--port', required=True)
    c.add_argument('--intrinsics', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--max-shift-px', type=float, default=1.5, help='max board corner motion during the touches')
    c.add_argument('--hold-s', type=float, default=2.0, help='how long the arm must stay still to record a touch')
    c.add_argument('--still-ticks', type=float, default=4, help='max joint motion over 0.4 s to count as still')
    c.add_argument('--min-move-ticks', type=float, default=60, help='joint motion required since the previous touch')
    c.set_defaults(run=collect)

    s = commands.add_parser('solve')
    s.add_argument('--out', required=True)
    s.add_argument('--square-mm', type=float, required=True, help='measured: width of 7 squares divided by 7')
    s.add_argument('--mjcf', type=Path, default=DEFAULT_MJCF)
    s.add_argument('--skip', type=int, nargs='*', default=[], help='placement numbers (from 1) to leave out')
    s.set_defaults(run=solve)

    k = commands.add_parser('check')
    k.add_argument('--camera', required=True)
    k.add_argument('--port', required=True)
    k.add_argument('--out', required=True)
    k.add_argument('--stride', type=int, default=25, help='draw every Nth mesh vertex')
    k.set_defaults(run=check)

    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
