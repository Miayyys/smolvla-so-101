"""Wrist camera intrinsics and hand-eye pose (camera relative to the MJCF `gripper` body).

The tablet stays flat on the table showing the ChArUco image, and the arm is moved by hand (torque off, nothing is
commanded). `capture` saves a wrist image with the joint readings whenever arm and board are still and the view is new.
Vary wrist_roll, wrist_flex, distance and side: hand-eye needs rotations about different axes. The top camera sees the
tablet before and after, which detects a bumped tablet and gives an independent check of the result.

    source scripts/so101_env.fish
    python scripts/calib_wrist.py capture --wrist $WRIST --top $TOP --port $FOLLOWER --out runs/calib/wrist
    python scripts/calib_wrist.py solve --out runs/calib/wrist --square-mm 27.9

Needs the top camera results in runs/calib/top and runs/calib/top_extrinsics_2. The result is in the OpenCV camera
convention (x right, y down, z forward).
"""

import argparse
import json
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import calib_intrinsics as intr
import calib_top_extrinsics as top

DEFAULT_EXTRINSICS = 'runs/calib/top_extrinsics_2/extrinsics.json'
WIDTH, HEIGHT, FPS = 640, 480, 30


def open_wrist(path):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {path}')
    return cap


def capture(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'poses.json'
    data = json.loads(path.read_text()) if path.exists() else {'views': {}}
    _, top_detector = top.make_board()
    _, detector = intr.make_detector()

    if 'top_before' not in data:
        cap_top = top.open_camera(args.top)
        data['top_before'] = top.observe_board(cap_top, top_detector)
        cap_top.release()
        if len(data['top_before']) < top.MIN_CORNERS:
            raise SystemExit(f'top camera sees only {len(data["top_before"])} corners; check the tablet')
        path.write_text(json.dumps(data, indent=1) + '\n')
    print(f'top camera locked the tablet, {len(data["top_before"])} corners. Do not move it until the end.', flush=True)

    size = (WIDTH, HEIGHT)
    saved = [intr.detect(detector, cv2.imread(str(out / name), cv2.IMREAD_GRAYSCALE)) for name in sorted(data['views'])]
    index = len(saved)
    cap = open_wrist(args.wrist)
    arm = top.Arm(args.port)
    preview = subprocess.Popen(intr.PREVIEW_COMMAND, stdin=subprocess.PIPE)
    joints, previous, still_since = deque(), {}, None
    last_status = 0.0
    print(f'{index} views so far, target {args.target}. Type q then Enter to stop early.', flush=True)
    try:
        while index < args.target:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit('wrist camera read failed (USB disconnect?); run again to continue')
            current = intr.detect(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            shift = intr.mean_shift(current, previous) if previous else None
            previous = current
            raw, now = arm.read_once(), time.time()
            joints.append((now, raw))
            while now - joints[0][0] > 0.4:
                joints.popleft()
            recent = np.array([r[:5] for _, r in joints])
            arm_still = (recent.max(0) - recent.min(0)).max() <= args.still_ticks and now - joints[0][0] > 0.2
            still = arm_still and len(current) >= args.min_corners and shift is not None and shift < args.still_px
            still_since = (still_since or now) if still else None
            nearest = min((s for s in (intr.mean_shift(current, v) for v in saved) if s is not None), default=None)
            novel = nearest is None or nearest / WIDTH >= args.min_novelty

            if not current:
                hint = 'no board visible'
            elif len(current) < args.min_corners:
                hint = f'{len(current)} corners, need {args.min_corners}'
            elif not still:
                hint = f'{len(current)} corners, moving'
            elif not novel:
                hint = 'too close to a saved view, change pose'
            else:
                hint = f'hold {now - still_since:.1f}/{args.hold_s:.1f} s'

            if still_since and novel and now - still_since >= args.hold_s:
                name = f'view_{index:03d}.png'
                cv2.imwrite(str(out / name), frame)
                data['views'][name] = {n: float(v) for n, v in zip(top.ALL_MOTORS, raw)}
                path.write_text(json.dumps(data, indent=1) + '\n')
                saved.append(current)
                index += 1
                still_since = None
                print(f'[{index}/{args.target}] saved, {len(current)} corners, roll '
                      f'{top.raw_to_lerobot_deg(data["views"][name], top.load_calibration())[4]:.0f} deg, '
                      f'coverage {intr.coverage(saved, size)[1]}', flush=True)
            elif now - last_status > 2.0:
                last_status = now
                print(f'  {hint}', flush=True)

            if top.poll_line() == 'q':
                break
            if preview:
                shown = intr.annotate(frame, current, saved, f'{index}/{args.target}  {hint}')
                try:
                    preview.stdin.write(cv2.imencode('.jpg', shown, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
                    preview.stdin.flush()
                except BrokenPipeError:
                    preview = None
    finally:
        arm.close()
        cap.release()
        if preview:
            preview.stdin.close()

    input('Move the arm out of the top camera view of the tablet, then press Enter: ')
    cap_top = top.open_camera(args.top)
    after = top.observe_board(cap_top, top_detector)
    cap_top.release()
    common = data['top_before'].keys() & after.keys()
    moved = max((float(np.linalg.norm(np.subtract(data['top_before'][i], after[i]))) for i in common), default=float('inf'))
    data['top_after'] = after
    path.write_text(json.dumps(data, indent=1) + '\n')
    print(f'{index} views. Tablet moved {moved:.2f} px in the top camera ({len(common)} corners)' +
          ('  WARNING: the tablet moved; do not use this session' if moved > 1.5 else ''))


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def from_vec(v):
    return to_T(cv2.Rodrigues(v[:3])[0], v[3:])


def to_vec(T):
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])


def rotation_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def project_error(X, B, base_gripper, obj, img, K, dist):
    """Pixel residuals of the board corners for each view, given hand-eye X and board pose B in the base frame."""
    residuals = []
    for G, o, i in zip(base_gripper, obj, img):
        T = np.linalg.inv(G @ X) @ B
        rvec = cv2.Rodrigues(T[:3, :3])[0]
        projected = cv2.projectPoints(o.astype(np.float64), rvec, T[:3, 3], K, dist)[0].reshape(-1, 2)
        residuals.append((projected - i).ravel())
    return np.concatenate(residuals)


def refine(X, B, base_gripper, obj, img, K, dist):
    """Levenberg-Marquardt on the hand-eye pose and the board pose, minimizing reprojection error."""
    params = np.concatenate([to_vec(X), to_vec(B)])

    def cost(p):
        return project_error(from_vec(p[:6]), from_vec(p[6:]), base_gripper, obj, img, K, dist)

    current, damping = cost(params), 1e-3
    for _ in range(60):
        J = np.zeros((current.size, params.size))
        for j in range(params.size):
            shifted = params.copy()
            shifted[j] += 1e-5
            J[:, j] = (cost(shifted) - current) / 1e-5
        A = J.T @ J
        delta = np.linalg.solve(A + damping * np.diag(np.diag(A) + 1e-12), -J.T @ current)
        trial = cost(params + delta)
        if trial @ trial < current @ current:
            params, current, damping = params + delta, trial, max(damping * 0.5, 1e-9)
            if np.abs(delta).max() < 1e-8:
                break
        else:
            damping *= 4
    return from_vec(params[:6]), from_vec(params[6:])


def rms(residuals):
    return float(np.sqrt(np.mean(residuals.reshape(-1, 2).__pow__(2).sum(1))))


def solve(args):
    out = Path(args.out)
    data = json.loads((out / 'poses.json').read_text())
    intr.solve(argparse.Namespace(views=str(out), square_mm=args.square_mm, model=args.model, outlier_factor=3.0,
                                  min_corners=args.min_corners, marker_ratio=intr.TABLET_MARKER_RATIO))
    intrinsics = json.loads((out / f'intrinsics_{args.model}.json').read_text())
    K, dist = np.array(intrinsics['K']), np.array(intrinsics['dist'])
    extrinsics = json.loads(Path(args.extrinsics).read_text())
    offsets = np.array([extrinsics['joint_offset_deg'][j] for j in top.JOINTS])
    calibration = top.load_calibration()
    kin = top.Kinematics(extrinsics['mjcf'])
    board, detector = intr.make_detector(args.square_mm / 1000)
    template = board.getChessboardCorners()

    obj, img, base_gripper, board_cam, names = [], [], [], [], []
    for name in intrinsics['views']:
        found = intr.detect(detector, cv2.imread(str(out / name), cv2.IMREAD_GRAYSCALE))
        ids = sorted(found)
        o = template[ids].astype(np.float32)
        i = np.array([found[n] for n in ids], np.float32)
        _, rvec, tvec = cv2.solvePnP(o, i, K, dist)
        kin.forward(top.raw_to_lerobot_deg(data['views'][name], calibration) + offsets)
        obj.append(o)
        img.append(i)
        base_gripper.append(to_T(kin.data.xmat[kin.gripper].reshape(3, 3), kin.data.xpos[kin.gripper]))
        board_cam.append(to_T(cv2.Rodrigues(rvec)[0], tvec))
        names.append(name)

    rotations = [G[:3, :3] for G in base_gripper]
    spans = np.array([cv2.Rodrigues(R @ rotations[0].T)[0].ravel() for R in rotations])
    print(f'gripper rotation span (deg, max about each base axis): {np.round(np.degrees(np.abs(spans).max(0)), 0).tolist()}')

    def closed_form(index):
        best = None
        for method, label in ((cv2.CALIB_HAND_EYE_TSAI, 'Tsai'), (cv2.CALIB_HAND_EYE_PARK, 'Park'),
                              (cv2.CALIB_HAND_EYE_HORAUD, 'Horaud'), (cv2.CALIB_HAND_EYE_DANIILIDIS, 'Daniilidis')):
            R, t = cv2.calibrateHandEye([base_gripper[i][:3, :3] for i in index], [base_gripper[i][:3, 3] for i in index],
                                        [board_cam[i][:3, :3] for i in index], [board_cam[i][:3, 3] for i in index],
                                        method=method)
            X = to_T(R, t)
            boards = np.array([base_gripper[i] @ X @ board_cam[i] for i in index])
            spread = float(np.linalg.norm(boards[:, :3, 3] - boards[:, :3, 3].mean(0), axis=1).mean())
            if best is None or spread < best[0]:
                best = spread, label, X, boards
        B = best[3][np.argmin(np.linalg.norm(best[3][:, :3, 3] - np.median(best[3][:, :3, 3], 0), axis=1))]
        return best[0], best[1], best[2], B

    def subset(index):
        return [base_gripper[i] for i in index], [obj[i] for i in index], [img[i] for i in index]

    all_views = list(range(len(names)))
    spread, label, X0, B0 = closed_form(all_views)
    X, B = refine(X0, B0, *subset(all_views), K, dist)
    residual = project_error(X, B, *subset(all_views), K, dist)

    held = all_views[::5]
    train = [i for i in all_views if i not in held]
    _, _, X_t, B_t = closed_form(train)
    X_t, B_t = refine(X_t, B_t, *subset(train), K, dist)
    held_residual = project_error(X_t, B_t, *subset(held), K, dist)

    top_intrinsics = json.loads(Path(extrinsics['intrinsics']).read_text())
    top_board, _ = top.make_board(args.square_mm / 1000)
    corners = top_board.getChessboardCorners()
    ids = sorted(int(i) for i in data['top_before'])
    _, rvec, tvec = cv2.solvePnP(corners[ids].astype(np.float32),
                                 np.array([data['top_before'][str(i)] for i in ids], np.float32),
                                 np.array(top_intrinsics['K']), np.array(top_intrinsics['dist']))
    top_base = to_T(np.array(extrinsics['R_cam_base']), extrinsics['t_cam_base_m'])
    B_top = np.linalg.inv(top_base) @ to_T(cv2.Rodrigues(rvec)[0], tvec)
    cross_mm = float(np.linalg.norm(B[:3, 3] - B_top[:3, 3]) * 1000)
    cross_deg = rotation_deg(B[:3, :3].T @ B_top[:3, :3])
    top_residual = project_error(X, B_top, *subset(all_views), K, dist)

    print(f'closed form {label}: board spread {spread * 1000:.1f} mm')
    print(f'reprojection after refinement: RMS {rms(residual):.2f} px; every 5th view held out: {rms(held_residual):.2f} px')
    print(f'board pose vs top camera: {cross_mm:.1f} mm, {cross_deg:.1f} deg; wrist reprojection using the top-camera '
          f'board pose: RMS {rms(top_residual):.2f} px')
    print('camera in gripper body (mm):', np.round(X[:3, 3] * 1000, 1).tolist())
    result = {
        'T_gripper_cam': X.tolist(), 'convention': 'OpenCV camera; gripper is the MJCF `gripper` body',
        'T_base_board_wrist': B.tolist(), 'T_base_board_top': B_top.tolist(),
        'reprojection_rms_px': rms(residual), 'held_out_rms_px': rms(held_residual),
        'top_camera_board_error_mm': cross_mm, 'top_camera_board_error_deg': cross_deg,
        'top_board_reprojection_rms_px': rms(top_residual), 'n_views': len(names), 'intrinsics': str(out / f'intrinsics_{args.model}.json'),
        'extrinsics': args.extrinsics, 'square_mm': args.square_mm,
    }
    (out / 'handeye.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'wrote {out / "handeye.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    c = commands.add_parser('capture')
    c.add_argument('--wrist', required=True)
    c.add_argument('--top', required=True)
    c.add_argument('--port', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--target', type=int, default=30)
    c.add_argument('--hold-s', type=float, default=0.7)
    c.add_argument('--still-px', type=float, default=0.8)
    c.add_argument('--still-ticks', type=float, default=4)
    c.add_argument('--min-novelty', type=float, default=0.05)
    c.add_argument('--min-corners', type=int, default=12)
    c.set_defaults(run=capture)

    s = commands.add_parser('solve')
    s.add_argument('--out', required=True)
    s.add_argument('--square-mm', type=float, required=True)
    s.add_argument('--extrinsics', default=DEFAULT_EXTRINSICS)
    s.add_argument('--model', choices=['standard', 'rational'], default='standard')
    s.add_argument('--min-corners', type=int, default=12)
    s.set_defaults(run=solve)

    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
