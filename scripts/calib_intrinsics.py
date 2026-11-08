"""Camera intrinsics from a 7x5 ChArUco board (DICT_6X6_250) shown on a screen.

No GUI (the OpenCV build is headless). `capture` watches the camera and saves a frame whenever the board is held still
in a pose that differs enough from the saved ones; the terminal shows what it sees. `solve` fits the intrinsics.

    source scripts/so101_env.fish
    python scripts/calib_intrinsics.py capture --camera $TOP --width 1280 --height 720 --fps 31 --out runs/calib/top
    python scripts/calib_intrinsics.py solve --views runs/calib/top --square-mm 28

Calibrate in the exact mode used for recording: intrinsics do not carry over to another resolution. The default marker
ratio is for artifacts/calibration/charuco_7x5_dict6x6_250_2880x1800.png (tablet, 340/256 px); the 2560x1600 laptop
image is 300/224 px, so pass --marker-ratio 0.74667 for it.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

BOARD_SQUARES = (7, 5)
N_CORNERS = (BOARD_SQUARES[0] - 1) * (BOARD_SQUARES[1] - 1)
TABLET_MARKER_RATIO = 256 / 340
# mpv reads the annotated frames as an MJPEG stream on stdin and shows each one as soon as it arrives.
# --no-terminal is required: otherwise mpv opens /dev/tty and takes the keys typed for this script.
PREVIEW_COMMAND = ['mpv', '--no-terminal', '--no-cache', '--untimed', '--no-correct-pts', '--profile=low-latency',
                   '--demuxer=lavf', '--demuxer-lavf-format=mjpeg', '--title=calibration', '--no-osc', '-']


def make_detector(square_mm=1.0, marker_ratio=TABLET_MARKER_RATIO):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(BOARD_SQUARES, square_mm, square_mm * marker_ratio, dictionary)
    return board, cv2.aruco.CharucoDetector(board)


def detect(detector, gray):
    """Return {corner id: (u, v)}; empty when the board is not found."""
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None:
        return {}
    return {int(i): c for i, c in zip(ids.ravel(), corners.reshape(-1, 2))}


def mean_shift(a, b):
    """Mean displacement over the corners seen in both views, or None if they share too few."""
    common = a.keys() & b.keys()
    if len(common) < 6:
        return None
    return float(np.mean([np.linalg.norm(a[i] - b[i]) for i in common]))


def coverage(views, size, cells=(6, 4)):
    """Corner count per image cell, as a one-line map: '.' none, '+' few, '#' many."""
    w, h = size
    grid = np.zeros((cells[1], cells[0]), int)
    for view in views:
        for u, v in view.values():
            grid[min(int(v / h * cells[1]), cells[1] - 1), min(int(u / w * cells[0]), cells[0] - 1)] += 1
    return grid, '/'.join(''.join('.' if n == 0 else '+' if n < 8 else '#' for n in row) for row in grid)


def annotate(frame, current, saved, text):
    """Copy of the frame for the preview only: corners seen now (green), corners already saved (orange), status."""
    view = frame.copy()
    for points in saved:
        for u, v in points.values():
            cv2.circle(view, (round(float(u)), round(float(v))), 2, (0, 140, 255), -1)
    for u, v in current.values():
        cv2.circle(view, (round(float(u)), round(float(v))), 6, (0, 255, 0), 2)
    cv2.putText(view, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(view, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return view


def capture(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob('view_*.png'))
    _, detector = make_detector(marker_ratio=args.marker_ratio)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {args.camera}')
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if size != (args.width, args.height):
        raise SystemExit(f'camera gave {size}, requested {(args.width, args.height)}')

    saved = [detect(detector, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in existing]
    index = len(existing)
    print(f'{args.camera}: {size[0]}x{size[1]}, {index} views already in {out}, target {args.target}. Ctrl-C to stop.',
          flush=True)

    preview = subprocess.Popen(PREVIEW_COMMAND, stdin=subprocess.PIPE) if args.preview else None
    previous, still_since, last_status = {}, None, 0.0
    while index < args.target:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit('camera read failed (USB disconnect?)')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = detect(detector, gray)
        shift = mean_shift(current, previous) if previous else None
        previous = current
        now = time.time()

        still = len(current) >= args.min_corners and shift is not None and shift < args.still_px
        still_since = (still_since or now) if still else None
        nearest = min((s for s in (mean_shift(current, v) for v in saved) if s is not None), default=None)
        novel = nearest is None or nearest / size[0] >= args.min_novelty

        if not current:
            state = 'no board visible'
        else:
            xy = np.array(list(current.values())) / size
            hint = 'too few corners' if len(current) < args.min_corners else \
                'moving' if not still else 'too close to a saved view, change pose' if not novel else 'holding'
            state = (f'{len(current)}/{N_CORNERS} corners at x {xy[:, 0].min():.2f}-{xy[:, 0].max():.2f} '
                     f'y {xy[:, 1].min():.2f}-{xy[:, 1].max():.2f}: {hint}')

        if still_since and now - still_since >= args.hold_s and novel:
            cv2.imwrite(str(out / f'view_{index:03d}.png'), frame)
            saved.append(current)
            index += 1
            still_since = None
            print(f'[{index}/{args.target}] saved, {len(current)}/{N_CORNERS} corners, '
                  f'coverage {coverage(saved, size)[1]}', flush=True)
        elif now - last_status > 2.0:
            last_status = now
            print(f'  {state}', flush=True)

        if preview:
            shown = annotate(frame, current, saved, f'{index}/{args.target}  {state.split(": ")[-1]}')
            try:
                preview.stdin.write(cv2.imencode('.jpg', shown, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
                preview.stdin.flush()
            except BrokenPipeError:  # preview window closed; keep capturing
                preview = None
    cap.release()
    if preview:
        preview.stdin.close()


def rms(errors):
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def view_errors(obj, img, rvec, tvec, K, dist):
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return np.linalg.norm(projected.reshape(-1, 2) - img.reshape(-1, 2), axis=1)


def fit(obj, img, size, flags):
    _, K, dist, rvecs, tvecs, stds, _, _ = cv2.calibrateCameraExtended(
        obj, img, size, None, None, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-9))
    errors = [view_errors(o, i, r, t, K, dist) for o, i, r, t in zip(obj, img, rvecs, tvecs)]
    return K, dist, stds.ravel(), errors


def solve(args):
    views = Path(args.views)
    board, detector = make_detector(args.square_mm, args.marker_ratio)
    template = board.getChessboardCorners()
    names, obj, img, detected, size = [], [], [], [], None
    for path in sorted(views.glob('view_*.png')):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        found = detect(detector, gray)
        if len(found) < args.min_corners:
            print(f'skip {path.name}: {len(found)} corners')
            continue
        ids = sorted(found)
        names.append(path.name)
        detected.append(found)
        obj.append(template[ids].astype(np.float32))
        img.append(np.array([found[i] for i in ids], np.float32))
    if len(names) < 10:
        raise SystemExit(f'only {len(names)} usable views; need at least 10')

    flags = cv2.CALIB_RATIONAL_MODEL if args.model == 'rational' else 0
    K, dist, stds, errors = fit(obj, img, size, flags)
    per_view = np.array([float(np.sqrt(np.mean(e ** 2))) for e in errors])
    limit = max(args.outlier_factor * float(np.median(per_view)), 0.5)
    keep = [i for i, e in enumerate(per_view) if e <= limit]
    dropped = [names[i] for i in range(len(names)) if i not in keep]
    if dropped:
        print(f'dropped {len(dropped)} views above {limit:.2f} px: {dropped}')
        names, obj, img, detected = ([x[i] for i in keep] for x in (names, obj, img, detected))
        K, dist, stds, errors = fit(obj, img, size, flags)
        per_view = np.array([float(np.sqrt(np.mean(e ** 2))) for e in errors])

    # Hold-out check: every 5th view is left out of the fit, then posed with the fitted intrinsics.
    held = list(range(0, len(names), 5))
    train = [i for i in range(len(names)) if i not in held]
    K_h, dist_h, _, _ = fit([obj[i] for i in train], [img[i] for i in train], size, flags)
    held_errors = []
    for i in held:
        _, rvec, tvec = cv2.solvePnP(obj[i], img[i], K_h, dist_h)
        held_errors.append(view_errors(obj[i], img[i], rvec, tvec, K_h, dist_h))

    grid, grid_text = coverage(detected, size)
    empty = int((grid == 0).sum())
    result = {
        'image_size': list(size),
        'model': args.model,
        'K': K.tolist(),
        'dist': dist.ravel().tolist(),
        'rms_px': rms(errors),
        'holdout_rms_px': rms(held_errors),
        'n_views': len(names),
        'n_views_held_out': len(held),
        'views': names,
        'std_fx_fy_cx_cy': stds[:4].tolist(),
        'board': {'squares': list(BOARD_SQUARES), 'dictionary': 'DICT_6X6_250', 'square_mm': args.square_mm,
                  'marker_ratio': args.marker_ratio},
        'empty_coverage_cells': empty,
    }
    (views / f'intrinsics_{args.model}.json').write_text(json.dumps(result, indent=2) + '\n')

    print(f'{len(names)} views, {size[0]}x{size[1]}, model {args.model}')
    print(f'fx {K[0, 0]:.2f} fy {K[1, 1]:.2f} cx {K[0, 2]:.2f} cy {K[1, 2]:.2f} '
          f'(std {stds[0]:.2f} {stds[1]:.2f} {stds[2]:.2f} {stds[3]:.2f})')
    print('dist', np.round(dist.ravel(), 5).tolist())
    print(f'RMS {rms(errors):.3f} px, hold-out RMS {rms(held_errors):.3f} px, worst view {per_view.max():.3f} px')
    print(f'coverage {grid_text} ({empty} of {grid.size} cells empty)')
    if empty:
        print('WARNING: some image regions have no corners; distortion there is extrapolated. Capture more near them.')
    print(f'wrote {views / f"intrinsics_{args.model}.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    c = commands.add_parser('capture')
    c.add_argument('--camera', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--width', type=int, required=True)
    c.add_argument('--height', type=int, required=True)
    c.add_argument('--fps', type=int, required=True)
    c.add_argument('--fourcc', default='MJPG')
    c.add_argument('--preview', action='store_true', help='show the annotated camera image in an mpv window')
    c.add_argument('--target', type=int, default=30)
    c.add_argument('--still-px', type=float, default=0.8, help='max mean corner motion between frames to count as still')
    c.add_argument('--hold-s', type=float, default=0.5, help='how long the board must stay still before a save')
    c.add_argument('--min-novelty', type=float, default=0.05,
                   help='min mean corner shift from the nearest saved view, as a fraction of image width')
    c.add_argument('--min-corners', type=int, default=12)
    c.add_argument('--marker-ratio', type=float, default=TABLET_MARKER_RATIO)
    c.set_defaults(run=capture)

    s = commands.add_parser('solve')
    s.add_argument('--views', required=True)
    s.add_argument('--square-mm', type=float, required=True, help='measured on the displayed board')
    s.add_argument('--model', choices=['standard', 'rational'], default='standard')
    s.add_argument('--outlier-factor', type=float, default=3.0)
    s.add_argument('--min-corners', type=int, default=12)
    s.add_argument('--marker-ratio', type=float, default=TABLET_MARKER_RATIO)
    s.set_defaults(run=solve)

    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
