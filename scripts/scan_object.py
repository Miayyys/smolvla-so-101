"""Photograph an object lying on the table with the wrist camera while the script moves the arm (this one commands the motors).

The object is found in a top camera image against the empty-workspace backgrounds of the source scan, and its pixels are
projected onto the fitted table plane. The wrist camera then visits every pose of the source scan's plan from which the
object is fully in view, beside the jaws. The plan's moves are re-checked with the object as an obstacle (a box over its
outline, OBSTACLE_MARGIN_M wider and taller than the object, at least OBSTACLE_HEIGHT_M) and moves that would touch it are dropped. Exposure is
metered on the first views as in the workspace scan.

    source scripts/so101_env.fish
    python scripts/scan_object.py --top $TOP --wrist $WRIST --port $FOLLOWER --source runs/scan/table_curtain --out runs/scan/key --dry-run
    python scripts/scan_object.py --top $TOP --wrist $WRIST --port $FOLLOWER --source runs/scan/table_curtain --out runs/scan/key

The arm must be in its folded rest pose (it hides little of the table there). Stopping works as in scan_auto.py.
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import calib_top_extrinsics as top
import scan_auto as auto
import scan_wrist as scan

WRIST_INTRINSICS = 'runs/calib/wrist/intrinsics_standard.json'
DIFF_THRESHOLD = 40  # brightest channel difference from the gain-matched background
ARM_MARGIN_PX = 20
TABLE_X_M, TABLE_Y_M = (0.05, 0.39), (-0.30, 0.31)  # the reconstructed table (recon_gs.py) less 2 cm at its edges
VISIBLE_FRACTION = 0.9
EDGE_MARGIN_PX = 12
JAW_MARGIN_PX = 10
MIN_DEPTH_M = 0.05
OUTLINE_POINTS = 400
OBSTACLE_HEIGHT_M = 0.02  # the least; a taller object gets its height plus OBSTACLE_MARGIN_M
OBSTACLE_MARGIN_M = 0.015


def top_projection():
    extrinsics = json.loads(Path(scan.EXTRINSICS).read_text())
    intrinsics = json.loads(Path(extrinsics['intrinsics']).read_text())
    return (np.array(intrinsics['K']), np.array(intrinsics['dist']), np.array(extrinsics['R_cam_base']),
            np.array(extrinsics['t_cam_base_m']))


def on_plane(pixels, plane):
    """Base-frame points where the top camera rays through the pixels meet the plane z = a x + b y + c."""
    K, dist, R, t = top_projection()
    normal = cv2.undistortPoints(pixels.astype(np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rays = np.column_stack([normal, np.ones(len(normal))]) @ R
    origin = -R.T @ t
    up = np.array([-plane[0], -plane[1], 1.0])
    return origin + ((plane[2] - up @ origin) / (rays @ up))[:, None] * rays


def locate(device, source, rest):
    """Top image, object mask and the object's pixels on the table plane, with the arm at rest."""
    cap = top.open_camera(device)
    try:
        top.grab(cap, 60)  # the top camera needs a couple of seconds to adjust after opening
        image = np.mean([top.grab(cap, 1) for _ in range(15)], axis=0)
    finally:
        cap.release()
    background = np.median(np.stack([cv2.imread(f) for f in sorted(glob.glob(f'{source}/top_backgrounds/top_*.png'))[1:]]), 0)
    gain = np.median(image / np.maximum(background, 1), axis=(0, 1))
    changed = (np.abs(image - background * gain).max(2) > DIFF_THRESHOLD).astype(np.uint8)
    arm = cv2.resize(auto.arm_masks(rest[None])[0].astype(np.uint8), changed.shape[::-1], interpolation=cv2.INTER_NEAREST)
    changed[cv2.dilate(arm, np.ones((2 * ARM_MARGIN_PX + 1,) * 2, np.uint8)) > 0] = 0
    plane = np.load(Path(source) / 'gs/table.npz')['plane']
    ys, xs = np.nonzero(changed)
    where = on_plane(np.column_stack([xs, ys]), plane)
    off_table = (where[:, 0] < TABLE_X_M[0]) | (where[:, 0] > TABLE_X_M[1]) | \
                (where[:, 1] < TABLE_Y_M[0]) | (where[:, 1] > TABLE_Y_M[1])
    changed[ys[off_table], xs[off_table]] = 0
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
    if count < 2:
        raise SystemExit('nothing found on the table in the top image')
    largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    mask = labels == largest
    ys, xs = np.nonzero(mask)
    points = on_plane(np.column_stack([xs, ys]), plane)
    print(f'object: {mask.sum()} top pixels (next largest change {np.sort(stats[1:, cv2.CC_STAT_AREA])[-2:-1]}), '
          f'centre {np.round(points.mean(0) * 1000)} mm, extent {np.round((points.max(0) - points.min(0))[:2] * 1000)} mm',
          flush=True)
    return image.round().astype(np.uint8), mask, points


def visible_fraction(camera, q, outline, K, dist, jaws):
    pose = camera.from_mjcf_deg(q[:5] + camera.offsets)
    local = (outline - pose[:3, 3]) @ pose[:3, :3]
    if (local[:, 2] < MIN_DEPTH_M).any():
        return 0.0
    uv = cv2.projectPoints(local, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    inside = (uv[:, 0] > EDGE_MARGIN_PX) & (uv[:, 0] < scan.WIDTH - EDGE_MARGIN_PX) & \
             (uv[:, 1] > EDGE_MARGIN_PX) & (uv[:, 1] < scan.HEIGHT - EDGE_MARGIN_PX)
    pixels = uv[inside].astype(int)
    inside[np.flatnonzero(inside)[jaws[pixels[:, 1], pixels[:, 0]]]] = False
    return float(inside.mean())


def obstacle(points, height):
    tall = max(OBSTACLE_HEIGHT_M, height + OBSTACLE_MARGIN_M)
    low, high = points.min(0), points.max(0)
    centre = (low + high) / 2
    half = (high - low) / 2 + OBSTACLE_MARGIN_M
    base = points[:, 2].min() - 0.005
    centre[2], half[2] = base + (tall + 0.005) / 2, (tall + 0.005) / 2
    return centre, half


def select(args, points):
    """Plan nodes with the object in view, and the plan's moves that stay clear of it."""
    planned = json.loads((Path(args.source) / 'auto_plan.json').read_text())
    nodes, leaving = np.array(planned['nodes']), np.array(planned['exit_path'])
    checker = auto.Checker(scan.demonstrated_states(args.dataset), boxes=[obstacle(points, args.height)])
    if any(checker.contacts(q, boxes_only=True) for q in leaving):
        raise SystemExit('the way out of the rest pose passes over the object; move it further out')
    camera = scan.CameraPose()
    intrinsics = json.loads(Path(WRIST_INTRINSICS).read_text())
    K, dist = np.array(intrinsics['K']), np.ravel(intrinsics['dist'])
    jaws = cv2.imread(str(Path(args.source) / 'jaw_mask.png'), cv2.IMREAD_GRAYSCALE) > 0
    jaws = cv2.dilate(jaws.astype(np.uint8), np.ones((2 * JAW_MARGIN_PX + 1,) * 2, np.uint8)) > 0
    outline = points[np.random.default_rng(0).choice(len(points), min(OUTLINE_POINTS, len(points)), replace=False)]
    outline = np.vstack([outline, outline + [0, 0, args.height]])  # the top of a tall object must be in view too
    seen = [i for i in range(1, len(nodes)) if visible_fraction(camera, nodes[i], outline, K, dist, jaws) >= args.visible]
    clear = {i for i in range(len(nodes)) if i == 0 or checker.pose_ok(nodes[i])}
    adjacency = [[] for _ in nodes]
    for i, others in enumerate(planned['adjacency']):
        for j in others:
            if i < j and i in clear and j in clear and checker.move_ok(nodes[i], nodes[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)
    kept = sum(map(len, adjacency)) // 2
    total = sum(map(len, planned['adjacency'])) // 2
    route, unreachable = auto.tour(nodes, adjacency, seen)
    seconds = sum(auto.duration(nodes[a], nodes[b], args.speed) for a, b in zip(route, route[1:])) + 2.5 * len(seen)
    print(f'{len(seen)} poses see the object; {len(clear) - 1} of {len(nodes) - 1} poses and {kept} of {total} moves '
          f'stay clear of it; {len(seen) - len(unreachable)} reachable, about {seconds / 60:.1f} min', flush=True)
    return {'nodes': nodes.tolist(), 'adjacency': adjacency, 'exit_path': leaving.tolist()}, route, \
        [i for i in seen if i not in unreachable]


def capture(args):
    out = Path(args.out)
    path = out / 'wrist_views.json'  # the layout of the workspace scan, so recon_table.py transfer gives refined poses
    if path.exists():
        raise SystemExit(f'{path} exists; use another --out')
    rest = np.array(json.loads((Path(args.source) / 'auto_plan.json').read_text())['exit_path'][0])
    image, mask, points = locate(args.top, args.source, rest)
    planned, route, wanted = select(args, points)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / 'top.png'), image)
    cv2.imwrite(str(out / 'top_mask.png'), mask.astype(np.uint8) * 255)
    shutil.copy(Path(args.source) / 'jaw_mask.png', out / 'jaw_mask.png')
    centre, half = obstacle(points, args.height)
    data = {'object': {'name': args.name, 'centre_m': points.mean(0).tolist(), 'obstacle_centre_m': centre.tolist(),
                       'obstacle_half_m': half.tolist(), 'height_m': args.height}, 'views': {}}
    np.save(out / 'top_points.npy', points)
    path.write_text(json.dumps(data, indent=1) + '\n')
    if args.dry_run:
        print(f'dry run: object saved in {out}; the motors were not touched')
        return

    wanted, index = set(wanted), 0
    preview = scan.Preview()
    blank = np.full((scan.MAP_SIZE, scan.MAP_SIZE, 3), 30, np.uint8)
    cap = auto.LiveCamera(scan.open_wrist(args.wrist), preview, blank, ['starting'])
    metering = [n for n in route if n in wanted][:auto.METERING_VIEWS]

    def prepare(arm, goto):
        def views():
            for node in metering:
                goto(node)
                auto.settle(arm)
                scan.settled_frame(cap, 4)
                yield node

        cap.status(blank, ['metering exposure'])
        locked = scan.lock_camera(args.wrist, cap, views)
        if locked is None:
            return False
        data['locked'] = locked
        path.write_text(json.dumps(data, indent=1) + '\n')
        return True

    def visit(arm, node):
        nonlocal index
        if node not in wanted:
            return False
        wanted.discard(node)
        if not auto.settle(arm):
            print(f'  node {node}: the joints did not settle, saving anyway', flush=True)
        scan.settled_frame(cap, 4)
        frame = scan.average_frames(cap).round().astype(np.uint8)
        name = f'view_{index:04d}.png'
        cv2.imwrite(str(out / name), frame)
        data['views'][name] = arm.raw()
        path.write_text(json.dumps(data, indent=1) + '\n')
        index += 1
        print(f'[{index}/{index + len(wanted)}] {name}', flush=True)
        cap.status(blank, [f'{index}/{index + len(wanted)}'])
        return top.poll_line() == 'q'

    intro = f'{len(wanted)} views of the {args.name} at {args.speed:.0f} deg/s peak, after a short exposure metering pass.'
    try:
        auto.drive(args.port, args.speed, planned, route, visit, intro, prepare)
        print(f'{index} views in {out}.')
    finally:
        cap.release()
        preview.close()
        scan.set_controls(args.wrist, auto_exposure=3, white_balance_automatic=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--top', required=True)
    parser.add_argument('--wrist', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--source', required=True, help='workspace scan with auto_plan.json, top_backgrounds, gs/table.npz')
    parser.add_argument('--out', required=True)
    parser.add_argument('--name', default='key')
    parser.add_argument('--visible', type=float, default=VISIBLE_FRACTION, help='least fraction of the object in view')
    parser.add_argument('--height', type=float, default=0.005, help='object height in metres')
    parser.add_argument('--dataset', default=scan.DATASET)
    parser.add_argument('--speed', type=float, default=25.0, help='peak joint speed, degrees per second')
    parser.add_argument('--dry-run', action='store_true', help='find the object and plan, without moving the arm')
    capture(parser.parse_args())


if __name__ == '__main__':
    main()
