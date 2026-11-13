"""Replay recorded episodes in the digital twin and compare with what the cameras recorded. Runs in .venv-gs.

    .venv-gs/bin/python scripts/twin_replay.py --dataset artifacts/datasets/so101_twin_key_day \
        --scan runs/scan/table_curtain --box runs/scan/box --key runs/scan/key --out runs/twin/replay_day

For each episode the box and the key ring are located in its first top image (the table must be unobstructed there),
the recorded actions are replayed open loop in MuJoCo (twin_scene.py) from the first recorded joint state, and both
camera images are rendered each frame (twin_render.py). Written per episode: the simulated and recorded joints, where
the key ring ends up, image differences, and a side-by-side video. --no-render runs the physics only (any environment
with MuJoCo); --episodes picks episodes.
"""

import argparse
import glob
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq

import twin_scene

EXTRINSICS = 'runs/calib/top_extrinsics_2/extrinsics.json'
FPS = 30
DIFF_THRESHOLD = 40  # as scan_object.py: brightest channel difference from the gain-matched background
WHITE_MIN, WHITE_SPREAD = 150, 60
TABLE_X_M, TABLE_Y_M = (0.05, 0.39), (-0.30, 0.31)
ARM_MARGIN_PX = 25
KEY_GATHER_M = 0.06  # pieces of the key ring's mask this close to its largest piece belong to it
MOVING_DEG = 1.0


class Episodes:
    """Actions, states and video frames of a LeRobot v3 dataset, by episode."""

    def __init__(self, dataset):
        self.root = Path(dataset)
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        files = sorted(glob.glob(str(self.root / 'data/chunk-*/*.parquet')))
        frames = pq.read_table(files, columns=['action', 'observation.state', 'episode_index']).to_pandas()
        self.frames = {int(i): (np.stack(g['action'].values), np.stack(g['observation.state'].values))
                       for i, g in frames.groupby('episode_index')}
        files = sorted(glob.glob(str(self.root / 'meta/episodes/chunk-*/*.parquet')))
        self.meta = pq.read_table(files).to_pandas().set_index('episode_index')

    def __len__(self):
        return len(self.frames)

    def video(self, episode, camera):
        """Yields the BGR frames of one camera for the episode."""
        import av

        key = f'observation.images.{camera}'
        row = self.meta.loc[episode]
        path = self.root / self.info['video_path'].format(video_key=key, chunk_index=int(row[f'videos/{key}/chunk_index']),
                                                           file_index=int(row[f'videos/{key}/file_index']))
        first = int(round(row[f'videos/{key}/from_timestamp'] * FPS))
        count = len(self.frames[episode][0])
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            if first:
                container.seek(int(max(first / FPS - 2, 0) / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                index = int(round(float(frame.pts * stream.time_base) * FPS))
                if index < first:
                    continue
                if index >= first + count:
                    break
                yield frame.to_ndarray(format='bgr24')


class TopCamera:
    def __init__(self):
        extrinsics = json.loads(Path(EXTRINSICS).read_text())
        intrinsics = json.loads(Path(extrinsics['intrinsics']).read_text())
        self.K, self.dist = np.array(intrinsics['K']), np.array(intrinsics['dist'])
        self.R, self.t = np.array(extrinsics['R_cam_base']), np.array(extrinsics['t_cam_base_m'])
        self.rvec = cv2.Rodrigues(self.R)[0]

    def project(self, points):
        return cv2.projectPoints(np.asarray(points, float), self.rvec, self.t, self.K, self.dist)[0].reshape(-1, 2)

    def on_plane(self, pixels, plane):
        """Base-frame points where the rays through the pixels meet the plane z = a x + b y + c."""
        normal = cv2.undistortPoints(np.asarray(pixels, np.float64).reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        rays = np.column_stack([normal, np.ones(len(normal))]) @ self.R
        origin = -self.R.T @ self.t
        up = np.array([-plane[0], -plane[1], 1.0])
        return origin + ((plane[2] - up @ origin) / (rays @ up))[:, None] * rays


def arm_mask(scene, camera, shape):
    """Top camera pixels covered by the arm in the scene's current pose, with a margin."""
    model, data = scene.model, scene.data
    mask = np.zeros(shape, np.uint8)
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 2:
            continue
        mesh = model.geom_dataid[geom]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        world = (data.geom_xmat[geom].reshape(3, 3) @ vertices.T).T + data.geom_xpos[geom]
        cv2.fillConvexPoly(mask, cv2.convexHull(camera.project(world).astype(np.int32)), 1)
    return cv2.dilate(mask, np.ones((2 * ARM_MARGIN_PX + 1,) * 2, np.uint8)) > 0


def simplex(cost, start, step, iterations=120):
    """Nelder-Mead minimisation; returns the best point and its cost."""
    points = np.vstack([start] + [start + np.eye(len(start))[i] * step[i] for i in range(len(start))])
    values = np.array([cost(p) for p in points])
    for _ in range(iterations):
        order = np.argsort(values)
        points, values = points[order], values[order]
        centre = points[:-1].mean(0)
        reflected = centre + (centre - points[-1])
        value = cost(reflected)
        if value < values[0]:
            expanded = centre + 2 * (centre - points[-1])
            other = cost(expanded)
            points[-1], values[-1] = (expanded, other) if other < value else (reflected, value)
        elif value < values[-2]:
            points[-1], values[-1] = reflected, value
        else:
            contracted = centre + 0.5 * (points[-1] - centre)
            other = cost(contracted)
            if other < values[-1]:
                points[-1], values[-1] = contracted, other
            else:
                points[1:] = points[0] + 0.5 * (points[1:] - points[0])
                values[1:] = [cost(p) for p in points[1:]]
    best = int(np.argmin(values))
    return points[best], float(values[best])


class Locator:
    """Finds the box and the key ring on the table in a top image."""

    def __init__(self, scan, scene, box_dir, key_dir):
        self.scene, self.camera, self.plane = scene, TopCamera(), scene.plane
        files = sorted(glob.glob(f'{scan}/top_backgrounds/top_*.png'))[1:]
        self.background = np.median(np.stack([cv2.imread(f) for f in files]), 0)
        box = json.loads((Path(box_dir) / 'object/box.json').read_text())
        h = box['size_m'] / 2
        self.box_corners = np.array([[sx * h, sy * h, z] for z in (0, box['height_m']) for sx in (-1, 1) for sy in (-1, 1)])
        # the key ring as the collision columns of twin_scene.py: eight corners each, filled one by one
        self.key_corners = np.array([[x + sx * hx, y + sy * hy, z] for x, y, hx, hy, height in twin_scene.object_columns(key_dir)
                                     for z in (0, height) for sx in (-1, 1) for sy in (-1, 1)])

    def silhouette(self, name, placement, shape):
        pose = self.scene.on_table(*placement)
        mask = np.zeros(shape, np.uint8)
        if name == 'box':
            pixels = self.camera.project(self.box_corners @ pose[:3, :3].T + pose[:3, 3])
            cv2.fillConvexPoly(mask, cv2.convexHull(pixels.astype(np.int32)), 1)
        else:
            pixels = self.camera.project(self.key_corners @ pose[:3, :3].T + pose[:3, 3]).astype(np.int32)
            for column in pixels.reshape(-1, 8, 2):
                cv2.fillConvexPoly(mask, cv2.convexHull(column), 1)
        return mask > 0

    def fit(self, name, mask, yaws):
        ys, xs = np.nonzero(mask)
        centre = self.camera.on_plane([[xs.mean(), ys.mean()]], self.plane)[0]

        def cost(p):
            drawn = self.silhouette(name, p, mask.shape)
            return 1 - (drawn & mask).sum() / max((drawn | mask).sum(), 1)

        best = min((simplex(cost, np.array([centre[0], centre[1], yaw]), np.array([0.01, 0.01, 0.15])) for yaw in yaws),
                   key=lambda r: r[1])
        return tuple(float(v) for v in best[0]), 1 - best[1]

    def __call__(self, image, joints):
        """Placements {'box': (x, y, yaw), 'key': ...}, the overlap of each fit with its mask, and the masks."""
        self.scene.set_arm(joints)
        image = image.astype(np.float32)
        gain = np.median(image / np.maximum(self.background, 1), axis=(0, 1))
        changed = (np.abs(image - self.background * gain).max(2) > DIFF_THRESHOLD)
        changed &= ~arm_mask(self.scene, self.camera, changed.shape)
        ys, xs = np.nonzero(changed)
        where = self.camera.on_plane(np.column_stack([xs, ys]), self.plane)
        off = (where[:, 0] < TABLE_X_M[0]) | (where[:, 0] > TABLE_X_M[1]) | (where[:, 1] < TABLE_Y_M[0]) | (where[:, 1] > TABLE_Y_M[1])
        changed[ys[off], xs[off]] = False
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        white = (image.min(2) > WHITE_MIN) & (image.max(2) - image.min(2) < WHITE_SPREAD)
        count, labels, stats, _ = cv2.connectedComponentsWithStats((changed > 0).astype(np.uint8))
        if count < 3:
            raise ValueError('fewer than two things found on the table')
        whiteness = [white[labels == k].mean() * stats[k, cv2.CC_STAT_AREA] for k in range(1, count)]
        box_label = 1 + int(np.argmax(whiteness))
        box_mask = labels == box_label
        box_mask = cv2.morphologyEx(box_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0

        near_box = cv2.dilate(box_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        others = [k for k in range(1, count) if k != box_label and not near_box[labels == k].any()]
        if not others:
            raise ValueError('only the box found on the table')
        spot = lambda k: self.camera.on_plane([[stats[k, 0] + stats[k, 2] / 2, stats[k, 1] + stats[k, 3] / 2]], self.plane)[0]
        largest = max(others, key=lambda k: stats[k, cv2.CC_STAT_AREA])
        key_mask = np.isin(labels, [k for k in others if np.linalg.norm(spot(k) - spot(largest)) < KEY_GATHER_M])

        box, box_overlap = self.fit('box', box_mask, [0.0, np.pi / 4])
        key, key_overlap = self.fit('key', key_mask, np.arange(12) * np.pi / 6)
        return {'box': box, 'key': key}, {'box': box_overlap, 'key': key_overlap}, {'box': box_mask, 'key': key_mask}


def first_still_frame(states):
    """Index of the last frame before the arm starts to move."""
    moved = np.abs(states[:, :5] - states[0, :5]).max(1) > MOVING_DEG
    return int(np.argmax(moved)) - 1 if moved.any() else 0


def key_in_box(scene, poses):
    """True when the key ring's origin lies over the box's floor and below its rim."""
    local = np.linalg.inv(poses['box']) @ poses['key'][:, 3]
    return bool(abs(local[0]) < 0.045 and abs(local[1]) < 0.045 and -0.005 < local[2] < 0.03)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scan', required=True)
    parser.add_argument('--box', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes', type=int, nargs='*')
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--table-drop', type=float, default=0.0, help='an extra drop of the physics table, m')
    parser.add_argument('--key-softness', type=float, default=0.0,
                        help='how far the jaws may press into the key ring before it pushes back fully, m (0: rigid)')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    episodes = Episodes(args.dataset)
    scene = twin_scene.Scene(args.scan, {'box': args.box, 'key': args.key}, table_drop_m=args.table_drop,
                             key_softness_m=args.key_softness)
    locator = Locator(args.scan, scene, args.box, args.key)
    renderer = None
    if not args.no_render:
        import twin_compare
        renderer = twin_compare.Comparison(args.scan, args.box, args.key)

    summary = []
    for episode in (args.episodes if args.episodes else range(len(episodes))):
        actions, states = episodes.frames[episode]
        top_frames = episodes.video(episode, 'top')
        first = next(top_frames)
        try:
            placements, overlap, masks = locator(first, states[0])
        except ValueError as error:
            print(f'episode {episode}: skipped, {error}', flush=True)
            continue
        check = first.copy()
        for name, colour in (('box', (0, 255, 0)), ('key', (0, 0, 255))):
            outline = locator.silhouette(name, placements[name], first.shape[:2]).astype(np.uint8)
            cv2.drawContours(check, cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, colour, 1)
        cv2.imwrite(str(out / f'ep{episode:02d}_located.jpg'), check)

        scene.reset(states[0], placements)
        simulated = np.zeros((len(actions), 6))
        tips, objects = [], {name: [] for name in scene.object_names}
        if renderer:
            renderer.start(out / f'ep{episode:02d}.mp4', placements)
            wrist_frames = episodes.video(episode, 'wrist')
        for frame, action in enumerate(actions):
            simulated[frame] = scene.data.qpos[:6]
            poses = scene.object_poses()
            for name in objects:
                objects[name].append(poses[name])
            if renderer:
                top_image = first if frame == 0 else next(top_frames, None)
                wrist_image = next(wrist_frames, None)
                if top_image is not None and wrist_image is not None:
                    renderer.frame(scene, poses, top_image, wrist_image)
            scene.step(action)
        final = scene.object_poses()
        recorded = scene.to_rad(states)
        error = np.degrees(np.abs(simulated[:, :5] - recorded[:, :5]))
        lifted = max(pose[2, 3] for pose in objects['key']) - objects['key'][0][2, 3]
        row = {'episode': episode, 'frames': len(actions), 'placements': placements, 'fit_overlap': overlap,
               'joint_error_deg_mean': float(error.mean()), 'joint_error_deg_max': float(error.max()),
               'key_lift_mm': float(lifted * 1000), 'key_in_box': key_in_box(scene, final),
               'key_final_in_box_frame_mm': (np.linalg.inv(final['box']) @ final['key'][:, 3])[:3].round(4).tolist()}
        if renderer:
            row.update(renderer.finish())
        np.savez(out / f'ep{episode:02d}.npz', simulated=simulated, recorded=recorded,
                 **{name: np.stack(poses) for name, poses in objects.items()})
        summary.append(row)
        print(json.dumps(row), flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    done = [r for r in summary]
    print(f'{len(done)} episodes: joint error mean {np.mean([r["joint_error_deg_mean"] for r in done]):.2f} deg, '
          f'key ring in the box at the end in {sum(r["key_in_box"] for r in done)}')


if __name__ == '__main__':
    main()
