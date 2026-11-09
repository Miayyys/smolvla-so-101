"""Prepare the wrist scan for reconstruction: jaw mask, and camera poses refined by structure from motion.

    python scripts/recon_table.py mask --scan runs/scan/table_curtain
    python scripts/recon_table.py colmap --scan runs/scan/table_curtain

Results are in <scan>/colmap: poses_base_from_cam.json (OpenCV camera in the MJCF base frame) and points_base.npz.

`mask` segments the jaws from the median of all views. The jaws are the only thing that stays in place in every image,
so the median keeps them sharp while the table blurs to a mid grey. `colmap` matches features outside the mask with the
calibrated intrinsics, then aligns the result to the joint-based poses, which fixes scale and the base frame.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import mujoco
import numpy as np

import calib_top_extrinsics as top
import scan_wrist as scan

INTRINSICS = 'runs/calib/wrist/intrinsics_standard.json'
PAIR_DISTANCE_M, PAIR_ANGLE_DEG = 0.15, 50.0
ALIGN_ROUNDS, ALIGN_MIN_RADIUS_M = 12, 0.008
MAX_POSITION_ERROR_MM, MAX_ROTATION_ERROR_DEG = 20.0, 8.0  # far beyond the calibration error of about 5 mm and 3 deg
MIN_TRACK_LENGTH = 3
MAX_TRANSFER_MM, MAX_TRANSFER_DEG = 8.0, 4.0
SIFT_PEAK_THRESHOLD = 0.004  # below the default: the black mat has little contrast
LINKS = ['base', 'shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper', 'moving_jaw_so101_v1']
SEEDS_ON_SURFACE, SEEDS_AROUND = 7000, 3000  # per link; those around it reach cables and the camera, absent from the mesh
AROUND_MARGIN_M = 0.03
JAW_MASK = 'jaw_mask.png'
BRIGHT_MIN, DARK_MAX = 170, 55  # the median table lies between: white plastic above, black plastic below
MASK_MARGIN_PX = 5


def views(scan):
    data = json.loads((Path(scan) / 'wrist_views.json').read_text())
    return sorted(data['views']), data


def make_mask(args):
    scan = Path(args.scan)
    names, _ = views(scan)
    median = np.median(np.stack([cv2.imread(str(scan / n)) for n in names[::2]]), axis=0).astype(np.uint8)
    jaws = ((median.min(2) > BRIGHT_MIN) | (median.max(2) < DARK_MAX)).astype(np.uint8)
    jaws = cv2.morphologyEx(jaws, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(jaws)
    height = jaws.shape[0]
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= height - 1]
    mask = np.isin(labels, keep).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_MARGIN_PX + 1,) * 2))
    cv2.imwrite(str(scan / JAW_MASK), mask * 255)
    cv2.imwrite(str(scan / 'median.png'), median)
    print(f'jaw mask covers {mask.mean():.1%} of the image, {len(keep)} regions')


def joint_poses(names, data):
    """T_base_cam of each view from the joint readings, the hand-eye mount and the joint zero offsets."""
    camera = scan.CameraPose()
    return {n: camera.from_raw([data['views'][n][m] for m in top.ALL_MOTORS]) for n in names}


def write_pairs(path, names, poses):
    centres = np.array([poses[n][:3, 3] for n in names])
    axes = np.array([poses[n][:3, 2] for n in names])
    lines = []
    for i in range(len(names)):
        close = (np.linalg.norm(centres[i + 1:] - centres[i], axis=1) < PAIR_DISTANCE_M) & \
            (axes[i + 1:] @ axes[i] > np.cos(np.radians(PAIR_ANGLE_DEG)))
        lines += [f'{names[i]} {names[i + 1 + j]}' for j in np.where(close)[0]]
    path.write_text('\n'.join(lines) + '\n')
    return len(lines)


def run_colmap(args):
    import pycolmap

    scan_dir = Path(args.scan)
    work = scan_dir / 'colmap'
    work.mkdir(exist_ok=True)
    database = work / 'database.db'
    if database.exists():
        database.unlink()
    names, data = views(scan_dir)
    poses = joint_poses(names, data)

    mask = cv2.imread(str(scan_dir / JAW_MASK), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(work / 'camera_mask.png'), 255 - mask)  # COLMAP takes no features where the mask is black
    intrinsics = json.loads(Path(INTRINSICS).read_text())
    K, dist = np.array(intrinsics['K']), np.ravel(intrinsics['dist'])
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = 'FULL_OPENCV'
    reader.camera_params = ','.join(str(float(v)) for v in [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *dist, 0, 0, 0])
    reader.camera_mask_path = str(work / 'camera_mask.png')
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.peak_threshold = SIFT_PEAK_THRESHOLD
    pycolmap.extract_features(database, scan_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
    pairs = work / 'pairs.txt'
    print(f'{write_pairs(pairs, names, poses)} image pairs to match', flush=True)
    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = str(pairs)
    pycolmap.match_image_pairs(database, pairing_options=pairing, device=pycolmap.Device.cpu)

    options = pycolmap.IncrementalPipelineOptions()
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.multiple_models = False
    models = pycolmap.incremental_mapping(database, scan_dir, work, options)
    if not models:
        raise SystemExit('COLMAP built no model')
    model = max(models.values(), key=lambda m: m.num_reg_images())
    print(model.summary(), flush=True)

    write_aligned(scan_dir, model, poses)


def umeyama(source, target):
    """Scale, rotation and translation with target = scale * R * source + t, least squares."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    a, b = source - source_mean, target - target_mean
    u, d, vt = np.linalg.svd(b.T @ a / len(a))
    flip = np.diag([1, 1, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    R = u @ flip @ vt
    scale = (d * np.diag(flip)).sum() / (a ** 2).sum(1).mean()
    return scale, R, target_mean - scale * R @ source_mean


def write_aligned(scan_dir, model, poses):
    """Bring the model into the base frame by matching camera centres, leaving out the few cameras COLMAP misplaced.

    COLMAP's own robust estimate accepts everything here: a handful of cameras are thousands of units away. So the fit
    starts from the cameras nearest the median and is repeated on those that agree with it.
    """
    work = scan_dir / 'colmap'
    images = [image for image in model.images.values() if image.has_pose and np.isfinite(image.projection_center()).all()]
    sfm = np.array([image.projection_center() for image in images])
    joints = np.array([poses[image.name][:3, 3] for image in images])
    distance = np.linalg.norm(sfm - np.median(sfm, axis=0), axis=1)
    keep = distance < np.percentile(distance, 80)
    for _ in range(ALIGN_ROUNDS):
        scale, R, t = umeyama(sfm[keep], joints[keep])
        error = np.linalg.norm(scale * (R @ sfm.T).T + t - joints, axis=1)
        keep = error < max(np.percentile(error, 70), ALIGN_MIN_RADIUS_M)
    position_mm, angle_deg, refined = [], [], {}
    for image, centre in zip(images, sfm):
        rotation = R @ image.cam_from_world().rotation.matrix().T
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, scale * R @ centre + t
        joint = poses[image.name]
        error_mm = np.linalg.norm(pose[:3, 3] - joint[:3, 3]) * 1000
        cosine = (np.trace(rotation.T @ joint[:3, :3]) - 1) / 2
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        if error_mm <= MAX_POSITION_ERROR_MM and angle <= MAX_ROTATION_ERROR_DEG:
            refined[image.name] = pose.tolist()
            position_mm.append(error_mm)
            angle_deg.append(angle)
    points = [p for p in model.points3D.values() if p.track.length() >= MIN_TRACK_LENGTH]
    xyz = scale * (R @ np.array([p.xyz for p in points]).T).T + t
    np.savez(work / 'points_base.npz', xyz=xyz, rgb=np.array([p.color for p in points]))
    (work / 'poses_base_from_cam.json').write_text(json.dumps(refined, indent=0) + '\n')
    print(f'{len(refined)} of {len(poses)} views kept ({len(images)} registered), scale {scale:.5f}, {len(points)} points')
    print(f'difference from the joint-based poses: position mean {np.mean(position_mm):.1f} mm, '
          f'p95 {np.percentile(position_mm, 95):.1f} mm; rotation mean {np.mean(angle_deg):.2f} deg, '
          f'p95 {np.percentile(angle_deg, 95):.2f} deg')
    print(f'point height in the base frame: p5 {np.percentile(xyz[:, 2], 5) * 1000:.1f}, median '
          f'{np.median(xyz[:, 2]) * 1000:.1f}, p95 {np.percentile(xyz[:, 2], 95) * 1000:.1f} mm')


def transfer_poses(args):
    """Give a scan the refined poses of another scan that followed the same path, corrected by the joint differences.

    Structure from motion needs texture and light: the night scan registers almost nothing. Sharing the poses also keeps
    both scans in exactly the same frame.
    """
    source, target = Path(args.source), Path(args.scan)
    refined = json.loads((source / 'colmap/poses_base_from_cam.json').read_text())
    source_names, source_data = views(source)
    source_joints = joint_poses(source_names, source_data)
    names, data = views(target)
    target_joints = joint_poses(names, data)
    available = [n for n in source_names if n in refined]
    centres = np.array([source_joints[n][:3, 3] for n in available])
    axes = np.array([source_joints[n][:3, 2] for n in available])
    result, position_mm = {}, []
    for name in names:
        pose = target_joints[name]
        distance = np.linalg.norm(centres - pose[:3, 3], axis=1)
        nearest = int(distance.argmin())
        angle = np.degrees(np.arccos(np.clip(axes[nearest] @ pose[:3, 2], -1, 1)))
        if distance[nearest] * 1000 <= MAX_TRANSFER_MM and angle <= MAX_TRANSFER_DEG:
            other = available[nearest]
            correction = np.linalg.inv(source_joints[other]) @ pose
            result[name] = (np.array(refined[other]) @ correction).tolist()
            position_mm.append(distance[nearest] * 1000)
    work = target / 'colmap'
    work.mkdir(exist_ok=True)
    (work / 'poses_base_from_cam.json').write_text(json.dumps(result, indent=0) + '\n')
    shutil.copy(source / 'colmap/points_base.npz', work / 'points_base.npz')
    print(f'{len(result)} of {len(names)} views matched; joint-based distance to the matched view: '
          f'mean {np.mean(position_mm):.1f} mm, max {np.max(position_mm):.1f} mm')


def align(args):
    import pycolmap

    scan_dir = Path(args.scan)
    names, data = views(scan_dir)
    write_aligned(scan_dir, pycolmap.Reconstruction(str(scan_dir / 'colmap/0')), joint_poses(names, data))


def export_arm(args):
    """Link poses of every arm stop, and seed points in each link's frame, for recon_arm.py (which has no MuJoCo)."""
    arm_dir = Path(args.arm)
    stops = sorted(json.loads((arm_dir / 'stops.json').read_text()).values(), key=lambda v: v['name'])
    camera = scan.CameraPose()
    model, state = camera.kinematics.model, camera.kinematics.data
    bodies = [model.body(name).id for name in LINKS]
    gripper = camera.calibration['gripper']
    low, high = model.jnt_range[5]

    def set_pose(raw):
        state.qpos[:5] = np.radians(top.raw_to_lerobot_deg(raw, camera.calibration) + camera.offsets)
        opening = (raw['gripper'] - gripper['range_min']) / (gripper['range_max'] - gripper['range_min'])
        state.qpos[5] = low + opening * (high - low)
        mujoco.mj_kinematics(model, state)

    poses = np.zeros((len(stops), len(LINKS), 4, 4))
    for i, stop in enumerate(stops):
        set_pose(stop['raw'])
        for j, body in enumerate(bodies):
            poses[i, j] = np.eye(4)
            poses[i, j, :3, :3], poses[i, j, :3, 3] = state.xmat[body].reshape(3, 3), state.xpos[body]

    rng = np.random.default_rng(0)
    seeds, owners = [], []
    set_pose(stops[0]['raw'])
    for j, body in enumerate(bodies):
        rotation, origin = state.xmat[body].reshape(3, 3), state.xpos[body]
        vertices, triangles = [], []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != body or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 2:
                continue
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            first, faces = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            world = (state.geom_xmat[geom].reshape(3, 3) @ model.mesh_vert[start:start + count].T).T + state.geom_xpos[geom]
            triangles.append(model.mesh_face[first:first + faces] + sum(len(v) for v in vertices))
            vertices.append((rotation.T @ (world - origin).T).T)
        vertices, triangles = np.vstack(vertices), np.vstack(triangles)
        corners = vertices[triangles]
        area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
        chosen = corners[rng.choice(len(corners), SEEDS_ON_SURFACE, p=area / area.sum())]
        a, b = rng.random((2, SEEDS_ON_SURFACE))
        flip = a + b > 1
        a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
        surface = chosen[:, 0] + a[:, None] * (chosen[:, 1] - chosen[:, 0]) + b[:, None] * (chosen[:, 2] - chosen[:, 0])
        around = rng.uniform(vertices.min(0) - AROUND_MARGIN_M, vertices.max(0) + AROUND_MARGIN_M, (SEEDS_AROUND, 3))
        seeds.append(np.vstack([surface, around]))
        owners.append(np.full(SEEDS_ON_SURFACE + SEEDS_AROUND, j))

    extrinsics = json.loads(Path(scan.EXTRINSICS).read_text())
    intrinsics = json.loads(Path(extrinsics['intrinsics']).read_text())
    np.savez(arm_dir / 'links.npz', poses=poses, links=LINKS, seeds=np.vstack(seeds), owners=np.concatenate(owners),
             names=[s['name'] for s in stops], K=np.array(intrinsics['K']), R=np.array(extrinsics['R_cam_base']),
             t=np.array(extrinsics['t_cam_base_m']))
    print(f'{len(stops)} stops, {len(LINKS)} links, {sum(len(s) for s in seeds)} seeds written to {arm_dir / "links.npz"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    mask = commands.add_parser('mask')
    mask.add_argument('--scan', required=True)
    mask.set_defaults(run=make_mask)
    links = commands.add_parser('armlinks', help='link poses and seeds of the arm capture, for recon_arm.py')
    links.add_argument('--arm', required=True)
    links.set_defaults(run=export_arm)
    sfm = commands.add_parser('colmap')
    sfm.add_argument('--scan', required=True)
    sfm.set_defaults(run=run_colmap)
    copy = commands.add_parser('transfer', help='take the refined poses of a scan that followed the same path')
    copy.add_argument('--scan', required=True)
    copy.add_argument('--source', required=True)
    copy.set_defaults(run=transfer_poses)
    realign = commands.add_parser('align', help='repeat only the alignment to the base frame')
    realign.add_argument('--scan', required=True)
    realign.set_defaults(run=align)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
