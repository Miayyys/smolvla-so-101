"""Model of an open box (a tray) lying on the table, from scan_object.py: measured size, fitted pose, one texture per face.

    python scripts/recon_table.py transfer --scan runs/scan/box --source runs/scan/table_curtain
    python scripts/recon_box.py --scan runs/scan/box --source runs/scan/table_curtain --size 0.09 --height 0.022 --wall 0.0017

The geometry is the measured one: outer walls, rim, inner walls and floor (floor as thick as the walls). The pose on the
table (x, y, yaw) starts from the white outline in the top image, projected onto the plane of the rim, and is refined by
matching the box's outline in the wrist views (Nelder-Mead on the mean overlap). Each face texel takes the median colour
of the views that see it unoccluded (ray cast against the faces), brought to the light of the workspace scan with the
bare table around the box.

Output in <scan>/object: box.json (dimensions, pose, gain), face_<name>.png (BGR), check.jpg (outline on some views).
The face geometry is shared with twin_render.py, so this file imports nothing heavy at module level.
"""

import argparse
import json
from pathlib import Path

import numpy as np

TEXEL_M = 0.0005
WHITE_MIN, WHITE_SPREAD = 165, 45  # box pixels: all channels bright, little colour (the lamp highlight on the mat is blue)
EDGE_SAMPLES = 24
SCALE = 4  # outline overlap computed at quarter resolution
VISIBLE_TOLERANCE_M = 0.001
RING_M = (0.012, 0.035)  # bare table around the box for the light ratio; the box's own shadow lies closer


def faces(size, height, wall):
    """Faces in the box frame (origin at the centre of its base, z up) as (name, origin, u axis, v axis, u length,
    v length). The visible side is along u x v. The rim is the whole top square less the opening, see in_hole."""
    h, i, t = size / 2, size / 2 - wall, wall
    x, y, z = np.eye(3)
    return [
        ('rim', (-h, -h, height), x, y, 2 * h, 2 * h),
        ('floor', (-i, -i, t), x, y, 2 * i, 2 * i),
        ('outer_px', (h, -h, 0), y, z, 2 * h, height),
        ('outer_nx', (-h, h, 0), -y, z, 2 * h, height),
        ('outer_py', (h, h, 0), -x, z, 2 * h, height),
        ('outer_ny', (-h, -h, 0), x, z, 2 * h, height),
        ('inner_px', (i, i, t), -y, z, 2 * i, height - t),
        ('inner_nx', (-i, -i, t), y, z, 2 * i, height - t),
        ('inner_py', (-i, i, t), x, z, 2 * i, height - t),
        ('inner_ny', (i, -i, t), -x, z, 2 * i, height - t),
    ]


def in_hole(name, u, v, size, wall):
    """True where a point of the rim face lies over the opening."""
    if name != 'rim':
        return np.zeros_like(u, bool)
    return (u > wall) & (u < size - wall) & (v > wall) & (v < size - wall)


def cast(origins, directions, size, height, wall):
    """Nearest front-facing face hit by each ray (box frame): distance (inf if none), face index, u, v."""
    best = np.full(len(origins), np.inf)
    index, best_u, best_v = np.full(len(origins), -1), np.zeros(len(origins)), np.zeros(len(origins))
    for k, (name, origin, u_axis, v_axis, u_len, v_len) in enumerate(faces(size, height, wall)):
        normal = np.cross(u_axis, v_axis)
        facing = directions @ normal
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((np.asarray(origin) - origins) @ normal) / facing
        hit = origins + t[:, None] * directions - origin
        u, v = hit @ u_axis, hit @ v_axis
        ok = (facing < 0) & (t > 0) & (u >= 0) & (u <= u_len) & (v >= 0) & (v <= v_len) & ~in_hole(name, u, v, size, wall)
        closer = ok & (t < best)
        best[closer], index[closer], best_u[closer], best_v[closer] = t[closer], k, u[closer], v[closer]
    return best, index, best_u, best_v


def base_from_box(x, y, yaw, plane):
    pose = np.eye(4)
    pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    pose[:3, 3] = x, y, plane[0] * x + plane[1] * y + plane[2]
    return pose


def edge_points(size, height):
    h = size / 2
    corners = np.array([[sx * h, sy * h, z] for z in (0, height) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if np.count_nonzero(np.abs(corners[a] - corners[b]) > 1e-9) == 1]
    s = np.linspace(0, 1, EDGE_SAMPLES)[:, None]
    return np.vstack([corners[a] + s * (corners[b] - corners[a]) for a, b in edges])


def main():
    import cv2

    import scan_object
    from sysid_actuators import nelder_mead

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scan', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--size', type=float, required=True, help='outer side, metres')
    parser.add_argument('--height', type=float, required=True)
    parser.add_argument('--wall', type=float, required=True)
    args = parser.parse_args()
    scan, source = Path(args.scan), Path(args.source)
    size, height, wall = args.size, args.height, args.wall
    table = np.load(source / 'gs/table.npz')
    plane = table['plane']
    intrinsics = json.loads(Path(scan_object.WRIST_INTRINSICS).read_text())
    K, dist = np.array(intrinsics['K']), np.ravel(intrinsics['dist'])
    jaws = cv2.imread(str(scan / 'jaw_mask.png'), cv2.IMREAD_GRAYSCALE) > 0
    poses = {n: np.array(p) for n, p in json.loads((scan / 'colmap/poses_base_from_cam.json').read_text()).items()}
    names = sorted(poses)
    images = {n: cv2.imread(str(scan / n)) for n in names}

    # Start: the white outline in the top image on the plane of the rim.
    top = cv2.imread(str(scan / 'top.png')).astype(int)
    white = ((top.min(2) > WHITE_MIN) & (top.max(2) - top.min(2) < WHITE_SPREAD)).astype(np.uint8)
    white &= cv2.dilate(cv2.imread(str(scan / 'top_mask.png'), cv2.IMREAD_GRAYSCALE), np.ones((9, 9), np.uint8)) > 0
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2)
    rim_plane = plane + [0, 0, height]
    points = scan_object.on_plane(outline, rim_plane)
    (cx, cy), (w, h), angle = cv2.minAreaRect((points[:, :2] * 1e4).astype(np.float32))
    start = np.array([cx / 1e4, cy / 1e4, np.radians(angle) % (np.pi / 2)])
    print(f'top image: box {w / 10:.1f} x {h / 10:.1f} mm (measured {size * 1000:.0f}), centre '
          f'{np.round(start[:2] * 1000, 1)} mm, yaw {np.degrees(start[2]):.1f} deg', flush=True)

    # Refine on the outlines in the wrist views.
    small = {}
    for n in names:
        image = images[n].astype(int)
        mask = ((image.min(2) > WHITE_MIN) & (image.max(2) - image.min(2) < WHITE_SPREAD) & ~jaws).astype(np.uint8)
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [c for c in found if cv2.contourArea(c) > 400], -1, 1, cv2.FILLED)
        small[n] = cv2.resize(filled, None, fx=1 / SCALE, fy=1 / SCALE, interpolation=cv2.INTER_AREA) > 0.5
    usable = cv2.resize((~jaws).astype(np.uint8), None, fx=1 / SCALE, fy=1 / SCALE, interpolation=cv2.INTER_AREA) > 0.5
    edges = np.column_stack([edge_points(size, height), np.ones(EDGE_SAMPLES * 12)])

    def silhouette(n, params):
        world = (base_from_box(*params, plane) @ edges.T).T[:, :3]
        local = (world - poses[n][:3, 3]) @ poses[n][:3, :3]
        if (local[:, 2] < 0.02).any():
            return None
        uv = cv2.projectPoints(local, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2) / SCALE
        drawn = np.zeros(usable.shape, np.uint8)
        cv2.fillConvexPoly(drawn, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        return drawn > 0

    def overlap(params):
        scores = []
        for n in names:
            predicted = silhouette(n, params)
            if predicted is None:
                continue
            a, b = predicted & usable, small[n] & usable
            union = (a | b).sum()
            if union:
                scores.append((a & b).sum() / union)
        return float(np.mean(scores))

    before = overlap(start)
    best, value = nelder_mead(lambda p: 1 - overlap(p), start, np.array([0.003, 0.003, np.radians(2)]), 150)
    shift = np.linalg.norm(best[:2] - start[:2]) * 1000
    print(f'wrist outlines: overlap {before:.3f} -> {1 - value:.3f}; moved {shift:.1f} mm, '
          f'turned {np.degrees(best[2] - start[2]):.1f} deg', flush=True)
    box_pose = base_from_box(*best, plane)
    box_from_base = np.linalg.inv(box_pose)

    # Light: the bare table around the box against the workspace scan's texture.
    radius = np.linspace(size / 2 + RING_M[0], size / 2 + RING_M[1], 12)
    ring = np.array([[r * np.cos(a), r * np.sin(a)] for r in radius for a in np.linspace(0, 2 * np.pi, 120, endpoint=False)])
    world_ring = (box_pose[:2, :2] @ ring.T).T + box_pose[:2, 3]
    world_ring = np.column_stack([world_ring, world_ring @ plane[:2] + plane[2]])

    def sample(n, world, box_points=None, face=None):
        """Colours of world points in view n, NaN where outside, on the jaws or (for box points) occluded."""
        pose = poses[n]
        local = (world - pose[:3, 3]) @ pose[:3, :3]
        uv = cv2.projectPoints(local, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2).astype(np.float32)
        ok = (local[:, 2] > 0.02) & (uv[:, 0] >= 1) & (uv[:, 0] < 638) & (uv[:, 1] >= 1) & (uv[:, 1] < 478)
        pixel = np.clip(np.round(uv).astype(int), 0, [639, 479])
        ok &= ~jaws[pixel[:, 1], pixel[:, 0]]
        camera = box_from_base[:3, :3] @ pose[:3, 3] + box_from_base[:3, 3]
        target = box_points if box_points is not None else (box_from_base[:3, :3] @ world.T).T + box_from_base[:3, 3]
        direction = target - camera
        distance = np.linalg.norm(direction, axis=1)
        t, index, _, _ = cast(np.repeat(camera[None], len(target), 0), direction / distance[:, None], size, height, wall)
        if box_points is None:
            ok &= t > distance - VISIBLE_TOLERANCE_M  # table points hidden by the box
        else:
            ok &= (index == face) & (np.abs(t - distance) < VISIBLE_TOLERANCE_M)
        padded = np.zeros((-(-len(uv) // 1024) * 1024, 2), np.float32)
        padded[:len(uv)] = uv
        maps = padded.reshape(-1, 1024, 2)
        colour = cv2.remap(images[n].astype(np.float32), maps[..., 0], maps[..., 1], cv2.INTER_LINEAR).reshape(-1, 3)[:len(uv)]
        colour[~ok] = np.nan
        return colour

    ring_colour = np.nanmedian(np.stack([sample(n, world_ring) for n in names]), 0)
    texture = table['texture'][..., ::-1].astype(np.float32)  # stored RGB
    rows = np.clip(((world_ring[:, 0] - table['x_range'][0]) / table['cell']).astype(int), 0, texture.shape[0] - 1)
    columns = np.clip(((world_ring[:, 1] - table['y_range'][0]) / table['cell']).astype(int), 0, texture.shape[1] - 1)
    seen = np.isfinite(ring_colour[:, 0])
    gain = np.median(ring_colour[seen] / np.maximum(texture[rows, columns][seen], 1), axis=0)
    print(f'light against the workspace scan: gain {np.round(gain, 2)} from {seen.sum()} table points', flush=True)

    out = scan / 'object'
    out.mkdir(exist_ok=True)
    report = []
    for k, (name, origin, u_axis, v_axis, u_len, v_len) in enumerate(faces(size, height, wall)):
        us = np.arange(TEXEL_M / 2, u_len, TEXEL_M)
        vs = np.arange(TEXEL_M / 2, v_len, TEXEL_M)
        gu, gv = np.meshgrid(us, vs, indexing='ij')
        box_points = np.asarray(origin) + gu.reshape(-1, 1) * u_axis + gv.reshape(-1, 1) * v_axis
        world = (box_pose[:3, :3] @ box_points.T).T + box_pose[:3, 3]
        samples = np.stack([sample(n, world, box_points, k) for n in names])
        count = np.isfinite(samples[..., 0]).sum(0)
        colour = np.nanmedian(np.where(count[None, :, None] >= 2, samples, np.nan), 0) / gain
        hole = in_hole(name, gu.ravel(), gv.ravel(), size, wall)
        missing = ~np.isfinite(colour[:, 0]) & ~hole
        known = np.isfinite(colour[:, 0]) & ~hole
        fill = np.median(colour[known], 0) if known.any() else np.full(3, 200.0)
        colour[missing] = fill
        colour[hole] = 0
        image = colour.reshape(len(us), len(vs), 3).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out / f'face_{name}.png'), image)
        report.append(f'{name} {known.mean() / max(1e-9, (~hole).mean()):.0%}')
    print('faces seen: ' + ', '.join(report), flush=True)

    meta = {'name': 'box', 'kind': 'box', 'size_m': size, 'height_m': height, 'wall_m': wall, 'texel_m': TEXEL_M,
            'pose': {'x_m': float(best[0]), 'y_m': float(best[1]), 'yaw_rad': float(best[2])}, 'plane': plane.tolist(),
            'gain_vs_table': gain.tolist(), 'outline_overlap': 1 - value, 'views': len(names)}
    (out / 'box.json').write_text(json.dumps(meta, indent=1) + '\n')

    sheet = []
    for n in names[::max(1, len(names) // 6)][:6]:
        shown = images[n].copy()
        world = (box_pose @ edges.T).T[:, :3]
        local = (world - poses[n][:3, 3]) @ poses[n][:3, :3]
        uv = cv2.projectPoints(local, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
        for p in uv[np.isfinite(uv).all(1)]:
            cv2.circle(shown, tuple(int(v) for v in p), 1, (0, 0, 255), -1)
        sheet.append(shown)
    cv2.imwrite(str(out / 'check.jpg'), cv2.resize(np.vstack([np.hstack(sheet[:3]), np.hstack(sheet[3:6])]), None, fx=0.5, fy=0.5))


if __name__ == '__main__':
    main()
