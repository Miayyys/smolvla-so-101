"""Appearance of a flat object lying on the table, from scan_object.py: a colour texture with transparency on a plane.

    python scripts/recon_table.py transfer --scan runs/scan/key --source runs/scan/table_curtain
    python scripts/recon_object.py --scan runs/scan/key --source runs/scan/table_curtain

The object views follow poses of the workspace scan, so they take its refined poses (transfer) and lie in the frame of
its table texture. Every view is projected onto a plane HEIGHT_M above the fitted table and the median colour is kept
per cell. Transparency comes from the difference to the empty table texture of the workspace scan, after matching
brightness on the bare table around the object. The parts of the object above the plane (key ring, stacked keys) are
slightly blurred by parallax.

Output in <scan>/object: texture.png (BGRA), object.json (cell size, corner and plane, in the base frame), check.jpg.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import scan_object

CELL_M = 0.0004
MARGIN_M = 0.02
HEIGHT_M = 0.002  # about half the thickness of a key
MIN_VIEWS = 4
ALPHA_LOW, ALPHA_HIGH = 14.0, 32.0  # colour difference to the empty table, 0-255
MIN_PART_CELLS = 40
LIGHT_SMOOTH_M = 0.015
MAX_HOLE_M2 = 1e-4


def sample_views(scan, grid):
    """Colour of each grid point in every view (NaN where not seen), as views x cells x 3."""
    intrinsics = json.loads(Path(scan_object.WRIST_INTRINSICS).read_text())
    K, dist = np.array(intrinsics['K']), np.ravel(intrinsics['dist'])
    poses = json.loads((scan / 'colmap/poses_base_from_cam.json').read_text())
    jaws = cv2.imread(str(scan / 'jaw_mask.png'), cv2.IMREAD_GRAYSCALE) > 0
    samples = []
    for name, pose in sorted(poses.items()):
        pose = np.array(pose)
        local = (grid - pose[:3, 3]) @ pose[:3, :3]
        uv = cv2.projectPoints(local, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2).astype(np.float32)
        image = cv2.imread(str(scan / name)).astype(np.float32)
        padded = np.zeros((-(-len(uv) // 1024) * 1024, 2), np.float32)
        padded[:len(uv)] = uv
        maps = padded.reshape(-1, 1024, 2)  # remap takes maps below 32767 on a side
        colour = cv2.remap(image, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR).reshape(-1, 3)[:len(uv)]
        h, w = jaws.shape
        inside = (local[:, 2] > 0.02) & (uv[:, 0] >= 1) & (uv[:, 0] < w - 2) & (uv[:, 1] >= 1) & (uv[:, 1] < h - 2)
        pixels = np.clip(uv.round().astype(int), 0, [w - 1, h - 1])
        inside &= ~jaws[pixels[:, 1], pixels[:, 0]]
        colour[~inside] = np.nan
        samples.append(colour)
    return np.stack(samples), len(poses)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scan', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    scan, source = Path(args.scan), Path(args.source)
    info = json.loads((scan / 'wrist_views.json').read_text())['object']
    table = np.load(source / 'gs/table.npz')
    plane = table['plane']

    centre = np.array(info['obstacle_centre_m'][:2])
    half = np.array(info['obstacle_half_m'][:2]) - scan_object.OBSTACLE_MARGIN_M + MARGIN_M
    xs = np.arange(centre[0] - half[0], centre[0] + half[0], CELL_M)
    ys = np.arange(centre[1] - half[1], centre[1] + half[1], CELL_M)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')  # rows along x (away from the base), columns along y
    grid = np.stack([gx, gy, plane[0] * gx + plane[1] * gy + plane[2] + HEIGHT_M], -1).reshape(-1, 3)

    samples, n_views = sample_views(scan, grid)
    seen = np.isfinite(samples[..., 0]).sum(0)
    fused = np.nanmedian(samples, 0)
    fused[seen < MIN_VIEWS] = np.nan
    fused = fused.reshape(len(xs), len(ys), 3)

    texture = table['texture'][..., ::-1].astype(np.float32)  # stored RGB by recon_gs.py
    tx = np.clip(((gx - table['x_range'][0]) / table['cell']).astype(int), 0, texture.shape[0] - 1)
    ty = np.clip(((gy - table['y_range'][0]) / table['cell']).astype(int), 0, texture.shape[1] - 1)
    empty = cv2.GaussianBlur(texture, (3, 3), 0)[tx, ty]
    footprint = cv2.resize(cv2.imread(str(scan / 'top_mask.png'), cv2.IMREAD_GRAYSCALE), None, fx=1, fy=1)
    points = np.load(scan / 'top_points.npy')
    near = np.zeros(gx.shape, np.uint8)
    rows = ((points[:, 0] - xs[0]) / CELL_M).astype(int)
    cols = ((points[:, 1] - ys[0]) / CELL_M).astype(int)
    ok = (rows >= 0) & (rows < len(xs)) & (cols >= 0) & (cols < len(ys))
    near[rows[ok], cols[ok]] = 1
    near = cv2.dilate(near, np.ones((int(0.012 / CELL_M),) * 2, np.uint8)) > 0  # top camera outline plus 6 mm
    ring = ~near & np.isfinite(fused[..., 0])
    gain = np.median(fused[ring] / np.maximum(empty[ring], 1), axis=0)
    # The light falls differently than in the workspace scan, and not by one factor everywhere: the ratio to the empty
    # table is measured on the bare table around the object and carried under it by normalized convolution.
    ratio = np.where(ring[..., None], np.nan_to_num(fused) / np.maximum(empty, 1), 0).astype(np.float32)
    weight = ring.astype(np.float32)
    sigma = LIGHT_SMOOTH_M / CELL_M
    local = cv2.GaussianBlur(ratio, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(weight, (0, 0), sigma), 1e-6)[..., None]
    difference = np.abs(np.nan_to_num(fused) / local - empty).max(-1)  # in the units of the workspace scan
    alpha = np.clip((difference - ALPHA_LOW) / (ALPHA_HIGH - ALPHA_LOW), 0, 1)
    alpha[~near | ~np.isfinite(fused[..., 0])] = 0
    solid = (alpha > 0.5).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid)
    keep = np.isin(labels, [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= MIN_PART_CELLS])
    alpha = alpha * cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8))
    alpha[cv2.erode(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 1.0  # soft only within a cell or two of the edge
    # Small enclosed holes are dark or glossy parts of the object (black key head); larger ones are table seen through
    # a loop (key ring, wire tag).
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~keep).astype(np.uint8), connectivity=4)
    border = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
    holes = [i for i in range(1, count) if i not in border and stats[i, cv2.CC_STAT_AREA] * CELL_M ** 2 < MAX_HOLE_M2]
    alpha[np.isin(labels, holes)] = 1.0

    out = scan / 'object'
    out.mkdir(exist_ok=True)
    colour = np.nan_to_num(fused / local).clip(0, 255)  # as lit in the workspace scan
    cv2.imwrite(str(out / 'texture.png'), np.dstack([colour, alpha * 255]).round().astype(np.uint8))
    meta = {'name': info['name'], 'cell_m': CELL_M, 'x0_m': float(xs[0]), 'y0_m': float(ys[0]),
            'rows_along': 'x', 'height_above_table_m': HEIGHT_M, 'plane': plane.tolist(),
            'gain_vs_table': gain.tolist(), 'views': int(n_views),
            'extent_m': (np.ptp(gx[alpha > 0.5]), np.ptp(gy[alpha > 0.5])) if (alpha > 0.5).any() else None}
    meta['extent_m'] = [float(v) for v in meta['extent_m']] if meta['extent_m'] else None
    (out / 'object.json').write_text(json.dumps(meta, indent=1) + '\n')

    shown = [np.nan_to_num(fused), empty * gain, np.repeat(alpha[..., None] * 255, 3, -1),
             colour * alpha[..., None] + 255 * (1 - alpha[..., None]) * np.array([0.2, 0.8, 0.2])]
    sheet = np.hstack([cv2.resize(s.clip(0, 255).astype(np.uint8), None, fx=1.5, fy=1.5) for s in shown])
    cv2.imwrite(str(out / 'check.jpg'), sheet)
    print(f'{n_views} views, cells seen {np.median(seen):.0f} times (median), gain {np.round(gain, 2)}, '
          f'object {(alpha > 0.5).sum() * CELL_M ** 2 * 1e4:.1f} cm2, extent {np.round(np.array(meta["extent_m"] or [0, 0]) * 1000)} mm')


if __name__ == '__main__':
    main()
