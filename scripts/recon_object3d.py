"""Gaussian splats of an object with height (the tied key ring), from scan_object.py views. Runs in .venv-gs.

    .venv/bin/python scripts/recon_table.py transfer --scan runs/scan/key_tied --source runs/scan/table_curtain
    .venv/bin/python scripts/recon_object.py --scan runs/scan/key_tied --source runs/scan/table_curtain
    .venv-gs/bin/python scripts/recon_object3d.py --scan runs/scan/key_tied --source runs/scan/table_curtain

recon_object.py's flat texture smears an object that stands 1-2.5 cm tall; it is still run first because its outline
gives the object's place on the table. Here splats are fitted inside a box over that outline, each view rendered over
the twin's own image of the empty table (brought to the view's light), and scored only around the object. Splats are
kept above the table and over the flat outline, so that the object's shadow and the light are not built into it.
When a height map from an earlier run exists, its columns (twin_scene.object_columns) give the object's silhouette in
each view, and opacity outside it is penalised: otherwise the splats at the rim paint the shadow on the mat as a dark
halo, which then moves with the object.

The object frame has its origin on the table under the centre of the outline, z along the table normal, x and y as the
base frame at the scan. Output in <scan>/object: splats.pt (in the object frame), heights.npz (the object's height over
a grid, for the collision shape) and check3d.jpg (held-out views: photograph, render).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import recon_gs
import twin_scene

HOLD_OUT_EVERY = 8
SEEDS = 20000
INITIAL_SIZE_M = 0.0012
MARGIN_M = 0.012
MIN_ABOVE_TABLE_M = 0.0015
ROI_GROW_PX = 12
LEARNING_RATES = {'means': 1e-4, 'scales': 5e-3, 'quats': 2e-3, 'opacities': 5e-2, 'colors': 5e-3}
OPACITY_WEIGHT = 0.002
FOOTPRINT_GROW_M = 0.003
HEIGHT_CELL_M = 0.003
SOLID_OPACITY = 0.5
HULL_GROW_PX = 4
OUTSIDE_WEIGHT = 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scan', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--height', type=float, default=0.03, help='the box the splats may fill, in metres above the table')
    parser.add_argument('--steps', type=int, default=6000)
    args = parser.parse_args()

    device = 'cuda'
    scan, source = Path(args.scan), Path(args.source)
    meta = json.loads((scan / 'object/object.json').read_text())
    alpha = cv2.imread(str(scan / 'object/texture.png'), cv2.IMREAD_UNCHANGED)[..., 3] > 127
    rows, columns = np.nonzero(alpha)
    centre = meta['x0_m'] + rows.mean() * meta['cell_m'], meta['y0_m'] + columns.mean() * meta['cell_m']
    half = np.array([np.ptp(rows), np.ptp(columns)]) * meta['cell_m'] / 2 + MARGIN_M
    # in the reconstruction's frame, like the views' poses
    base_from_object = twin_scene.on_plane(np.load(source / 'gs/table.npz')['plane'], centre[0], centre[1], 0.0)
    print(f'object frame at x {centre[0]:.4f} y {centre[1]:.4f}, box {2 * half[0] * 1000:.0f} x {2 * half[1] * 1000:.0f} x '
          f'{args.height * 1000:.0f} mm', flush=True)

    names, images, cam_from_base, K_np, usable = recon_gs.load_views(scan)
    height, width = images.shape[1:3]
    gain = np.array(meta['gain_vs_table'])[::-1]  # stored for BGR
    images_t = torch.from_numpy(np.clip(images / gain, 0, 255) / 255).float()
    K = torch.tensor(K_np, dtype=torch.float32, device=device)[None]
    cameras = torch.tensor(cam_from_base, dtype=torch.float32, device=device)
    cam_from_object = cameras @ torch.tensor(base_from_object, dtype=torch.float32, device=device)

    splats = {k: v.to(device) for k, v in torch.load(source / 'gs/splats.pt').items()}
    table = np.load(source / 'gs/table.npz')
    layer = (torch.tensor(table['plane'], dtype=torch.float32, device=device),
             (torch.tensor(table['texture'], dtype=torch.float32, device=device) / 255).permute(2, 0, 1)[None])
    corners = np.array([[sx * half[0], sy * half[1], z, 1] for sx in (-1, 1) for sy in (-1, 1) for z in (0, args.height)])
    backgrounds, rois = [], []
    with torch.no_grad():
        for index in range(len(names)):
            backgrounds.append(recon_gs.composite(splats, cameras[index:index + 1], K, width, height, layer).cpu())
            local = (cam_from_base[index] @ base_from_object @ corners.T).T[:, :3]
            pixels = (local[:, :2] / local[:, 2:]) * [K_np[0, 0], K_np[1, 1]] + [K_np[0, 2], K_np[1, 2]]
            roi = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(roi, cv2.convexHull(pixels.astype(np.int32)), 1)
            rois.append((cv2.dilate(roi, np.ones((2 * ROI_GROW_PX + 1,) * 2, np.uint8)) > 0) & usable)
    backgrounds, rois = torch.stack(backgrounds), torch.from_numpy(np.stack(rois))
    print(f'{len(names)} views, the object\'s box covers {rois.float().mean():.1%} of a view', flush=True)
    hulls = None
    if (scan / 'object/heights.npz').exists():
        pillars = twin_scene.object_columns(scan)
        column_corners = np.array([[x + sx * hx, y + sy * hy, z, 1] for x, y, hx, hy, h in pillars
                                   for z in (0, h) for sx in (-1, 1) for sy in (-1, 1)])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HULL_GROW_PX + 1,) * 2)
        hulls = []
        for index in range(len(names)):
            local = (cam_from_base[index] @ base_from_object @ column_corners.T).T[:, :3]
            pixels = ((local[:, :2] / local[:, 2:]) * [K_np[0, 0], K_np[1, 1]] + [K_np[0, 2], K_np[1, 2]]).astype(np.int32)
            hull = np.zeros((height, width), np.uint8)
            for column in pixels.reshape(-1, 8, 2):
                cv2.fillConvexPoly(hull, cv2.convexHull(column), 1)
            hulls.append(cv2.dilate(hull, kernel) > 0)
        hulls = torch.from_numpy(np.stack(hulls))
        print(f'silhouettes from {len(pillars)} columns of the earlier height map', flush=True)

    # Splats stay over the flat outline (grown a little): elsewhere they would only paint the light and the shadow.
    grow = int(round(FOOTPRINT_GROW_M / meta['cell_m']))
    footprint = cv2.dilate(alpha.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)) > 0
    footprint_t = torch.from_numpy(footprint).to(device)
    origin = torch.tensor([rows.mean(), columns.mean()], dtype=torch.float32, device=device)

    def over_footprint(xy):
        cell = torch.round(xy / meta['cell_m'] + origin).long()
        inside = (cell[:, 0] >= 0) & (cell[:, 0] < footprint.shape[0]) & (cell[:, 1] >= 0) & (cell[:, 1] < footprint.shape[1])
        inside[inside.clone()] = footprint_t[cell[inside, 0], cell[inside, 1]]
        return inside

    rng = np.random.default_rng(0)
    cells = np.column_stack(np.nonzero(footprint))
    picked = cells[rng.integers(len(cells), size=SEEDS)] + rng.uniform(-0.5, 0.5, (SEEDS, 2))
    seeds = np.column_stack([(picked - [rows.mean(), columns.mean()]) * meta['cell_m'],
                             rng.uniform(MIN_ABOVE_TABLE_M, args.height, SEEDS)])
    quats = torch.zeros(SEEDS, 4, device=device)
    quats[:, 0] = 1
    params = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(torch.tensor(seeds, dtype=torch.float32, device=device)),
        'scales': torch.nn.Parameter(torch.full((SEEDS, 3), float(np.log(INITIAL_SIZE_M)), device=device)),
        'quats': torch.nn.Parameter(quats),
        'opacities': torch.nn.Parameter(torch.full((SEEDS,), float(torch.logit(torch.tensor(0.1))), device=device)),
        'colors': torch.nn.Parameter(torch.full((SEEDS, 3), 0.4, device=device)),
    })
    optimizer = torch.optim.Adam([{'params': [params[k]], 'lr': lr} for k, lr in LEARNING_RATES.items()], eps=1e-15)
    decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.05 ** (1 / args.steps))
    low = torch.tensor([-half[0], -half[1], MIN_ABOVE_TABLE_M], dtype=torch.float32, device=device)
    high = torch.tensor([half[0], half[1], args.height], dtype=torch.float32, device=device)
    held = np.arange(0, len(names), HOLD_OUT_EVERY)
    trained = np.setdiff1d(np.arange(len(names)), held)

    def compose(index):
        colours, alpha, _ = recon_gs.render(params, cam_from_object[index:index + 1], K, width, height)
        return colours[0] + (1 - alpha[0]) * backgrounds[index].to(device), alpha[0]

    for step in range(args.steps):
        index = int(rng.choice(trained))
        roi = rois[index].to(device)
        rendered, covered = compose(index)
        loss = (rendered - images_t[index].to(device)).abs()[roi].mean() + \
            OPACITY_WEIGHT * torch.sigmoid(params['opacities']).mean()
        if hulls is not None:
            outside = roi & ~hulls[index].to(device)
            loss = loss + OUTSIDE_WEIGHT * covered[..., 0][outside].mean()
        loss.backward()
        before = params['means'].detach().clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        decay.step()
        with torch.no_grad():
            moved = torch.minimum(torch.maximum(params['means'], low), high)
            params['means'].copy_(torch.where(over_footprint(moved[:, :2])[:, None], moved, before))
        if step % 1000 == 0 or step == args.steps - 1:
            print(f'step {step} loss {loss.item():.4f}', flush=True)

    with torch.no_grad():
        scores, empty, rows = [], [], []
        for index in held:
            roi, real = rois[index].to(device), images_t[index].to(device)
            rendered, _ = compose(int(index))
            scores.append(-10 * torch.log10(((rendered.clamp(0, 1) - real) ** 2)[roi].mean()).item())
            empty.append(-10 * torch.log10(((backgrounds[index].to(device) - real) ** 2)[roi].mean()).item())
            ys, xs = np.nonzero(rois[index].numpy())
            window = np.s_[max(ys.min() - 20, 0):ys.max() + 20, max(xs.min() - 20, 0):xs.max() + 20]
            pair = np.hstack([real.cpu().numpy()[window], rendered.clamp(0, 1).cpu().numpy()[window]])
            rows.append(cv2.resize(pair, (640, 320)))
        solid = params['means'][torch.sigmoid(params['opacities']) > SOLID_OPACITY].cpu().numpy()
    print(f'held-out PSNR around the object: {np.median(scores):.2f} dB with it, {np.median(empty):.2f} dB with the empty '
          f'table; {len(solid)} of {SEEDS} splats are solid', flush=True)

    cells = (np.ceil(2 * half / HEIGHT_CELL_M)).astype(int)
    heights = np.zeros(cells)
    index = np.clip(((solid[:, :2] + half) / HEIGHT_CELL_M).astype(int), 0, cells - 1)
    counts = np.zeros(cells, int)
    np.add.at(counts, (index[:, 0], index[:, 1]), 1)
    for (i, j) in zip(*np.nonzero(counts >= 3)):
        heights[i, j] = np.percentile(solid[(index[:, 0] == i) & (index[:, 1] == j), 2], 90)
    print(f'height map: {int((heights > 0).sum())} cells of {HEIGHT_CELL_M * 1000:.0f} mm, tallest {heights.max() * 1000:.0f} mm, '
          f'median {np.median(heights[heights > 0]) * 1000:.0f} mm', flush=True)

    out = scan / 'object'
    torch.save({'params': {k: v.detach().cpu() for k, v in params.items()}, 'base_from_object': base_from_object,
                'half_m': half, 'height_m': args.height}, out / 'splats.pt')
    np.savez(out / 'heights.npz', heights=heights, cell=HEIGHT_CELL_M, half=half)
    cv2.imwrite(str(out / 'check3d.jpg'), cv2.cvtColor((np.vstack(rows) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f'saved {out / "splats.pt"}')


if __name__ == '__main__':
    main()
