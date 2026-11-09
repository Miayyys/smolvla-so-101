"""Build the appearance model of the empty workspace from the wrist scan. Runs in .venv-gs (CUDA 13 torch and gsplat).

    .venv-gs/bin/python scripts/recon_gs.py --scan runs/scan/table_curtain --out runs/scan/table_curtain/gs

Uses the poses and points from recon_table.py, in the MJCF base frame. Images are undistorted to the pinhole model that
gsplat renders. Every eighth view is held out for evaluation.

The workspace is modelled in two layers. The table in front of the lamp is a plane with a texture: all views are
projected onto it and fused with a per-cell median, which removes the arm's shadow and the reflections on the black mat,
because they move from view to view. Everything beyond the plane (lamp, cables, wall) is fitted with 3D Gaussian splats,
with view-independent colours. Fitting splats to the table itself was tried and gives blotches: the mat has too little
texture. A render takes the table layer wherever a ray hits the plane and the splats elsewhere.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

INTRINSICS = 'runs/calib/wrist/intrinsics_standard.json'
HOLD_OUT_EVERY = 8
SCENE_SCALE_M = 0.5
INITIAL_SIZE_M = 0.004
TABLE_X_M, TABLE_Y_M = (0.0, 0.41), (-0.32, 0.33)  # the mat and wood strip in front of the lamp, flat and empty
CELL_M = 0.001
MIN_DEPTH_M = 0.03
MIN_FUSION_VIEWS = 6
SHADOW_RATIO, HIGHLIGHT_RATIO = 0.72, 1.35  # image brightness relative to the fused table
TRIM_START_STEP = 1000
TRIM_KEEP_FRACTION = 0.92  # moving shadows also fall outside the plane
MIN_BACKGROUND_FRACTION = 0.05  # views that are nearly all table have nothing to teach the splats
SSIM_WEIGHT = 0.2
MASK_GROW_PX = 2


def load_views(scan):
    poses = json.loads((scan / 'colmap/poses_base_from_cam.json').read_text())
    intrinsics = json.loads(Path(INTRINSICS).read_text())
    K, dist = np.array(intrinsics['K']), np.ravel(intrinsics['dist'])
    jaws = cv2.imread(str(scan / 'jaw_mask.png'), cv2.IMREAD_GRAYSCALE)
    usable = cv2.undistort(255 - jaws, K, dist)
    valid = cv2.undistort(np.full_like(jaws, 255), K, dist)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_GROW_PX + 1,) * 2)
    mask = cv2.erode(((usable > 250) & (valid > 250)).astype(np.uint8), kernel) > 0
    names = sorted(poses)
    images = np.stack([cv2.cvtColor(cv2.undistort(cv2.imread(str(scan / n)), K, dist), cv2.COLOR_BGR2RGB) for n in names])
    cam_from_base = np.stack([np.linalg.inv(np.array(poses[n])) for n in names])
    return names, images, cam_from_base, K, mask


def over_table(x, y):
    return (x >= TABLE_X_M[0]) & (x <= TABLE_X_M[1]) & (y >= TABLE_Y_M[0]) & (y <= TABLE_Y_M[1])


def fit_plane(xyz):
    """z = a x + b y + c through the reconstructed points over the table, ignoring points off the plane."""
    points = xyz[over_table(xyz[:, 0], xyz[:, 1])]
    keep = np.abs(points[:, 2] - np.median(points[:, 2])) < 0.01
    for _ in range(5):
        design = np.column_stack([points[keep, :2], np.ones(keep.sum())])
        plane = np.linalg.lstsq(design, points[keep, 2], rcond=None)[0]
        residual = points[:, 2] - np.column_stack([points[:, :2], np.ones(len(points))]) @ plane
        keep = np.abs(residual) < max(2.5 * np.median(np.abs(residual[keep])), 0.001)
    print(f'table plane: z = {plane[0]:.4f} x + {plane[1]:.4f} y + {plane[2] * 1000:.1f} mm, {keep.sum()} points, '
          f'residual {np.abs(residual[keep]).mean() * 1000:.2f} mm', flush=True)
    return plane


def fuse_table(images, cameras, K, mask, plane):
    """Median colour of each table cell over the views; shadows and highlights move, so the median ignores them."""
    xs = np.arange(TABLE_X_M[0], TABLE_X_M[1], CELL_M)
    ys = np.arange(TABLE_Y_M[0], TABLE_Y_M[1], CELL_M)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([grid_x, grid_y, plane[0] * grid_x + plane[1] * grid_y + plane[2], np.ones_like(grid_x)], -1)
    stack = np.full((len(images), *grid_x.shape, 3), np.nan, np.float32)
    usable = mask.astype(np.float32)
    for index, (image, camera) in enumerate(zip(images, cameras)):
        projected = (K @ (points @ camera.T)[..., :3][..., None])[..., 0]
        ahead = projected[..., 2] > MIN_DEPTH_M  # points behind the camera would project mirrored into the image
        depth = np.where(ahead, projected[..., 2], 1.0)
        u = np.where(ahead, projected[..., 0] / depth, -1).astype(np.float32)
        v = np.where(ahead, projected[..., 1] / depth, -1).astype(np.float32)
        valid = cv2.remap(usable, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
        colours = cv2.remap(image.astype(np.float32), u, v, cv2.INTER_LINEAR)
        stack[index][valid] = colours[valid]
    texture = np.nanmedian(stack, axis=0)
    missing = np.isfinite(stack[..., 0]).sum(0) < MIN_FUSION_VIEWS  # with few views the median keeps their shadows
    missing = cv2.dilate(missing.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    print(f'fused table texture {texture.shape[0]}x{texture.shape[1]} cells, {1 - missing.mean():.1%} from at least '
          f'{MIN_FUSION_VIEWS} views, the rest filled in', flush=True)
    filled = cv2.inpaint(np.nan_to_num(texture).astype(np.uint8), missing.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    texture[missing] = filled[missing]
    return texture


def transient_masks(images, cameras, K, plane, texture):
    """For each view, which pixels see the table, and which of those are much darker or brighter than the fused table."""
    height, width = images.shape[1:3]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    rays = np.linalg.inv(K) @ np.stack([u, v, np.ones_like(u)], -1)[..., None]
    normal = np.array([-plane[0], -plane[1], 1.0])
    masks, hits = [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for image, camera in zip(images, cameras):
        pose = np.linalg.inv(camera)
        directions = (pose[:3, :3] @ rays)[..., 0]
        depth = (plane[2] - normal @ pose[:3, 3]) / (directions @ normal)
        hit = pose[:3, 3] + directions * depth[..., None]
        row = np.round((hit[..., 0] - TABLE_X_M[0]) / CELL_M).astype(int)
        column = np.round((hit[..., 1] - TABLE_Y_M[0]) / CELL_M).astype(int)
        inside = (depth > 0) & (row >= 0) & (row < texture.shape[0]) & (column >= 0) & (column < texture.shape[1])
        reference = np.full((height, width), np.nan, np.float32)
        reference[inside] = texture[row[inside], column[inside]].mean(-1)
        ratio = (image.mean(-1) + 1) / (reference + 1)
        with np.errstate(invalid='ignore'):
            transient = ((ratio < SHADOW_RATIO) | (ratio > HIGHLIGHT_RATIO)).astype(np.uint8)
        transient = cv2.dilate(cv2.morphologyEx(transient, cv2.MORPH_OPEN, kernel), kernel) > 0
        masks.append(transient)
        hits.append(inside)
    return np.stack(masks), np.stack(hits)


def seed_points(scan, device):
    data = np.load(scan / 'colmap/points_base.npz')
    outside = ~over_table(data['xyz'][:, 0], data['xyz'][:, 1])
    print(f'{outside.sum()} seed points beyond the table', flush=True)
    return (torch.tensor(data['xyz'][outside], dtype=torch.float32, device=device),
            torch.tensor(data['rgb'][outside] / 255.0, dtype=torch.float32, device=device))


def table_layer(camera, K, width, height, plane, texture):
    """Colour of the table plane seen by a camera, and which pixels hit it. texture is 1 x 3 x rows x columns in 0-1."""
    device = camera.device
    u, v = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device), indexing='xy')
    pixels = torch.stack([u, v, torch.ones_like(u)], -1).float()
    pose = torch.inverse(camera[0])
    rays = (pose[:3, :3] @ torch.linalg.inv(K[0]) @ pixels[..., None])[..., 0]
    normal = torch.stack([-plane[0], -plane[1], torch.ones_like(plane[0])])
    depth = (plane[2] - normal @ pose[:3, 3]) / (rays @ normal)
    hit = pose[:3, 3] + rays * depth[..., None]
    rows, columns = texture.shape[2:]
    across = 2 * (hit[..., 1] - TABLE_Y_M[0]) / (CELL_M * (columns - 1)) - 1
    along = 2 * (hit[..., 0] - TABLE_X_M[0]) / (CELL_M * (rows - 1)) - 1
    colour = F.grid_sample(texture, torch.stack([across, along], -1)[None], align_corners=True)[0].permute(1, 2, 0)
    return colour, (depth > 0) & (across.abs() <= 1) & (along.abs() <= 1)


def ssim(a, b):
    """Mean SSIM of two images, channels first, with an 11-pixel Gaussian window."""
    x = torch.arange(11, dtype=a.dtype, device=a.device) - 5
    g = torch.exp(-x ** 2 / 4.5)
    g = (g / g.sum())[None]
    window = (g.T @ g)[None, None].repeat(a.shape[1], 1, 1, 1)
    blur = lambda v: F.conv2d(v, window, padding=5, groups=a.shape[1])
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    return (((2 * mu_a * mu_b + 1e-4) * (2 * cov + 9e-4)) / ((mu_a ** 2 + mu_b ** 2 + 1e-4) * (var_a + var_b + 9e-4))).mean()


def render(params, cam_from_base, K, width, height, **kwargs):
    return rasterization(params['means'], F.normalize(params['quats'], dim=-1), torch.exp(params['scales']),
                         torch.sigmoid(params['opacities']), params['colors'], cam_from_base, K, width, height,
                         packed=False, **kwargs)


def composite(params, camera, K, width, height, layer):
    rendered, _, _ = render(params, camera, K, width, height)
    colour, on_table = table_layer(camera, K, width, height, *layer)
    return torch.where(on_table[..., None], colour, rendered[0].clamp(0, 1))


def evaluate(params, images, cameras, K, masks, device, width, height, layer):
    errors = []
    with torch.no_grad():
        for image, camera, mask in zip(images, cameras, masks):
            colours = composite(params, camera[None].to(device), K, width, height, layer)
            real = image.to(device, dtype=torch.float32) / 255
            errors.append(((colours - real) ** 2)[mask].mean().item())
    return float(np.median([-10 * np.log10(e) for e in errors]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scan', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--eval-every', type=int, default=3000)
    parser.add_argument('--max-splats', type=int, default=1200000)
    args = parser.parse_args()

    device = 'cuda'
    scan, out = Path(args.scan), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names, images, cameras, K, mask_np = load_views(scan)
    height, width = images.shape[1:3]
    held = np.arange(0, len(names), HOLD_OUT_EVERY)
    trained = np.setdiff1d(np.arange(len(names)), held)
    plane = fit_plane(np.load(scan / 'colmap/points_base.npz')['xyz'])
    texture = fuse_table(images[trained], cameras[trained], K, mask_np, plane)
    transient, on_table = transient_masks(images, cameras, K, plane, texture)
    print(f'transient pixels: median {np.median(transient.mean((1, 2))):.1%} of a view', flush=True)
    np.savez(out / 'table.npz', plane=plane, texture=texture, x_range=TABLE_X_M, y_range=TABLE_Y_M, cell=CELL_M)
    cv2.imwrite(str(out / 'table_texture.png'), cv2.cvtColor(np.nan_to_num(texture).astype(np.uint8), cv2.COLOR_RGB2BGR))
    transient, on_table = torch.from_numpy(transient), torch.from_numpy(on_table)
    background_views = trained[((~on_table[trained]) & torch.from_numpy(mask_np)).float().mean((1, 2)).numpy() > MIN_BACKGROUND_FRACTION]
    print(f'{len(background_views)} of {len(trained)} views show enough beyond the table', flush=True)
    images = torch.from_numpy(images)
    cameras = torch.tensor(cameras, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32, device=device)[None]
    mask = torch.from_numpy(mask_np).to(device)
    print(f'{len(trained)} views to fit, {len(held)} held out, usable pixels {mask_np.mean():.1%}', flush=True)

    means, colours = seed_points(scan, device)
    layer = (torch.tensor(plane, dtype=torch.float32, device=device),
             (torch.tensor(texture, dtype=torch.float32, device=device) / 255).permute(2, 0, 1)[None])

    count = len(means)
    quats = torch.zeros(count, 4, device=device)
    quats[:, 0] = 1
    params = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(means),
        'scales': torch.nn.Parameter(torch.full((count, 3), float(np.log(INITIAL_SIZE_M)), device=device)),
        'quats': torch.nn.Parameter(quats),
        'opacities': torch.nn.Parameter(torch.full((count,), float(torch.logit(torch.tensor(0.1))), device=device)),
        'colors': torch.nn.Parameter(colours),
    })
    rates = {'means': 1.6e-4 * SCENE_SCALE_M, 'scales': 2.5e-3, 'quats': 1e-3, 'opacities': 2.5e-2, 'colors': 2.5e-3}
    optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in rates.items()}
    final_rate = rates['means'] * 0.01
    means_decay = torch.optim.lr_scheduler.ExponentialLR(optimizers['means'], gamma=0.01 ** (1 / args.steps))
    strategy = DefaultStrategy(refine_stop_iter=args.steps // 2, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=SCENE_SCALE_M)
    print(f'means learning rate goes from {rates["means"]:.1e} to {final_rate:.1e}', flush=True)

    rng = np.random.default_rng(0)
    for step in range(args.steps):
        index = int(rng.choice(background_views))
        camera = cameras[index][None].to(device)
        real = images[index].to(device, dtype=torch.float32) / 255
        rendered, _, info = render(params, camera, K, width, height, absgrad=strategy.absgrad)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        difference = (rendered[0] - real).abs()
        keep = mask & ~on_table[index].to(device)
        if step >= TRIM_START_STEP:
            error = difference.detach().mean(-1)
            keep = keep & (error <= torch.quantile(error[keep][::16], TRIM_KEEP_FRACTION))
        l1 = difference[keep].mean()
        weight = keep[None, None].float()
        similarity = ssim((rendered[0] * weight[0, 0, ..., None]).permute(2, 0, 1)[None],
                          (real * weight[0, 0, ..., None]).permute(2, 0, 1)[None])
        loss = (1 - SSIM_WEIGHT) * l1 + SSIM_WEIGHT * (1 - similarity)
        loss.backward()
        if len(params['means']) < args.max_splats:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            params['opacities'][over_table(params['means'][:, 0], params['means'][:, 1])] = -20.0  # the plane covers these
        means_decay.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f'step {step} loss {loss.item():.4f} splats {len(params["means"])} '
                  f'memory {torch.cuda.max_memory_allocated() // 2 ** 20} MB', flush=True)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            psnr = evaluate(params, images[held], cameras[held], K, [mask] * len(held), device, width, height, layer)
            steady = [mask & ~t.to(device) for t in transient[held]]
            psnr_steady = evaluate(params, images[held], cameras[held], K, steady, device, width, height, layer)
            print(f'step {step + 1}: held-out PSNR (median) all {psnr:.2f} dB, without shadows and highlights '
                  f'{psnr_steady:.2f} dB', flush=True)
            torch.save({k: v.detach().cpu() for k, v in params.items()}, out / 'splats.pt')

    with torch.no_grad():
        rows = []
        for index in held[np.linspace(0, len(held) - 1, 6).astype(int)]:
            rendered = composite(params, cameras[index][None].to(device), K, width, height, layer)
            rows.append(np.hstack([images[index].numpy(), (rendered.cpu().numpy() * 255).astype(np.uint8)]))
        cv2.imwrite(str(out / 'compare_held.jpg'), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f'{len(params["means"])} splats saved to {out / "splats.pt"}')


if __name__ == '__main__':
    main()
