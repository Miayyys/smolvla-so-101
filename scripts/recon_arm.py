"""Fit Gaussian splats bound to the arm's links from the top camera images of the arm capture. Runs in .venv-gs.

    .venv/bin/python scripts/recon_table.py armlinks --arm runs/scan/table_curtain/arm
    .venv-gs/bin/python scripts/recon_arm.py --arm runs/scan/table_curtain/arm --scan-top runs/scan/table_curtain/top_backgrounds

Splats live in link frames and follow the measured joints, so the model can be rendered at any pose. Each image is
rendered over the median of all images, which shows the table without the arm. Colours are scaled so that the table
matches the background photographs of the workspace scan, taken in different light. Every eighth stop is held out.
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

HOLD_OUT_EVERY = 8
INITIAL_SIZE_M = 0.0025
LEARNING_RATES = {'means': 2e-4, 'scales': 5e-3, 'quats': 2e-3, 'opacities': 5e-2, 'colors': 5e-3}
SILHOUETTE_MARGIN_PX = 40
OUTSIDE_ALPHA_WEIGHT = 0.2
TABLE_ROI = (60, 165, 300, 900)  # rows and columns of the wood strip: never covered by the arm in these poses


def matrix_to_quaternion(R):
    """Rotation matrices to w, x, y, z quaternions."""
    w = 0.5 * torch.sqrt(torch.clamp(1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2], min=1e-8))
    return torch.stack([w, (R[..., 2, 1] - R[..., 1, 2]) / (4 * w), (R[..., 0, 2] - R[..., 2, 0]) / (4 * w),
                        (R[..., 1, 0] - R[..., 0, 1]) / (4 * w)], -1)


def multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def render(params, owners, link_poses, camera, K, width, height):
    """Colours (over black) and alpha of the arm at the given link poses, seen by the camera."""
    rotation, origin = link_poses[owners, :3, :3], link_poses[owners, :3, 3]
    means = (rotation @ params['means'][..., None])[..., 0] + origin
    quats = multiply(matrix_to_quaternion(rotation), F.normalize(params['quats'], dim=-1))
    colours, alpha, _ = rasterization(means, quats, torch.exp(params['scales']), torch.sigmoid(params['opacities']),
                                      params['colors'], camera, K, width, height, packed=False)
    return colours[0], alpha[0]


def silhouettes(seeds, owners, poses, K, R, t, width, height):
    """Which pixels the arm may cover in each image: the projected seeds, closed up and with a margin."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SILHOUETTE_MARGIN_PX + 1,) * 2)
    masks = np.zeros((len(poses), height, width), bool)
    for index, link_poses in enumerate(poses):
        world = (link_poses[owners, :3, :3] @ seeds[..., None])[..., 0] + link_poses[owners, :3, 3]
        pixels = (K @ ((R @ world.T).T + t).T).T
        u, v = (pixels[:, 0] / pixels[:, 2]).astype(int), (pixels[:, 1] / pixels[:, 2]).astype(int)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        mask = np.zeros((height, width), np.uint8)
        mask[v[inside], u[inside]] = 1
        masks[index] = cv2.dilate(mask, kernel) > 0
    return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--scan-top', required=True)
    parser.add_argument('--steps', type=int, default=8000)
    args = parser.parse_args()

    device = 'cuda'
    arm = Path(args.arm)
    data = np.load(arm / 'links.npz')
    poses, seeds, owners_np, names = data['poses'], data['seeds'], data['owners'], list(data['names'])
    images = np.stack([cv2.cvtColor(cv2.imread(str(arm / f'top_{n}.png')), cv2.COLOR_BGR2RGB) for n in names]).astype(np.float32)
    height, width = images.shape[1:3]

    empty = np.median(np.stack([cv2.imread(f) for f in sorted(glob.glob(f'{args.scan_top}/top_*.png'))[1:]]), axis=0)[..., ::-1]
    background = np.median(images, axis=0)
    r0, r1, c0, c1 = TABLE_ROI
    gain = np.median(empty[r0:r1, c0:c1].reshape(-1, 3), axis=0) / np.median(background[r0:r1, c0:c1].reshape(-1, 3), axis=0)
    print(f'colour gain from this capture to the scan: {gain.round(2)}', flush=True)
    images = np.clip(images * gain, 0, 255)
    background = np.clip(background * gain, 0, 255)

    K = torch.tensor(data['K'], dtype=torch.float32, device=device)[None]
    camera = torch.eye(4, device=device)
    camera[:3, :3], camera[:3, 3] = torch.tensor(data['R']), torch.tensor(data['t'])
    camera = camera[None].float()
    masks_np = silhouettes(seeds, owners_np, poses, data['K'], data['R'], data['t'], width, height)
    print(f'silhouette covers {masks_np.mean():.1%} of an image on average', flush=True)

    count = len(seeds)
    quats = torch.zeros(count, 4, device=device)
    quats[:, 0] = 1
    params = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(torch.tensor(seeds, dtype=torch.float32, device=device)),
        'scales': torch.nn.Parameter(torch.full((count, 3), float(np.log(INITIAL_SIZE_M)), device=device)),
        'quats': torch.nn.Parameter(quats),
        'opacities': torch.nn.Parameter(torch.full((count,), float(torch.logit(torch.tensor(0.1))), device=device)),
        'colors': torch.nn.Parameter(torch.full((count, 3), 0.3, device=device)),
    })
    optimizer = torch.optim.Adam([{'params': [params[k]], 'lr': lr} for k, lr in LEARNING_RATES.items()], eps=1e-15)
    decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.05 ** (1 / args.steps))

    owners = torch.tensor(owners_np, device=device)
    poses_t = torch.tensor(poses, dtype=torch.float32, device=device)
    images_t = torch.from_numpy(images / 255).float()
    background_t = torch.tensor(background.copy() / 255, dtype=torch.float32, device=device)
    masks = torch.from_numpy(masks_np)
    held = np.arange(0, len(names), HOLD_OUT_EVERY)
    trained = np.setdiff1d(np.arange(len(names)), held)

    def compose(index):
        colours, alpha = render(params, owners, poses_t[index], camera, K, width, height)
        return colours + (1 - alpha) * background_t, alpha

    rng = np.random.default_rng(0)
    for step in range(args.steps):
        index = int(rng.choice(trained))
        real, mask = images_t[index].to(device), masks[index].to(device)
        rendered, alpha = compose(index)
        loss = (rendered - real).abs()[mask].mean() + OUTSIDE_ALPHA_WEIGHT * alpha[..., 0][~mask].mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        decay.step()
        if step % 1000 == 0 or step == args.steps - 1:
            print(f'step {step} loss {loss.item():.4f} memory {torch.cuda.max_memory_allocated() // 2 ** 20} MB', flush=True)

    with torch.no_grad():
        scores, rows = [], []
        for index in held:
            real, mask = images_t[index].to(device), masks[index].to(device)
            rendered, _ = compose(int(index))
            scores.append(-10 * torch.log10(((rendered.clamp(0, 1) - real) ** 2)[mask].mean()).item())
            rows.append((real.cpu().numpy(), rendered.clamp(0, 1).cpu().numpy()))
    print(f'held-out PSNR within the arm silhouette: median {np.median(scores):.2f} dB, min {np.min(scores):.2f} dB', flush=True)
    torch.save({'params': {k: v.detach().cpu() for k, v in params.items()}, 'owners': owners.cpu(), 'links': list(data['links']),
                'gain': gain}, arm / 'arm_splats.pt')
    picked = [rows[i] for i in np.linspace(0, len(rows) - 1, 5).astype(int)]
    out = np.vstack([np.hstack([r, c]) for r, c in picked])
    cv2.imwrite(str(arm / 'compare_arm.jpg'), cv2.cvtColor(cv2.resize((out * 255).astype(np.uint8), None, fx=0.6, fy=0.6), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f'saved {arm / "arm_splats.pt"}')


if __name__ == '__main__':
    main()
