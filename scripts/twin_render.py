"""Render the wrist and top camera images of the digital twin from its reconstructed parts. Runs in .venv-gs.

    .venv-gs/bin/python scripts/twin_render.py --scan runs/scan/table_curtain --arm runs/scan/table_curtain/arm --out runs/twin/check_day

Wrist view: the table texture and background splats are rendered as a pinhole image, warped to the camera's distortion,
and the jaws are laid over them. Top view: the arm's splats over a photograph of the empty workspace. Objects are placed
by their 4 x 4 pose in the base frame and drawn in both views, nearest in front: a box from its face textures
(recon_box.py), an object with height from its splats (recon_object3d.py), a flat one from its texture
(recon_object.py). The command compares renders with held-out photographs.
"""

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import recon_arm
import recon_box
import recon_gs

WRIST_INTRINSICS = 'runs/calib/wrist/intrinsics_standard.json'
REFERENCE_TOP = 'runs/scan/table_curtain/top_backgrounds'  # the arm's colours were scaled to match these photographs
TABLE_ROI = recon_arm.TABLE_ROI
HOLD_OUT_EVERY = 8
JAW_REFERENCE_OPENING = 39.6  # the opening used for the workspace scans
SATURATED = 200
DARK = 30
LIGHT = np.array([0.3, 0.2, 1.0])  # direction towards the light, base frame, for generated objects
AMBIENT = 0.55


def median_background(directory):
    """Median of the empty-workspace photographs; the first is skipped, the camera has not settled yet."""
    return np.median(np.stack([cv2.imread(f) for f in sorted(glob.glob(f'{directory}/top_*.png'))[1:]]), axis=0)


class Twin:
    def __init__(self, scan, arm, device='cuda'):
        self.device = device
        self.objects = {}
        scan, arm = Path(scan), Path(arm)
        intrinsics = json.loads(Path(WRIST_INTRINSICS).read_text())
        self.K_np, self.dist = np.array(intrinsics['K']), np.ravel(intrinsics['dist'])
        self.wrist_K = torch.tensor(self.K_np, dtype=torch.float32, device=device)[None]
        self.width, self.height = 640, 480
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        points = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64)
        normal = cv2.undistortPoints(points, self.K_np, self.dist).reshape(self.height, self.width, 2)
        self.map_x = (self.K_np[0, 0] * normal[..., 0] + self.K_np[0, 2]).astype(np.float32)
        self.map_y = (self.K_np[1, 1] * normal[..., 1] + self.K_np[1, 2]).astype(np.float32)

        self.splats = {k: v.to(device) for k, v in torch.load(scan / 'gs/splats.pt').items()}
        table = np.load(scan / 'gs/table.npz')
        texture = torch.tensor(table['texture'], dtype=torch.float32, device=device) / 255
        self.layer = (torch.tensor(table['plane'], dtype=torch.float32, device=device), texture.permute(2, 0, 1)[None])

        jaws = np.load(arm / 'jaw_layers.npz')
        self.jaw_openings, self.jaw_layers = jaws['openings'], jaws['layers'].astype(np.float32)
        self.jaw_gain = self.match_jaws(cv2.imread(str(scan / 'median.png')).astype(np.float32))

        links = np.load(arm / 'links.npz')
        saved = torch.load(arm / 'arm_splats.pt', weights_only=False)
        self.arm_params = {k: v.to(device) for k, v in saved['params'].items()}
        self.owners = saved['owners'].to(device)
        self.top_K = torch.tensor(links['K'], dtype=torch.float32, device=device)[None]
        camera = torch.eye(4, device=device)
        camera[:3, :3], camera[:3, 3] = torch.tensor(links['R']), torch.tensor(links['t'])
        self.top_camera = camera[None].float()
        self.background = median_background(scan / 'top_backgrounds')
        reference = median_background(REFERENCE_TOP)
        r0, r1, c0, c1 = TABLE_ROI
        scale = np.median(self.background[r0:r1, c0:c1].reshape(-1, 3), 0) / np.median(reference[r0:r1, c0:c1].reshape(-1, 3), 0)
        self.arm_params['colors'] = self.arm_params['colors'] * torch.tensor(scale[::-1].copy(), dtype=torch.float32, device=device)
        self.background_t = torch.tensor(self.background[..., ::-1].copy() / 255, dtype=torch.float32, device=device)

    def match_jaws(self, median):
        """Per-channel scale from the arm capture to this scan's light, measured on the jaws themselves."""
        layer = self.jaw_at(JAW_REFERENCE_OPENING)
        captured = layer[..., :3] * 255
        usable = (layer[..., 3] > 0.99) & (median.max(-1) < SATURATED) & (captured.max(-1) < SATURATED) \
            & (median.min(-1) > DARK) & (captured.min(-1) > DARK)  # ratios of near-black pixels mean nothing
        gain = np.median(median[usable], axis=0) / np.median(layer[..., :3][usable] * 255, axis=0)
        print(f'jaw gain {gain.round(2)} from {int(usable.sum())} pixels', flush=True)
        return gain

    def jaw_at(self, opening):
        """Jaw image at an opening, blended from the two nearest captures, as colour 0-1 (BGR) and alpha."""
        opening = float(np.clip(opening, self.jaw_openings[0], self.jaw_openings[-1]))
        upper = int(np.clip(np.searchsorted(self.jaw_openings, opening), 1, len(self.jaw_openings) - 1))
        weight = (opening - self.jaw_openings[upper - 1]) / (self.jaw_openings[upper] - self.jaw_openings[upper - 1])
        a, b = self.jaw_layers[upper - 1] / 255, self.jaw_layers[upper] / 255
        alpha = (1 - weight) * a[..., 3:] + weight * b[..., 3:]
        premultiplied = (1 - weight) * a[..., :3] * a[..., 3:] + weight * b[..., :3] * b[..., 3:]
        colour = np.divide(premultiplied, alpha, out=np.zeros_like(premultiplied), where=alpha > 1e-3)
        return np.dstack([colour, alpha])

    def load_box(self, directory):
        """A box from recon_box.py. Its origin is the centre of its base; returns its name and the scanned pose."""
        meta = json.loads((Path(directory) / 'object/box.json').read_text())
        textures = []
        for name, *_ in recon_box.faces(meta['size_m'], meta['height_m'], meta['wall_m']):
            image = cv2.imread(str(Path(directory) / f'object/face_{name}.png'))[..., ::-1].astype(np.float32) / 255
            textures.append(torch.tensor(image.copy(), device=self.device).permute(2, 0, 1)[None])
        self.objects[meta['name']] = {'kind': 'box', 'meta': meta, 'textures': textures}
        return meta['name'], meta['pose']

    def rays(self, camera, K, width, height):
        """Origin and per-pixel rays (base frame, scaled so that the distance along a ray is the camera depth)."""
        u, v = torch.meshgrid(torch.arange(width, device=self.device), torch.arange(height, device=self.device), indexing='xy')
        pixels = torch.stack([u, v, torch.ones_like(u)], -1).float().reshape(-1, 3)
        pose = torch.inverse(camera[0])
        return pose[:3, 3], (pose[:3, :3] @ torch.linalg.inv(K[0]) @ pixels.T).T

    def box_layer(self, camera, name, pose, K, width, height):
        """Colour (RGB), coverage and depth of a box whose base centre has the pose (4 x 4, base frame)."""
        item = self.objects[name]
        meta = item['meta']
        size, wall = meta['size_m'], meta['wall_m']
        box_from_base = torch.tensor(np.linalg.inv(pose), dtype=torch.float32, device=self.device)
        start_point, rays = self.rays(camera, K, width, height)
        rays = rays @ box_from_base[:3, :3].T
        origin = box_from_base[:3, :3] @ start_point + box_from_base[:3, 3]
        best = torch.full((len(rays),), float('inf'), device=self.device)
        colour = torch.zeros(len(rays), 3, device=self.device)
        for k, (face, start, u_axis, v_axis, u_len, v_len) in enumerate(recon_box.faces(size, meta['height_m'], wall)):
            start, u_axis, v_axis = (torch.tensor(np.asarray(a, np.float32), device=self.device) for a in (start, u_axis, v_axis))
            normal = torch.linalg.cross(u_axis, v_axis)
            facing = rays @ normal
            t = ((start - origin) @ normal) / facing
            hit = origin + t[:, None] * rays - start
            a, b = hit @ u_axis, hit @ v_axis
            ok = (facing < 0) & (t > 0) & (a >= 0) & (a <= u_len) & (b >= 0) & (b <= v_len) & (t < best)
            if face == 'rim':
                ok &= ~((a > wall) & (a < size - wall) & (b > wall) & (b < size - wall))
            grid = torch.stack([2 * b / v_len - 1, 2 * a / u_len - 1], -1)  # texture rows along u, columns along v
            sampled = F.grid_sample(item['textures'][k], grid[None, None], align_corners=False)[0, :, 0].T
            colour = torch.where(ok[:, None], sampled, colour)
            best = torch.where(ok, t, best)
        covered = torch.isfinite(best).float()
        return (colour.reshape(height, width, 3), covered.reshape(height, width, 1),
                torch.where(covered > 0, best, torch.zeros_like(best)).reshape(height, width, 1))

    def splat_layer(self, camera, name, pose, K, width, height):
        """Colour (RGB, not premultiplied), alpha and depth of an object from recon_object3d.py at the pose."""
        item = self.objects[name]
        cam_from_object = camera @ torch.tensor(pose, dtype=torch.float32, device=self.device)[None]
        rendered, alpha, _ = recon_gs.render(item['params'], cam_from_object, K, width, height, render_mode='RGB+ED')
        alpha = alpha[0]
        colour = rendered[0, ..., :3] / alpha.clamp(min=1e-4)
        return colour.clamp(0, 1), alpha, rendered[0, ..., 3:]

    def load_object(self, directory):
        """An object scan; returns its name and its pose (4 x 4, base frame) while it was scanned."""
        directory = Path(directory)
        if (directory / 'object/box.json').exists():
            name, placement = self.load_box(directory)
            return name, recon_box.base_from_box(placement['x_m'], placement['y_m'], placement['yaw_rad'],
                                                 self.layer[0].cpu().numpy())
        meta = json.loads((directory / 'object/object.json').read_text())
        if (directory / 'object/splats.pt').exists():
            saved = torch.load(directory / 'object/splats.pt', weights_only=False)
            self.objects[meta['name']] = {'kind': 'splats', 'meta': meta,
                                          'params': {k: v.to(self.device) for k, v in saved['params'].items()}}
            return meta['name'], np.asarray(saved['base_from_object'])
        texture = cv2.imread(str(directory / 'object/texture.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        rows, columns = np.nonzero(texture[..., 3] > 0.5)
        centre = (meta['x0_m'] + rows.mean() * meta['cell_m'], meta['y0_m'] + columns.mean() * meta['cell_m'])
        rgba = np.dstack([texture[..., 2::-1], texture[..., 3:]])
        self.objects[meta['name']] = {'kind': 'flat', 'texture': torch.tensor(rgba, device=self.device).permute(2, 0, 1)[None],
                                      'meta': meta, 'centre': centre}
        return meta['name'], recon_box.base_from_box(*centre, 0.0, self.layer[0].cpu().numpy())

    def flat_layer(self, camera, name, pose, K, width, height):
        """Colour (RGB), alpha and depth of a flat object from recon_object.py whose origin has the pose; only the
        position and the turn about z are used, the texture lies on the reconstructed table."""
        x, y, yaw = pose[0, 3], pose[1, 3], np.arctan2(pose[1, 0], pose[0, 0])
        item = self.objects[name]
        meta, (cx, cy), texture = item['meta'], item['centre'], item['texture']
        a, b, c = (float(v) for v in self.layer[0])
        plane = torch.tensor([a, b, c + meta['height_above_table_m']], device=self.device)
        origin, rays = self.rays(camera, K, width, height)
        rays = rays.reshape(height, width, 3)
        normal = torch.stack([-plane[0], -plane[1], torch.ones_like(plane[0])])
        depth = (plane[2] - normal @ origin) / (rays @ normal)
        hit = origin + rays * depth[..., None]
        dx, dy = hit[..., 0] - x, hit[..., 1] - y
        sx = np.cos(yaw) * dx + np.sin(yaw) * dy + cx  # into the frame of the scan
        sy = -np.sin(yaw) * dx + np.cos(yaw) * dy + cy
        rows, columns = texture.shape[2:]
        along = 2 * (sx - meta['x0_m']) / (meta['cell_m'] * (rows - 1)) - 1
        across = 2 * (sy - meta['y0_m']) / (meta['cell_m'] * (columns - 1)) - 1
        sampled = F.grid_sample(texture, torch.stack([across, along], -1)[None], align_corners=True,
                                padding_mode='zeros')[0].permute(1, 2, 0)
        alpha = sampled[..., 3:] * (depth > 0)[..., None]
        return sampled[..., :3], alpha, depth[..., None]

    def add_shape(self, name, pieces, colour):
        """A generated object (twin_shapes.Shape): its convex pieces (object frame) in one flat colour (RGB 0-1)."""
        planes = [hull_planes(np.asarray(p, float)) for p in pieces]
        most = max(len(p) for p in planes)
        padded = np.zeros((len(planes), most, 4))
        padded[..., 3] = 1.0  # 0 . x <= 1 holds everywhere: a face that does not exist
        for k, p in enumerate(planes):
            padded[k, :len(p)] = p
        self.objects[name] = {'kind': 'mesh', 'planes': torch.tensor(padded, dtype=torch.float32, device=self.device),
                              'colour': torch.tensor(colour, dtype=torch.float32, device=self.device)}

    def mesh_layer(self, camera, name, pose, K, width, height):
        """Colour (RGB), coverage and depth of a generated object by ray casting its convex pieces; pose may scale
        (into the wrist reconstruction's frame). Diffuse shading under a light from above."""
        item = self.objects[name]
        object_from_base = torch.tensor(np.linalg.inv(pose), dtype=torch.float32, device=self.device)
        rotation = np.asarray(pose)[:3, :3]
        rotation = rotation / np.linalg.norm(rotation[:, 0])
        light = torch.tensor(rotation.T @ (LIGHT / np.linalg.norm(LIGHT)), dtype=torch.float32, device=self.device)
        start, rays = self.rays(camera, K, width, height)
        origin = object_from_base[:3, :3] @ start + object_from_base[:3, 3]
        rays = rays @ object_from_base[:3, :3].T
        best = torch.full((len(rays),), float('inf'), device=self.device)
        shade = torch.zeros(len(rays), device=self.device)
        for planes in item['planes']:
            normals, bounds = planes[:, :3], planes[:, 3]
            denominator = rays @ normals.T  # (pixels, faces)
            numerator = bounds - normals @ origin
            t = numerator / torch.where(denominator.abs() < 1e-12, torch.full_like(denominator, 1e-12), denominator)
            entering = torch.where(denominator < 0, t, torch.full_like(t, -float('inf')))
            t_in, face = entering.max(1)
            t_out = torch.where(denominator > 0, t, torch.full_like(t, float('inf'))).min(1).values
            outside = ((denominator.abs() < 1e-12) & (numerator < 0)).any(1)
            hit = (t_in <= t_out) & (t_out > 0) & (t_in > 0) & ~outside & (t_in < best)
            best = torch.where(hit, t_in, best)
            shade = torch.where(hit, (normals[face] @ light).clamp(min=0), shade)
        covered = torch.isfinite(best)
        colour = item['colour'] * (AMBIENT + (1 - AMBIENT) * shade)[:, None]
        return (colour.reshape(height, width, 3), covered.float().reshape(height, width, 1),
                torch.where(covered, best, torch.zeros_like(best)).reshape(height, width, 1))

    def objects_over(self, image, camera, K, width, height, objects, gains=None):
        """The objects, given as (name, pose) pairs, laid over an image (RGB, 0-1), nearer ones in front."""
        if not objects:
            return image, torch.zeros_like(image[..., :1]), {}
        layers = {'box': self.box_layer, 'splats': self.splat_layer, 'flat': self.flat_layer, 'mesh': self.mesh_layer}
        colours, alphas, depths, masks = [], [], [], {}
        for name, pose in objects:
            colour, alpha, depth = layers[self.objects[name]['kind']](camera, name, pose, K, width, height)
            if gains and name in gains:
                colour = colour * torch.tensor(gains[name], dtype=torch.float32, device=self.device)
            colours.append(colour)
            alphas.append(alpha)
            depths.append(torch.where(alpha > 1e-3, depth, torch.full_like(depth, float('inf'))))
            masks[name] = alpha[..., 0]
        order = torch.argsort(torch.stack(depths), dim=0, descending=True)  # farthest first
        colours, alphas = torch.stack(colours), torch.stack(alphas)
        covered = torch.zeros_like(image[..., :1])
        for rank in range(len(objects)):
            index = order[rank]
            colour = torch.gather(colours, 0, index[None].expand(1, *colours.shape[1:]))[0]
            alpha = torch.gather(alphas, 0, index[None])[0]
            image = colour * alpha + image * (1 - alpha)
            covered = alpha + covered * (1 - alpha)
        return image, covered, masks

    @torch.no_grad()
    def wrist(self, base_from_cam, opening, objects=()):
        """Wrist image (BGR, uint8) for a camera pose in the base frame, a jaw opening in percent and objects as
        (name, pose) pairs."""
        camera = torch.tensor(np.linalg.inv(base_from_cam), dtype=torch.float32, device=self.device)[None]
        pinhole = recon_gs.composite(self.splats, camera, self.wrist_K, self.width, self.height, self.layer)
        pinhole, _, _ = self.objects_over(pinhole, camera, self.wrist_K, self.width, self.height, list(objects))
        pinhole = pinhole.cpu().numpy()
        scene = cv2.remap(pinhole[..., ::-1].copy(), self.map_x, self.map_y, cv2.INTER_LINEAR)
        jaw = self.jaw_at(opening)
        colour = np.clip(jaw[..., :3] * self.jaw_gain, 0, 1)
        return ((colour * jaw[..., 3:] + scene * (1 - jaw[..., 3:])) * 255).clip(0, 255).astype(np.uint8)

    @torch.no_grad()
    def top(self, link_poses, objects=(), gains=None, masks=False):
        """Top image (BGR, uint8) for the link poses in the base frame, with objects as (name, pose) pairs drawn under
        the arm; gains: per-object colour scale (RGB) for this camera. With masks, also the arm's alpha and each
        object's alpha (before the arm is laid over them)."""
        poses = torch.tensor(link_poses, dtype=torch.float32, device=self.device)
        colours, alpha = recon_arm.render(self.arm_params, self.owners, poses, self.top_camera, self.top_K, 1280, 720)
        scene, _, object_masks = self.objects_over(self.background_t, self.top_camera, self.top_K, 1280, 720, list(objects),
                                                   gains)
        image = (colours + (1 - alpha) * scene).clamp(0, 1).cpu().numpy()
        image = (image[..., ::-1] * 255).astype(np.uint8)
        if not masks:
            return image
        return image, alpha[..., 0].cpu().numpy(), {k: v.cpu().numpy() for k, v in object_masks.items()}


def hull_planes(points, tolerance=1e-7):
    """Faces of the convex hull of a few points as rows (n, b), unit n pointing out, n . x <= b inside."""
    out = []
    count = len(points)
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                normal = np.cross(points[j] - points[i], points[k] - points[i])
                if np.linalg.norm(normal) < 1e-12:
                    continue
                normal /= np.linalg.norm(normal)
                side = points @ normal - normal @ points[i]
                if (side <= tolerance).all():
                    plane = np.append(normal, normal @ points[i])
                elif (side >= -tolerance).all():
                    plane = np.append(-normal, -normal @ points[i])
                else:
                    continue
                if not any(np.allclose(plane, other, atol=1e-6) for other in out):
                    out.append(plane)
    return np.array(out)


def psnr(a, b, mask=None):
    difference = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    return float(-10 * np.log10((difference[mask] if mask is not None else difference).mean() / 255 ** 2))


def check_object(twin, directory, out):
    """Renders with the object where it lay during its scan, beside the photographs brought to the workspace scan's light."""
    name, pose = twin.load_object(directory)
    meta = directory / 'object/box.json'
    gain = np.array(json.loads((meta if meta.exists() else directory / 'object/object.json').read_text())['gain_vs_table'])
    poses = json.loads((directory / 'colmap/poses_base_from_cam.json').read_text())
    names = sorted(poses)[::max(1, len(poses) // 6)]
    rows, scores = [], []
    for view in names:
        real = np.clip(cv2.imread(str(directory / view)) / gain, 0, 255).astype(np.uint8)
        camera = np.array(poses[view])
        rendered = twin.wrist(camera, JAW_REFERENCE_OPENING, [(name, pose)])
        empty = twin.wrist(camera, JAW_REFERENCE_OPENING)
        scores.append((psnr(real, rendered), psnr(real, empty)))
        rows.append(np.hstack([real, rendered, empty]))
    scores = np.median(scores, 0)
    print(f'{name}: {len(names)} views, whole image {scores[0]:.2f} dB with the object, {scores[1]:.2f} dB without', flush=True)
    cv2.imwrite(str(out / f'{name}.jpg'), cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scan', required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--object', help='an object scan (scan_object.py, recon_object.py) to check instead')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scan, arm = Path(args.scan), Path(args.arm)
    twin = Twin(scan, arm)
    if args.object:
        check_object(twin, Path(args.object), out)
        return

    poses = json.loads((scan / 'colmap/poses_base_from_cam.json').read_text())
    names = sorted(poses)[::HOLD_OUT_EVERY]
    jaw_mask = cv2.imread(str(scan / 'jaw_mask.png'), cv2.IMREAD_GRAYSCALE) > 128
    whole, jaws, scene, rows = [], [], [], []
    for name in names:
        real = cv2.imread(str(scan / name))
        rendered = twin.wrist(np.array(poses[name]), JAW_REFERENCE_OPENING)
        whole.append(psnr(real, rendered))
        jaws.append(psnr(real, rendered, jaw_mask))
        scene.append(psnr(real, rendered, ~jaw_mask))
        rows.append(np.hstack([real, rendered]))
    print(f'wrist, {len(names)} held-out views: whole image {np.median(whole):.2f} dB, jaws {np.median(jaws):.2f} dB, '
          f'scene {np.median(scene):.2f} dB', flush=True)
    picked = [rows[i] for i in np.linspace(0, len(rows) - 1, 5).astype(int)]
    cv2.imwrite(str(out / 'wrist.jpg'), np.vstack(picked), [cv2.IMWRITE_JPEG_QUALITY, 85])

    links = np.load(arm / 'links.npz')
    gain = torch.load(arm / 'arm_splats.pt', weights_only=False)['gain'][::-1]
    r0, r1, c0, c1 = TABLE_ROI
    light = np.median(twin.background[r0:r1, c0:c1].reshape(-1, 3), 0) / np.median(median_background(REFERENCE_TOP)[r0:r1, c0:c1].reshape(-1, 3), 0)
    held = np.arange(0, len(links['names']), HOLD_OUT_EVERY)
    scores, rows = [], []
    for index in held:
        real = np.clip(cv2.imread(str(arm / f"top_{links['names'][index]}.png")) * gain * light, 0, 255).astype(np.uint8)
        rendered = twin.top(links['poses'][index])
        scores.append(psnr(real, rendered))
        rows.append(np.hstack([real, rendered]))
    print(f'top, {len(held)} held-out stops, whole image: {np.median(scores):.2f} dB', flush=True)
    picked = [rows[i] for i in np.linspace(0, len(rows) - 1, 4).astype(int)]
    cv2.imwrite(str(out / 'top.jpg'), cv2.resize(np.vstack(picked), None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    main()
