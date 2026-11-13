"""Refine the wrist camera's hand-eye mount and the reconstruction's place in the base frame from a workspace scan.

    .venv/bin/python scripts/calib_refine_handeye.py --scan runs/scan/table_curtain

recon_table.py brings the COLMAP model into the base frame by matching its camera centres to the joint-based ones.
That fit cannot tell a wrong mount from a wrong model: with the checkerboard mount the model came out 3 % too small,
which put the table 12 mm above where the jaws touch it, and the mount was 3 deg off. Here both are fitted to all views
at once, under the model

    C_i (reconstruction) = to_reconstruction(G_i (gripper, from the joints) @ X (mount)),
    to_reconstruction(p) = (R p + t) / scale,

with Huber weights. Half the views are fitted and the other half scored, then all are used. Written:
runs/calib/wrist/refined.json with T_gripper_cam (the new mount), and R, t, scale of to_reconstruction. twin_scene.py
and twin_render.py use them: the physics and the arm live in the base frame, the reconstructed scene in its own.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import recon_table

HANDEYE = 'runs/calib/wrist/handeye.json'
OUT = 'runs/calib/wrist/refined.json'
METRES_PER_RADIAN = 0.1  # weight of rotation against position in the residual
HUBER_M = 0.005
ITERATIONS = 30


def rigid(p):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = cv2.Rodrigues(np.asarray(p[:3], float))[0], p[3:6]
    return out


def residuals(p, views, gripper, mount):
    """Per view: rotation (scaled) and position of the reconstruction's camera against the predicted one."""
    world, x, scale = rigid(p[:6]), rigid(p[6:12]) @ mount, 1 + p[12]
    out = []
    for camera, grip in zip(views, gripper):
        predicted = world @ grip @ x
        predicted[:3, 3] /= scale
        difference = np.linalg.inv(camera) @ predicted
        out.append(np.concatenate([cv2.Rodrigues(difference[:3, :3])[0].ravel() * METRES_PER_RADIAN, difference[:3, 3]]))
    return np.concatenate(out)


def fit(views, gripper, mount):
    """Gauss-Newton with a numerical Jacobian, Huber weights."""
    p = np.zeros(13)
    for _ in range(ITERATIONS):
        r = residuals(p, views, gripper, mount)
        jacobian = np.empty((len(r), len(p)))
        for k in range(len(p)):
            step = np.zeros_like(p)
            step[k] = 1e-6
            jacobian[:, k] = (residuals(p + step, views, gripper, mount) - r) / 1e-6
        weight = np.sqrt(np.where(np.abs(r) < HUBER_M, 1.0, HUBER_M / np.abs(r)))
        delta = np.linalg.lstsq(jacobian * weight[:, None], -r * weight, rcond=None)[0]
        p += delta
        if np.abs(delta).max() < 1e-10:
            break
    return p


def scores(p, views, gripper, mount):
    r = residuals(p, views, gripper, mount).reshape(-1, 6)
    position = np.linalg.norm(r[:, 3:], axis=1) * 1000
    angle = np.degrees(np.linalg.norm(r[:, :3], axis=1) / METRES_PER_RADIAN)
    return f'position mean {position.mean():.1f} mm, p95 {np.percentile(position, 95):.1f} mm; rotation mean {angle.mean():.2f} deg'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scan', required=True)
    args = parser.parse_args()

    scan = Path(args.scan)
    cameras = {k: np.array(v) for k, v in json.loads((scan / 'colmap/poses_base_from_cam.json').read_text()).items()}
    data = json.loads((scan / 'wrist_views.json').read_text())
    names = sorted(cameras)
    joint_based = recon_table.joint_poses(names, data)
    mount = np.array(json.loads(Path(HANDEYE).read_text())['T_gripper_cam'])
    views = [cameras[n] for n in names]
    gripper = [joint_based[n] @ np.linalg.inv(mount) for n in names]

    print(f'{len(names)} views; checkerboard mount, reconstruction as aligned: {scores(np.zeros(13), views, gripper, mount)}')
    half = fit(views[::2], gripper[::2], mount)
    print(f'fitted on even views, scored on odd: {scores(half, views[1::2], gripper[1::2], mount)} '
          f'(before: {scores(np.zeros(13), views[1::2], gripper[1::2], mount)}); scale +{half[12] * 100:.2f} %')
    p = fit(views, gripper, mount)
    print(f'fitted on all: {scores(p, views, gripper, mount)}')

    world, new_mount = rigid(p[:6]), rigid(p[6:12]) @ mount
    scale = 1 + p[12]
    a, b, c = np.load(scan / 'gs/table.npz')['plane']
    corners = np.array([[x, y, a * x + b * y + c] for x in (0.15, 0.35) for y in (-0.2, 0.2)])
    in_base = (world[:3, :3].T @ (scale * corners - world[:3, 3]).T).T
    print(f'scale +{p[12] * 100:.2f} %, reconstruction turned {np.degrees(np.linalg.norm(p[:3])):.2f} deg, '
          f'mount turned {np.degrees(np.linalg.norm(p[6:9])):.2f} deg and moved '
          f'{np.linalg.norm(new_mount[:3, 3] - mount[:3, 3]) * 1000:.1f} mm')
    print(f'table corners in the base frame: {np.round(corners[:, 2] * 1000, 1)} mm as reconstructed, '
          f'{np.round(in_base[:, 2] * 1000, 1)} mm refined')
    Path(OUT).write_text(json.dumps({'T_gripper_cam': new_mount.tolist(), 'R': world[:3, :3].tolist(),
                                     't': world[:3, 3].tolist(), 'scale': float(scale), 'scan': str(scan),
                                     'note': 'reconstruction point = (R @ base point + t) / scale'}, indent=1) + '\n')
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
