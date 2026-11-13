"""Side-by-side comparison of recorded camera images with the digital twin's renders, frame by frame. Runs in .venv-gs.

Used by twin_replay.py (without --no-render). Each frame the twin's wrist and top images are rendered from the
simulated arm, jaw and objects (twin_render.py) and written next to the recorded ones into a video; per episode the
median image scores are returned.

The physics, the arm and the top camera are in the base frame; the wrist view is rendered in the reconstruction's
frame (twin_scene.Scene.to_reconstruction, from calib_refine_handeye.py), objects included.

Scores: PSNR after matching each image's colour gain (the recording's light is not the scan's), the top one only over
the mat (the operator's hand and the leader arm can show beside it). Object overlap in the top view: pixels that differ
from the empty-table photograph against the drawn box and key, both outside the arm; the key alone while it lies
apart from the box.
"""

from pathlib import Path

import av
import cv2
import numpy as np

import twin_render
import twin_replay

DIFF_THRESHOLD = twin_replay.DIFF_THRESHOLD
ARM_MARGIN_PX = 15
KEY_NEAR_PX = 40
BOX_MARGIN_PX = 8
SOLID = 0.5
MIN_KEY_PIXELS = 50
TILE = (640, 360), (640, 480)  # top, wrist


def psnr_matched(real, rendered, mask=None):
    """PSNR after scaling the render's channels to the real image's medians (over the mask)."""
    real, rendered = real.astype(np.float32), rendered.astype(np.float32)
    pick = (lambda a: a[mask]) if mask is not None else (lambda a: a.reshape(-1, 3))
    gain = np.median(pick(real), 0) / np.maximum(np.median(pick(rendered), 0), 1)
    difference = (np.clip(rendered * gain, 0, 255) - real) ** 2
    return float(-10 * np.log10(pick(difference).mean() / 255 ** 2))


def overlap(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float('nan')


class Comparison:
    def __init__(self, scan, box, key):
        self.twin = twin_render.Twin(scan, Path(scan) / 'arm')
        self.box, _ = self.twin.load_object(box)
        self.key, _ = self.twin.load_object(key)
        self.background = self.twin.background.astype(np.float32)
        camera, plane = twin_replay.TopCamera(), self.twin.layer[0].cpu().numpy()
        x, y = twin_replay.TABLE_X_M, twin_replay.TABLE_Y_M
        corners = [[px, py, plane[0] * px + plane[1] * py + plane[2]] for px, py in ((x[0], y[0]), (x[1], y[0]), (x[1], y[1]), (x[0], y[1]))]
        self.mat = np.zeros(self.background.shape[:2], np.uint8)
        cv2.fillConvexPoly(self.mat, camera.project(corners).astype(np.int32), 1)
        self.mat = self.mat > 0

    def start(self, path, placements):
        self.container = av.open(str(path), 'w')
        self.stream = self.container.add_stream('libx264', rate=twin_replay.FPS)
        self.stream.width, self.stream.height = TILE[0][0] * 2, TILE[0][1] + TILE[1][1]
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': '26'}
        self.gains, self.scores, self.index = None, {k: [] for k in ('wrist', 'top', 'objects', 'key')}, 0

    def render(self, scene, poses):
        order = [self.box, self.key]
        wrist = self.twin.wrist(scene.to_reconstruction(scene.wrist_camera()), scene.opening_percent(),
                                [(n, scene.to_reconstruction(poses[n])) for n in order])
        top, arm, masks = self.twin.top(scene.link_poses(), [(n, poses[n]) for n in order], self.gains, masks=True)
        return wrist, top, arm, masks

    def frame(self, scene, poses, top_image, wrist_image):
        wrist, top, arm, masks = self.render(scene, poses)
        arm_region = cv2.dilate((arm > 0.2).astype(np.uint8), np.ones((2 * ARM_MARGIN_PX + 1,) * 2, np.uint8)) > 0
        if self.gains is None:
            # the objects' colours were captured by the wrist camera: scale them to the top camera on the first frame
            self.gains = {}
            for name, mask in masks.items():
                visible = (mask > 0.9) & ~arm_region
                if visible.sum() > 20:
                    self.gains[name] = (np.median(top_image[visible], 0) / np.maximum(np.median(top[visible], 0), 1))[::-1].copy()
            wrist, top, arm, masks = self.render(scene, poses)

        self.scores['wrist'].append(psnr_matched(wrist_image, wrist))
        self.scores['top'].append(psnr_matched(top_image, top, self.mat))
        real = top_image.astype(np.float32)
        gain = np.median(real[self.mat] / np.maximum(self.background[self.mat], 1), 0)
        changed = (np.abs(real - self.background * gain).max(2) > DIFF_THRESHOLD) & self.mat & ~arm_region
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        box, key = (masks[self.box] > SOLID) & ~arm_region, (masks[self.key] > SOLID) & ~arm_region
        self.scores['objects'].append(overlap(changed, box | key))
        near_box = cv2.dilate(box.astype(np.uint8), np.ones((2 * BOX_MARGIN_PX + 1,) * 2, np.uint8)) > 0
        if key.sum() > MIN_KEY_PIXELS and not (key & near_box).any():
            near_key = cv2.dilate(key.astype(np.uint8), np.ones((2 * KEY_NEAR_PX + 1,) * 2, np.uint8)) > 0
            self.scores['key'].append(overlap(changed & near_key & ~near_box, key))

        tiles = [cv2.resize(image, size) for image, size in
                 ((top_image, TILE[0]), (top, TILE[0]), (wrist_image, TILE[1]), (wrist, TILE[1]))]
        for tile, label in zip(tiles, ('recorded', 'twin', 'recorded', 'twin')):
            cv2.putText(tile, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(tiles[1], f'frame {self.index}  objects IoU {self.scores["objects"][-1]:.2f}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(grid, format='bgr24')):
            self.container.mux(packet)
        self.index += 1

    def finish(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        median = lambda values: float(np.nanmedian(values)) if len(values) and not np.all(np.isnan(values)) else None
        return {'psnr_wrist_db': median(self.scores['wrist']), 'psnr_top_mat_db': median(self.scores['top']),
                'top_objects_iou': median(self.scores['objects']), 'top_key_iou': median(self.scores['key']),
                'top_key_iou_frames': len(self.scores['key']),
                'top_gains': {k: np.round(v, 3).tolist() for k, v in (self.gains or {}).items()}}
