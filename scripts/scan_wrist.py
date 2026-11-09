"""Scan the empty workspace for the digital twin, with the arm moved by hand (torque off, nothing is commanded).

`wrist` saves a wrist image with the joint readings whenever the arm is still and the view is new. The window shows how
much of the camera poses used in the demonstrations is covered, and the nearest pose still missing. Exposure and white
balance are locked for the session, because automatic changes between views break the reconstruction; both go back to
automatic when the script ends.

`top` saves a low-noise top camera image for each arm pose. Images with the arm on either side are later combined into
a background without the arm.

    source scripts/so101_env.fish
    python scripts/scan_wrist.py wrist --wrist $WRIST --port $FOLLOWER --out runs/scan/table
    python scripts/scan_wrist.py top --top $TOP --port $FOLLOWER --out runs/scan/table

Keep the table empty (no key, no box, no leader arm) and the gripper at one fixed opening for the whole session, so
that the jaws occupy the same pixels in every image.
"""

import argparse
import glob
import json
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

import calib_top_extrinsics as top
from calib_intrinsics import PREVIEW_COMMAND

WIDTH, HEIGHT, FPS = 640, 480, 30
EXTRINSICS = 'runs/calib/top_extrinsics_2/extrinsics.json'
HANDEYE = 'runs/calib/wrist/handeye.json'
DATASET = 'artifacts/datasets/so101_pilot_v3'
N_TARGETS = 6000
COVER_M, COVER_DEG = 0.04, 15.0
NOVEL_M, NOVEL_DEG = 0.015, 6.0
GRIPPER_TOLERANCE_TICKS = 15
LOCK_COUNTDOWN_S = 8.0
TARGET_TABLE_BRIGHTNESS = 92  # bare table in the wrist images: median 86 in the demonstrations, 98 in the v5 tests
LOCKED_TEMPERATURE = 4400  # the black mat is neutral here in daylight (red/blue 1.0); colour varies with view, not settings
BRIGHTNESS_TOLERANCE = 0.2  # log ratio
METERING_CLOSE_ENOUGH = 0.08  # log ratio
METERING_ROUNDS = 3
INITIAL_EXPOSURE = 150
MIN_TABLE_PIXELS = 3000
MAP_SIZE, MAP_SCALE, MAP_Y_MAX = 480, 750.0, 0.32  # pixels, pixels per metre, half width in metres


class CameraPose:
    """Wrist camera pose in the base frame from joint readings (OpenCV axes: z is the optical axis)."""

    def __init__(self):
        extrinsics = json.loads(Path(EXTRINSICS).read_text())
        self.offsets = np.array([extrinsics['joint_offset_deg'][j] for j in top.JOINTS])
        self.mount = np.array(json.loads(Path(HANDEYE).read_text())['T_gripper_cam'])
        self.kinematics = top.Kinematics(extrinsics['mjcf'])
        self.calibration = top.load_calibration()

    def from_mjcf_deg(self, q_deg):
        k = self.kinematics
        k.forward(q_deg)
        gripper = np.eye(4)
        gripper[:3, :3], gripper[:3, 3] = k.data.xmat[k.gripper].reshape(3, 3), k.data.xpos[k.gripper]
        return gripper @ self.mount

    def from_raw(self, raw):
        return self.from_mjcf_deg(top.raw_to_lerobot_deg(dict(zip(top.ALL_MOTORS, raw)), self.calibration) + self.offsets)


def demonstrated_states(dataset=DATASET):
    """All joint states of the demonstrations in LeRobot units: five arm joints in degrees, gripper in percent."""
    files = sorted(glob.glob(str(Path(dataset) / 'data/chunk-*/*.parquet')))
    frame = pq.read_table(files, columns=['observation.state', 'intervention']).to_pandas()
    frame = frame[[bool(np.all(x)) for x in frame['intervention']]]
    return np.stack(frame['observation.state'].values)


def demonstrated_poses(camera, dataset=DATASET):
    """A random subset of the camera poses reached in the demonstrations, as positions and optical axes."""
    states = demonstrated_states(dataset)[:, :5]
    chosen = np.random.default_rng(0).choice(len(states), min(N_TARGETS, len(states)), replace=False)
    poses = np.array([camera.from_mjcf_deg(q + camera.offsets) for q in states[chosen]])
    return poses[:, :3, 3], poses[:, :3, 2]


def within(positions, axes, position, axis, radius_m, angle_deg):
    return (np.linalg.norm(positions - position, axis=1) < radius_m) & (axes @ axis > np.cos(np.radians(angle_deg)))


def open_wrist(path):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {path}')
    return cap


class Preview:
    """The mpv window: camera image, coverage map and a few lines of status."""

    def __init__(self):
        self.process = subprocess.Popen(PREVIEW_COMMAND, stdin=subprocess.PIPE)

    def show(self, frame, map_image, lines):
        if self.process is None:
            return
        shown = np.hstack([frame, map_image])
        for i, line in enumerate(lines):
            cv2.putText(shown, line, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(shown, line, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
        try:
            self.process.stdin.write(cv2.imencode('.jpg', shown, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            self.process.stdin.flush()
        except BrokenPipeError:
            self.process = None

    def close(self):
        if self.process:
            self.process.stdin.close()


def set_controls(device, **controls):
    arg = ','.join(f'{name}={value}' for name, value in controls.items())
    subprocess.run(['v4l2-ctl', '-d', device, '-c', arg], check=True)


def settled_frame(cap, frames=10):
    frame = None
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            raise SystemExit('wrist camera read failed (USB disconnect?); run again to continue')
    return frame.astype(float)


def average_frames(cap, count=8):
    """Mean of several new frames with the camera still: about a third of the sensor noise of a single frame."""
    return np.mean([settled_frame(cap, 1) for _ in range(count)], axis=0)


def brightness_and_balance(frame):
    usable = frame[:HEIGHT * 2 // 3]  # the jaws fill the bottom of the image
    b, g, r = usable.mean((0, 1))
    return (b + g + r) / 3, r / b


def table_brightness(frame):
    """Median brightness of the bare table: no wood, jaws, clipped highlights or near-black pixels, or None if too little shows."""
    upper = frame[:HEIGHT * 2 // 3]
    table = (upper[..., 2] < 1.25 * upper[..., 0]) & (upper.max(2) < 235) & (upper.min(2) > 8)
    return float(np.median(upper[table].mean(1))) if table.sum() >= MIN_TABLE_PIXELS else None


def lock_camera(device, cap, views):
    """Fix white balance and set the exposure so that the table is as bright as in the demonstrations.

    views() is an iterable that puts the camera at one metering view per step: the table varies by a factor of two
    between views, so a single view is a poor guide. Returns the settings, or None if the target cannot be reached.
    """
    exposure = INITIAL_EXPOSURE
    set_controls(device, auto_exposure=1, white_balance_automatic=0)
    set_controls(device, exposure_time_absolute=exposure, white_balance_temperature=LOCKED_TEMPERATURE)
    settled_frame(cap, 15)
    brightness = None
    for attempt in range(METERING_ROUNDS):
        readings = [table_brightness(average_frames(cap)) for _ in views()]
        readings = [r for r in readings if r]
        if not readings:
            print('metering: no table visible', flush=True)
            return None
        brightness = float(np.median(readings))
        if abs(np.log(brightness / TARGET_TABLE_BRIGHTNESS)) < METERING_CLOSE_ENOUGH or attempt == METERING_ROUNDS - 1:
            break
        exposure = int(np.clip(exposure * TARGET_TABLE_BRIGHTNESS / brightness, 10, 626))  # roughly proportional
        set_controls(device, exposure_time_absolute=exposure)
        settled_frame(cap)
    print(f'locked: exposure {exposure}, white balance {LOCKED_TEMPERATURE}; table brightness {brightness:.0f} '
          f'(target {TARGET_TABLE_BRIGHTNESS})', flush=True)
    if abs(np.log(brightness / TARGET_TABLE_BRIGHTNESS)) > BRIGHTNESS_TOLERANCE:
        print('  the table cannot be brought to the target; check the lighting', flush=True)
        return None
    return {'exposure_time_absolute': exposure, 'white_balance_temperature': LOCKED_TEMPERATURE}


def draw_map(positions, covered, current, goal):
    """Top view of the camera poses: the arm base is at the bottom centre, red still missing, green covered."""
    canvas = np.full((MAP_SIZE, MAP_SIZE, 3), 30, np.uint8)
    def pixel(p):
        return np.column_stack([(MAP_Y_MAX - p[..., 1]) * MAP_SCALE, MAP_SIZE - 12 - p[..., 0] * MAP_SCALE]).astype(int)

    for mask, colour in ((covered, (60, 130, 60)), (~covered, (50, 50, 220))):
        points = pixel(positions[mask])
        keep = (points >= 0).all(1) & (points < MAP_SIZE).all(1)
        canvas[points[keep, 1], points[keep, 0]] = colour
    base = tuple(int(v) for v in pixel(np.zeros((1, 3)))[0])
    cv2.drawMarker(canvas, base, (200, 200, 200), cv2.MARKER_CROSS, 16, 2)
    if goal is not None:
        cv2.circle(canvas, tuple(int(v) for v in pixel(goal[None])[0]), 11, (255, 220, 0), 2)
    cv2.circle(canvas, tuple(int(v) for v in pixel(current[None])[0]), 6, (0, 230, 255), -1)
    return canvas


def scan_wrist(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'wrist_views.json'
    data = json.loads(path.read_text()) if path.exists() else {'views': {}}

    camera = CameraPose()
    positions, axes = demonstrated_poses(camera, args.dataset)
    covered = np.zeros(len(positions), bool)
    saved_positions, saved_axes = [], []
    for raw in data['views'].values():
        pose = camera.from_raw([raw[m] for m in top.ALL_MOTORS])
        saved_positions.append(pose[:3, 3])
        saved_axes.append(pose[:3, 2])
        covered |= within(positions, axes, pose[:3, 3], pose[:3, 2], COVER_M, COVER_DEG)
    index = len(data['views'])
    gripper_ticks = data.get('gripper_ticks')

    cap = open_wrist(args.wrist)
    arm = top.Arm(args.port)
    preview = Preview()

    try:
        if 'locked' not in data:
            print(f'Point the camera at the table from working height and hold; locking in {LOCK_COUNTDOWN_S:.0f} s.', flush=True)
            start = time.time()
            while time.time() - start < LOCK_COUNTDOWN_S:
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit('wrist camera read failed (USB disconnect?); run again to continue')
                pose = camera.from_raw(arm.read_once())
                preview.show(frame, draw_map(positions, covered, pose[:3, 3], None),
                             ['look at the table and hold', f'locking exposure in {LOCK_COUNTDOWN_S - (time.time() - start):.0f} s'])
            data['locked'] = lock_camera(args.wrist, cap, lambda: [None])
            if data['locked'] is None:
                raise SystemExit('could not lock the exposure; point the camera at the bare table and run again')
            path.write_text(json.dumps(data, indent=1) + '\n')
        else:
            set_controls(args.wrist, auto_exposure=1, white_balance_automatic=0)
            set_controls(args.wrist, **data['locked'])
            settled_frame(cap, 15)
            print(f'resuming with {index} views, locked {data["locked"]}', flush=True)

        joints, still_since, last_status = deque(), None, 0.0
        print(f'Type q then Enter to stop. Goal: {args.goal:.0%} of the demonstrated camera poses.', flush=True)
        while covered.mean() < args.goal and index < args.max_views:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit('wrist camera read failed (USB disconnect?); run again to continue')
            raw, now = arm.read_once(), time.time()
            joints.append((now, raw))
            while now - joints[0][0] > 0.4:
                joints.popleft()
            recent = np.array([r[:5] for _, r in joints])
            still = (recent.max(0) - recent.min(0)).max() <= args.still_ticks and now - joints[0][0] > 0.2
            still_since = (still_since or now) if still else None

            pose = camera.from_raw(raw)
            position, axis = pose[:3, 3], pose[:3, 2]
            novel = not saved_positions or not within(np.array(saved_positions), np.array(saved_axes), position, axis,
                                                      NOVEL_M, NOVEL_DEG).any()
            if gripper_ticks is None:
                gripper_ok = True
            else:
                gripper_ok = abs(raw[5] - gripper_ticks) <= GRIPPER_TOLERANCE_TICKS

            goal, advice = None, 'all covered'
            if not covered.all():
                missing = np.where(~covered)[0]
                distance = np.linalg.norm(positions[missing] - position, axis=1) / COVER_M
                angle = np.degrees(np.arccos(np.clip(axes[missing] @ axis, -1, 1))) / COVER_DEG
                nearest = missing[np.argmin(distance + angle)]
                goal = positions[nearest]
                turn = np.degrees(np.arccos(np.clip(axes[nearest] @ axis, -1, 1)))
                advice = f'next: height {(goal[2] - position[2]) * 1000:+.0f} mm, tilt {turn:.0f} deg'

            if not gripper_ok:
                hint = 'gripper moved: put it back'
            elif not still:
                hint = 'moving'
            elif not novel:
                hint = 'already saved here, move on'
            else:
                hint = f'hold {now - still_since:.1f}/{args.hold_s:.1f} s'

            if still_since and novel and gripper_ok and now - still_since >= args.hold_s:
                frame_int = average_frames(cap).round().astype(np.uint8)
                name = f'view_{index:04d}.png'
                cv2.imwrite(str(out / name), frame_int)
                data['views'][name] = {m: float(v) for m, v in zip(top.ALL_MOTORS, raw)}
                if gripper_ticks is None:
                    gripper_ticks = float(raw[5])
                    data['gripper_ticks'] = gripper_ticks
                path.write_text(json.dumps(data, indent=1) + '\n')
                saved_positions.append(position)
                saved_axes.append(axis)
                covered |= within(positions, axes, position, axis, COVER_M, COVER_DEG)
                index += 1
                still_since = None
                print(f'[{index}] covered {covered.mean():.1%}', flush=True)
            elif now - last_status > 3.0:
                last_status = now
                print(f'  {hint}; {advice}', flush=True)

            if top.poll_line() == 'q':
                break
            preview.show(frame, draw_map(positions, covered, position, goal),
                         [f'{index} views, covered {covered.mean():.0%}', hint, advice])
    finally:
        arm.close()
        cap.release()
        preview.close()
        set_controls(args.wrist, auto_exposure=3, white_balance_automatic=1)
    print(f'{index} views, covered {covered.mean():.1%}. Exposure and white balance are back on automatic.')


def scan_top(args):
    out = Path(args.out) / 'top_backgrounds'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'poses.json'
    poses = json.loads(path.read_text()) if path.exists() else {}
    arm = top.Arm(args.port)
    cap = top.open_camera(args.top)
    try:
        print('Empty table. Move the arm (for example far left, then far right, then folded) and keep hands out of view.')
        while True:
            answer = input(f'{len(poses)} saved. Press Enter to save this pose, or q then Enter to finish: ').strip().lower()
            if answer == 'q':
                break
            top.grab(cap, 15)
            image = np.mean([top.grab(cap, 1) for _ in range(args.frames)], axis=0).round().astype(np.uint8)
            name = f'top_{len(poses):02d}.png'
            cv2.imwrite(str(out / name), image)
            poses[name] = arm.read()
            path.write_text(json.dumps(poses, indent=1) + '\n')
    finally:
        arm.close()
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    wrist = commands.add_parser('wrist')
    wrist.add_argument('--wrist', required=True)
    wrist.add_argument('--port', required=True)
    wrist.add_argument('--out', required=True)
    wrist.add_argument('--dataset', default=DATASET)
    wrist.add_argument('--goal', type=float, default=0.95)
    wrist.add_argument('--max-views', type=int, default=320)
    wrist.add_argument('--hold-s', type=float, default=0.25)
    wrist.add_argument('--still-ticks', type=int, default=3)
    wrist.set_defaults(run=scan_wrist)
    background = commands.add_parser('top')
    background.add_argument('--top', required=True)
    background.add_argument('--port', required=True)
    background.add_argument('--out', required=True)
    background.add_argument('--frames', type=int, default=30)
    background.set_defaults(run=scan_top)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
