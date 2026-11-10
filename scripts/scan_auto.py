"""Scan the empty workspace with the wrist camera while the script moves the arm (this one commands the motors).

`plan` needs no hardware. It picks joint poses from the demonstrations that cover the camera poses used there, and joins
them with straight joint-space moves. A move is accepted only if every point on it stays within JOINT_NEAR_DEG of a
demonstrated pose and, with the gravity sag predicted by the identified servo model, clears the table in MuJoCo. Only
table contact is checked: the MJCF collision meshes report self-contact on most poses the real arm reached without any.
The arm leaves and re-enters its folded rest pose along a recorded demonstration, which touches nothing on the real arm
although the model reports contact there.

`run` drives the plan at low speed, stops at each pose, saves an image with the measured joints, and ends at the rest
pose with torque off. Images go to the same files as scan_wrist.py, so hand-guided views can be added afterwards.

`arm` stops at poses spread in joint space and sets a different jaw opening at each, saving top and wrist images for
the appearance model of the arm. Jaws move only while the arm is stopped, at poses clear of the table at every opening.

`top` visits poses chosen so that together they reveal the whole table to the top camera, for a background without the
arm, and saves a low-noise top image at each.

    source scripts/so101_env.fish
    python scripts/scan_auto.py plan --out runs/scan/table_auto
    python scripts/scan_auto.py run --wrist $WRIST --port $FOLLOWER --out runs/scan/table_auto --limit 5 --speed 12   # rehearsal
    python scripts/scan_auto.py run --wrist $WRIST --port $FOLLOWER --out runs/scan/table_auto
    python scripts/scan_auto.py top --top $TOP --port $FOLLOWER --out runs/scan/table_auto

Stopping: Ctrl-C, or a joint falling behind its command (collision, stall), makes the arm hold where it is, and torque
is released after you take hold of the arm and press Enter. q then Enter finishes the current move and returns to rest.
"""

import argparse
import glob
import heapq
import json
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq

import calib_top_extrinsics as top
import scan_wrist as scan

ACTUATORS = 'runs/sysid/v4/actuators.json'
GRIPPER_OPEN_PCT = 40.0  # the opening used in the demonstrations; the jaws stay there for the whole scan
TABLE_CLEARANCE_M = 0.016  # collision plane above z = 0, the base plane; a tablet on the table measured 8 mm
JOINT_NEAR_DEG = 20.0
FOLDED_LIFT_DEG = -75.0  # with shoulder_lift below this the arm is folded over its base: the jaws open only beyond it
HUB_LIFT_DEG = -55.0
PATH_STEP_DEG = 2.0
NEIGHBOURS = 14
N_CANDIDATES = 2500
CONTROL_HZ = 30
BUS_RETRIES = 2
TRACKING_LIMIT_DEG = 9.0
TRACKING_LIMIT_TICKS = 5
START_TOLERANCE_DEG = 10.0
EXIT_FRAME_STRIDE = 4
METERING_VIEWS = 8
LIGHT_WINDOW_VIEWS = 12  # the table varies between views, so light changes are judged on a running median
LIGHT_CHANGE_RATIO = 1.3
SETTLE_DEG, SETTLE_S, SETTLE_TIMEOUT_S = 0.4, 0.4, 4.0
TOP_WIDTH, TOP_HEIGHT = 1280, 720
MASK_MARGIN_PX = 30  # around the projected arm: calibration error and the nearest part of its shadow
WANTED_FREE_VIEWS = 4
JAW_OPENINGS_PCT = [float(v) for v in np.linspace(2, 60, 12)]  # the demonstrations stay below about 40%
GRIPPER_MOVE_S = 0.8


class Checker:
    """Validity of arm poses given in LeRobot units (five joints in degrees)."""

    def __init__(self, states, boxes=()):
        """boxes: extra obstacles on the table as (centre, half sizes) in metres, base frame."""
        extrinsics = json.loads(Path(scan.EXTRINSICS).read_text())
        self.offsets = np.array([extrinsics['joint_offset_deg'][j] for j in top.JOINTS])
        self.kp = json.loads(Path(ACTUATORS).read_text())['params']['kp']
        spec = mujoco.MjSpec.from_file(extrinsics['mjcf'])
        table = spec.worldbody.add_geom()
        table.type = mujoco.mjtGeom.mjGEOM_PLANE
        table.size = [2, 2, 0.01]
        table.pos = [0, 0, TABLE_CLEARANCE_M]
        table.contype = table.conaffinity = 1
        for index, (centre, half) in enumerate(boxes):
            box = spec.worldbody.add_geom()
            box.name = f'obstacle_{index}'
            box.type = mujoco.mjtGeom.mjGEOM_BOX
            box.size, box.pos = list(half), list(centre)
            box.contype = box.conaffinity = 1
        self.model = spec.compile()
        self.boxes = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'obstacle_{i}') for i in range(len(boxes))}
        self.data = mujoco.MjData(self.model)
        low, high = self.model.jnt_range[5]
        self.gripper = low + GRIPPER_OPEN_PCT / 100 * (high - low)
        self.states = states[:, :4]

    def contacts(self, q, boxes_only=False):
        """Number of table (or only obstacle box) contacts at the pose the arm reaches when commanded to q, sag included."""
        m, d = self.model, self.data
        d.qpos[:5] = np.radians(q[:5] + self.offsets)
        d.qpos[5] = self.gripper
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        d.qpos[:5] -= d.qfrc_bias[:5] / self.kp
        mujoco.mj_forward(m, d)
        if boxes_only:
            return sum(c.geom1 in self.boxes or c.geom2 in self.boxes for c in d.contact[:d.ncon])
        return sum(m.geom_bodyid[c.geom1] == 0 or m.geom_bodyid[c.geom2] == 0 for c in d.contact[:d.ncon])

    def near_demonstrations(self, q):
        return np.abs(self.states - q[:4]).max(1).min() <= JOINT_NEAR_DEG

    def pose_ok(self, q):
        return q[1] >= FOLDED_LIFT_DEG and self.near_demonstrations(q) and self.contacts(q) == 0

    def move_ok(self, a, b):
        steps = int(np.ceil(np.abs(b[:5] - a[:5]).max() / PATH_STEP_DEG)) + 1
        return all(self.pose_ok(a + (b - a) * s) for s in np.linspace(0, 1, steps + 1))


def route_to(nodes, adjacency, start, goals):
    """Shortest joint-space route through the accepted moves from start to the nearest of goals, or None."""
    best, previous, heap = {start: 0.0}, {}, [(0.0, start)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost > best[node]:
            continue
        if node in goals:
            path = [node]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return path[::-1][1:]
        for other in adjacency[node]:
            total = cost + np.abs(nodes[other] - nodes[node]).max()
            if total < best.get(other, np.inf):
                best[other], previous[other] = total, node
                heapq.heappush(heap, (total, other))
    return None


def tour(nodes, adjacency, todo):
    """Node sequence starting and ending at node 0 (rest) that visits every reachable node of todo."""
    route, current, todo = [0], 0, set(todo) - {0}
    while todo:
        path = route_to(nodes, adjacency, current, todo)
        if path is None:
            break
        route += path
        current = path[-1]
        todo -= set(path)
    return route + (route_to(nodes, adjacency, current, {0}) or []) if current != 0 else route, todo


def duration(a, b, speed):
    """Time for a cosine-profile move whose fastest joint peaks at `speed` degrees per second."""
    return max(0.3, np.pi / 2 * np.abs(b[:5] - a[:5]).max() / speed)


def plan(args):
    states = scan.demonstrated_states(args.dataset)
    checker = Checker(states)
    camera = scan.CameraPose()
    rng = np.random.default_rng(0)

    leaving = exit_path(args.dataset, checker)
    hub = leaving[-1].copy()
    hub[5] = GRIPPER_OPEN_PCT
    chosen = rng.choice(len(states), N_CANDIDATES, replace=False)
    candidates = states[chosen].copy()
    candidates[:, 5] = GRIPPER_OPEN_PCT
    candidates = candidates[[checker.pose_ok(q) for q in candidates]]
    print(f'{len(candidates)} of {N_CANDIDATES} demonstrated poses are valid here', flush=True)

    positions, axes = scan.demonstrated_poses(camera, args.dataset)
    poses = np.array([camera.from_mjcf_deg(q[:5] + camera.offsets) for q in candidates])
    cover = np.array([scan.within(positions, axes, p[:3, 3], p[:3, 2], scan.COVER_M, scan.COVER_DEG) for p in poses])
    covered, picked = np.zeros(len(positions), bool), []
    while covered.mean() < args.goal and len(picked) < args.max_views:
        gains = (cover & ~covered).sum(1)
        if gains.max() < 2:
            break
        picked.append(int(gains.argmax()))
        covered |= cover[picked[-1]]
    nodes = np.vstack([hub, candidates[picked]])

    adjacency = [[] for _ in nodes]
    for i, q in enumerate(nodes):
        order = np.argsort(np.abs(nodes[:, :5] - q[:5]).max(1))[1:NEIGHBOURS + 1]
        for j in order:
            if j not in adjacency[i] and checker.move_ok(q, nodes[j]):
                adjacency[i].append(int(j))
                adjacency[j].append(i)
    route, unreachable = tour(nodes, adjacency, range(len(nodes)))
    reached = sorted(set(route))
    reachable_cover = np.zeros(len(positions), bool)
    for node in reached:
        if node:
            reachable_cover |= cover[picked[node - 1]]
    seconds = sum(duration(nodes[a], nodes[b], args.speed) for a, b in zip(route, route[1:])) + 1.2 * len(route)
    print(f'{len(nodes) - 1} views picked, {len(reached) - 1} reachable; covers {reachable_cover.mean():.1%} of the '
          f'demonstrated camera poses; {len(route) - 1} moves, about {seconds / 60:.1f} min at {args.speed:.0f} deg/s')
    if unreachable:
        print(f'  {len(unreachable)} picked views have no safe route and are skipped')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan_data = {'nodes': nodes.tolist(), 'adjacency': adjacency, 'exit_path': leaving.tolist()}
    (out / 'auto_plan.json').write_text(json.dumps(plan_data, indent=0) + '\n')


def exit_path(dataset, checker):
    """A recorded way out of the rest pose: from the most typical start to the first frame that passes the checks."""
    files = sorted(glob.glob(str(Path(dataset) / 'data/chunk-*/*.parquet')))
    frame = pq.read_table(files, columns=['observation.state', 'episode_index', 'intervention']).to_pandas()
    frame = frame[[bool(np.all(x)) for x in frame['intervention']]]
    episodes = [np.stack(group['observation.state'].values) for _, group in frame.groupby('episode_index')]
    starts = np.array([e[0] for e in episodes])
    median = np.median(starts, 0)
    episode = episodes[int(np.abs(starts[:, :4] - median[:4]).max(1).argmin())]
    last = next(i for i, q in enumerate(episode) if q[1] >= HUB_LIFT_DEG and checker.pose_ok(q))
    return np.vstack([episode[:last + 1][::EXIT_FRAME_STRIDE], episode[last]])


class Abort(Exception):
    pass


class LiveCamera:
    """Reads the camera in a background thread, so the window stays live and the control loop never waits for a frame.

    read() has the interface of cv2.VideoCapture.read and returns each new frame once.
    """

    def __init__(self, cap, preview, map_image, lines):
        self.cap, self.preview, self.map_image, self.lines = cap, preview, map_image, lines
        self.frame, self.count, self.seen, self.failed, self.stopping = None, 0, 0, False, False
        self.changed = threading.Condition()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while not self.stopping:
            ok, frame = self.cap.read()
            with self.changed:
                self.failed = not ok
                if ok:
                    self.frame, self.count = frame, self.count + 1
                self.changed.notify_all()
            if not ok:
                return
            if self.count % 2 == 0:
                self.preview.show(frame, self.map_image, self.lines)

    def status(self, map_image, lines):
        self.map_image, self.lines = map_image, lines

    def read(self):
        with self.changed:
            if not self.changed.wait_for(lambda: self.failed or self.count > self.seen, timeout=2.0) or self.failed:
                return False, None
            self.seen = self.count
            return True, self.frame.copy()

    def release(self):
        self.stopping = True
        self.thread.join(timeout=2.0)
        self.cap.release()


class Arm:
    """The follower with torque on, configured as for recording. Goals and positions are in LeRobot units."""

    def __init__(self, port):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id='follower', use_degrees=True))
        self.bus = self.robot.bus
        self.bus.connect()
        if not self.bus.is_calibrated:
            self.bus.disconnect(disable_torque=False)
            raise SystemExit('the motors do not match the calibration file; nothing was changed')
        self.torque = False

    def position(self):
        values = self.bus.sync_read('Present_Position', num_retry=BUS_RETRIES)
        return np.array([values[m] for m in top.ALL_MOTORS])

    def raw(self):
        values = self.bus.sync_read('Present_Position', normalize=False, num_retry=BUS_RETRIES)
        return {m: float(values[m]) for m in top.ALL_MOTORS}

    def enable(self):
        """Torque on with the goal already at the present position, so nothing jumps."""
        self.bus.disable_torque()
        self.goal(self.position())
        self.robot.configure()
        self.torque = True

    def goal(self, q):
        self.bus.sync_write('Goal_Position', {m: float(v) for m, v in zip(top.ALL_MOTORS, q)}, num_retry=BUS_RETRIES)

    def hold(self):
        self.goal(self.position())

    def release(self):
        self.bus.disable_torque()
        self.torque = False

    def close(self):
        self.bus.disconnect(disable_torque=False)


def move(arm, a, b, speed):
    follow(arm, np.array([a, b]), speed)


def follow(arm, points, speed):
    """Go through joint-space points with one cosine speed profile; raise Abort if a joint stays too far behind."""
    lengths = np.abs(np.diff(points[:, :5], axis=0)).max(1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    total, behind, start = max(0.3, np.pi / 2 * arc[-1] / speed), 0, time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        fraction = min(1.0, elapsed / total)
        position = arc[-1] * 0.5 * (1 - np.cos(np.pi * fraction))
        target = np.array([np.interp(position, arc, points[:, j]) for j in range(6)])
        arm.goal(target)
        error = np.abs(arm.position()[:5] - target[:5]).max()
        behind = behind + 1 if error > TRACKING_LIMIT_DEG else 0
        if behind >= TRACKING_LIMIT_TICKS:
            raise Abort(f'a joint is {error:.1f} deg behind its command')
        if fraction >= 1.0:
            return
        time.sleep(max(0.0, 1 / CONTROL_HZ - (time.perf_counter() - start - elapsed)))


def settle(arm):
    """Wait until the measured joints stop changing."""
    window, start = [], time.time()
    while time.time() - start < SETTLE_TIMEOUT_S:
        window.append((time.time(), arm.position()[:5]))
        window = [w for w in window if w[0] >= window[-1][0] - SETTLE_S]
        values = np.array([w[1] for w in window])
        if window[-1][0] - window[0][0] >= SETTLE_S * 0.9 and (values.max(0) - values.min(0)).max() <= SETTLE_DEG:
            return True
        time.sleep(0.02)
    return False


def drive(port, speed, planned, route, visit, intro, prepare=None):
    """Leave the rest pose, follow the route calling visit(arm, node) at each stop, return to rest and release.

    prepare(arm, goto), if given, runs first and may move the arm with goto(node); if it returns False the route is
    skipped and the arm goes home. visit returns True to end early. On Ctrl-C, a tracking fault or any error the arm holds its position until you have
    taken hold of it.
    """
    nodes, adjacency, leaving = np.array(planned['nodes']), planned['adjacency'], np.array(planned['exit_path'])
    arm = Arm(port)
    try:
        present = arm.position()
        if np.abs(present[:4] - leaving[0, :4]).max() > START_TOLERANCE_DEG:
            raise SystemExit(f'put the arm in the folded rest pose first: joints {present[:4].round(0)}, '
                             f'rest {leaving[0, :4].round(0)}')
        print(intro)
        print('The arm will move. Table empty, hands clear, ready to cut the power.')
        if input('Type GO then Enter: ').strip() != 'GO':
            raise SystemExit('not started; the motors were not touched')
        arm.enable()
        follow(arm, np.vstack([present, leaving]), speed)
        move(arm, leaving[-1], nodes[0], speed)
        current = 0

        def goto(node):
            nonlocal current
            for step in route_to(nodes, adjacency, current, {node}) or [] if current != node else []:
                move(arm, nodes[current], nodes[step], speed)
                current = step

        proceed = prepare is None or prepare(arm, goto)
        goto(0)
        for node in route[1:] if proceed else []:
            move(arm, nodes[current], nodes[node], speed)
            current = node
            if visit(arm, node):
                break
        for node in route_to(nodes, adjacency, current, {0}) or [] if current != 0 else []:
            move(arm, nodes[current], nodes[node], speed)
            current = node
        move(arm, nodes[0], leaving[-1], speed)
        follow(arm, leaving[::-1], speed)
        settle(arm)
        arm.release()
        print('Back at rest, torque off.')
    except (Abort, KeyboardInterrupt) as stop:
        if arm.torque:
            arm.hold()
            print(f'\nSTOPPED: {str(stop) or "Ctrl-C"}. The arm is holding its position.', flush=True)
            input('Take hold of the arm, then press Enter to release the motors: ')
            arm.release()
    finally:
        if arm.torque:
            print('\nError while moving. The arm is still powered. Support it, then cut the power or press Enter.', flush=True)
            try:
                arm.hold()
                input('Press Enter to release the motors: ')
                arm.release()
            except Exception as error:
                print(f'could not release the motors ({error}); support the arm and cut the power')
        arm.close()


def run(args):
    out = Path(args.out)
    planned = json.loads((out / 'auto_plan.json').read_text())
    nodes, adjacency = np.array(planned['nodes']), planned['adjacency']
    path = out / 'wrist_views.json'
    data = json.loads(path.read_text()) if path.exists() else {'views': {}}
    done = set(data.get('auto_done', []))
    camera = scan.CameraPose()
    positions, axes = scan.demonstrated_poses(camera, args.dataset)
    covered = np.zeros(len(positions), bool)
    for raw in data['views'].values():
        pose = camera.from_raw([raw[m] for m in top.ALL_MOTORS])
        covered |= scan.within(positions, axes, pose[:3, 3], pose[:3, 2], scan.COVER_M, scan.COVER_DEG)
    todo = [i for i in range(1, len(nodes)) if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    wanted = set(todo)
    route, _ = tour(nodes, adjacency, todo)
    index, stops = len(data['views']), 0
    recent = []

    preview = scan.Preview()
    cap = LiveCamera(scan.open_wrist(args.wrist), preview, scan.draw_map(positions, covered, np.zeros(3), None), ['starting'])

    def light_changed():
        """True if the last views are much brighter or darker than the first ones of the session; those views are dropped."""
        nonlocal index, recent
        if len(recent) < LIGHT_WINDOW_VIEWS:
            return False
        recent = recent[-LIGHT_WINDOW_VIEWS:]
        level = float(np.median([b for _, _, b in recent]))
        reference = data.setdefault('table_reference', level)
        if max(level / reference, reference / level) < LIGHT_CHANGE_RATIO:
            return False
        for name, node, _ in recent:
            del data['views'][name]
            done.discard(node)
        index -= len(recent)
        data['auto_done'] = sorted(done)
        path.write_text(json.dumps(data, indent=1) + '\n')
        print(f'STOPPING: the light changed (table brightness {level:.0f}, was {reference:.0f}). '
              f'The last {len(recent)} views were dropped; run again when the light is back.', flush=True)
        return True

    def visit(arm, node):
        nonlocal index, stops
        if node not in wanted or node in done:
            return False
        if not settle(arm):
            print(f'  node {node}: the joints did not settle, saving anyway', flush=True)
        scan.settled_frame(cap, 4)
        image = scan.average_frames(cap).round().astype(np.uint8)
        raw = arm.raw()
        name = f'view_{index:04d}.png'
        cv2.imwrite(str(out / name), image)
        data['views'][name] = raw
        done.add(node)
        data['auto_done'] = sorted(done)
        data.setdefault('gripper_ticks', raw['gripper'])
        path.write_text(json.dumps(data, indent=1) + '\n')
        pose = camera.from_raw([raw[m] for m in top.ALL_MOTORS])
        covered[:] |= scan.within(positions, axes, pose[:3, 3], pose[:3, 2], scan.COVER_M, scan.COVER_DEG)
        index += 1
        stops += 1
        print(f'[{stops}/{len(todo)}] {name}, covered {covered.mean():.1%}', flush=True)
        cap.status(scan.draw_map(positions, covered, pose[:3, 3], None), [f'{stops}/{len(todo)}', f'covered {covered.mean():.0%}'])
        brightness = scan.table_brightness(image.astype(float))
        if brightness:
            recent.append((name, node, brightness))
        return light_changed() or top.poll_line() == 'q'

    def prepare(arm, goto):
        if 'locked' in data:
            scan.set_controls(args.wrist, auto_exposure=1, white_balance_automatic=0)
            scan.set_controls(args.wrist, **data['locked'])
            scan.settled_frame(cap, 15)
            return True

        def views():
            for node in todo[:METERING_VIEWS]:
                goto(node)
                settle(arm)
                scan.settled_frame(cap, 4)
                yield node

        cap.status(scan.draw_map(positions, covered, np.zeros(3), None), ['metering exposure'])
        locked = scan.lock_camera(args.wrist, cap, views)
        if locked is None:
            return False
        data['locked'] = locked
        path.write_text(json.dumps(data, indent=1) + '\n')
        return True

    intro = f'{len(todo)} poses to visit at {args.speed:.0f} deg/s peak, after a short exposure metering pass.'
    try:
        drive(args.port, args.speed, planned, route, visit, intro, prepare)
        print(f'{index} views in {out}.')
    finally:
        cap.release()
        preview.close()
        scan.set_controls(args.wrist, auto_exposure=3, white_balance_automatic=1)


def arm_masks(nodes):
    """For each pose, a quarter-resolution mask of the top camera pixels hidden by the arm, its margin included."""
    extrinsics = json.loads(Path(scan.EXTRINSICS).read_text())
    intrinsics = json.loads(Path(extrinsics['intrinsics']).read_text())
    K, dist = np.array(intrinsics['K']), np.array(intrinsics['dist'])
    rvec, t = cv2.Rodrigues(np.array(extrinsics['R_cam_base']))[0], np.array(extrinsics['t_cam_base_m'])
    offsets = np.array([extrinsics['joint_offset_deg'][j] for j in top.JOINTS])
    kinematics = top.Kinematics(extrinsics['mjcf'])
    model, state = kinematics.model, kinematics.data
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_MARGIN_PX // 4 + 1,) * 2)
    masks = np.zeros((len(nodes), TOP_HEIGHT // 4, TOP_WIDTH // 4), bool)
    for index, q in enumerate(nodes):
        kinematics.forward(q[:5] + offsets)
        mask = np.zeros((TOP_HEIGHT // 4, TOP_WIDTH // 4), np.uint8)
        for geom in range(model.ngeom):
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 2:
                continue
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            world = (state.geom_xmat[geom].reshape(3, 3) @ model.mesh_vert[start:start + count].T).T + state.geom_xpos[geom]
            points = cv2.projectPoints(world, rvec, t, K, dist)[0].reshape(-1, 2) / 4
            cv2.fillConvexPoly(mask, cv2.convexHull(points.astype(np.int32)), 1)
        masks[index] = cv2.dilate(mask, kernel) > 0
    return masks


def pick_background_poses(nodes, count):
    """Poses that leave every table pixel visible in as many images as possible, up to WANTED_FREE_VIEWS."""
    free = ~arm_masks(nodes)
    seen, picked = np.zeros(free.shape[1:], int), []
    for _ in range(count):
        gains = (np.minimum(seen + free, WANTED_FREE_VIEWS) - np.minimum(seen, WANTED_FREE_VIEWS)).sum((1, 2)).astype(float)
        gains[[0] + picked] = -1
        picked.append(int(gains.argmax()))
        seen += free[picked[-1]]
    print(f'{count} poses: {(seen >= 1).mean():.1%} of the image visible at least once, '
          f'{(seen >= WANTED_FREE_VIEWS).mean():.1%} at least {WANTED_FREE_VIEWS} times', flush=True)
    return picked


def top_backgrounds(args):
    planned = json.loads((Path(args.out) / 'auto_plan.json').read_text())
    nodes, adjacency = np.array(planned['nodes']), planned['adjacency']
    out = Path(args.out) / 'top_backgrounds'
    path = out / 'poses.json'
    if path.exists():
        raise SystemExit(f'{path} exists; move it away to capture again')
    out.mkdir(parents=True, exist_ok=True)
    picked = pick_background_poses(nodes, args.poses)
    wanted, poses = set(picked), {}
    route, _ = tour(nodes, adjacency, picked)
    cap = top.open_camera(args.top)

    def visit(arm, node):
        if node not in wanted:
            return False
        wanted.discard(node)
        settle(arm)
        top.grab(cap, 15)
        image = np.mean([top.grab(cap, 1) for _ in range(args.frames)], axis=0).round().astype(np.uint8)
        name = f'top_{len(poses):02d}.png'
        cv2.imwrite(str(out / name), image)
        poses[name] = arm.raw()
        path.write_text(json.dumps(poses, indent=1) + '\n')
        print(f'[{len(poses)}/{len(picked)}] {name}', flush=True)
        return False

    try:
        drive(args.port, args.speed, planned, route, visit, f'{len(picked)} top camera backgrounds. Stay out of the top camera view.')
    finally:
        cap.release()


def spread_stops(nodes, count, checker):
    """Poses that stay clear of the table at every jaw opening, spread out in joint space (farthest-point sampling)."""
    usable = []
    low, high = checker.model.jnt_range[5]
    for node in range(1, len(nodes)):
        clear = True
        for opening in (min(JAW_OPENINGS_PCT), max(JAW_OPENINGS_PCT)):
            checker.gripper = low + opening / 100 * (high - low)
            clear = clear and checker.pose_ok(nodes[node])
        if clear:
            usable.append(node)
    joints = nodes[usable, :5]
    chosen = [int(np.abs(joints - np.median(joints, 0)).max(1).argmin())]
    distance = np.abs(joints - joints[chosen[0]]).max(1)
    while len(chosen) < min(count, len(usable)):
        chosen.append(int(distance.argmax()))
        distance = np.minimum(distance, np.abs(joints - joints[chosen[-1]]).max(1))
    return [usable[i] for i in chosen]


def move_gripper(arm, q, start_pct, end_pct):
    """Open or close the jaws with the arm held at q."""
    steps = max(2, int(GRIPPER_MOVE_S * CONTROL_HZ))
    for fraction in np.linspace(0, 1, steps):
        target = q.copy()
        target[5] = start_pct + (end_pct - start_pct) * fraction
        arm.goal(target)
        arm.position()
        time.sleep(1 / CONTROL_HZ)


def arm_views(args):
    """Top and wrist images of the arm at spread poses and jaw openings, for the arm's appearance model."""
    source = Path(args.source)
    planned = json.loads((source / 'auto_plan.json').read_text())
    locked = json.loads((source / 'wrist_views.json').read_text())['locked']
    nodes, adjacency = np.array(planned['nodes']), planned['adjacency']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'stops.json'
    stops = json.loads(path.read_text()) if path.exists() else {}
    checker = Checker(scan.demonstrated_states(args.dataset))
    chosen = spread_stops(nodes, args.stops, checker)
    opening_of = {node: JAW_OPENINGS_PCT[i % len(JAW_OPENINGS_PCT)] for i, node in enumerate(chosen)}
    todo = [node for node in chosen if str(node) not in stops]
    wanted = set(todo)
    route, _ = tour(nodes, adjacency, todo)
    print(f'{len(chosen)} stops, {len(todo)} still to do, jaw openings {min(JAW_OPENINGS_PCT):.0f}-{max(JAW_OPENINGS_PCT):.0f}%', flush=True)

    preview = scan.Preview()
    blank = np.full((scan.MAP_SIZE, scan.MAP_SIZE, 3), 30, np.uint8)
    wrist = LiveCamera(scan.open_wrist(args.wrist), preview, blank, ['starting'])
    top_camera = top.open_camera(args.top)

    def prepare(arm, goto):
        scan.set_controls(args.wrist, auto_exposure=1, white_balance_automatic=0)
        scan.set_controls(args.wrist, **locked)
        scan.settled_frame(wrist, 15)
        top.grab(top_camera, 60)  # the top camera needs a couple of seconds to adjust after opening
        return True

    def visit(arm, node):
        if node not in wanted:
            return False
        wanted.discard(node)
        opening = opening_of[node]
        travel = float(nodes[node, 5])
        move_gripper(arm, nodes[node], travel, opening)
        settle(arm)
        scan.settled_frame(wrist, 6)
        wrist_image = scan.average_frames(wrist).round().astype(np.uint8)
        top.grab(top_camera, 8)
        top_image = np.mean([top.grab(top_camera, 1) for _ in range(args.frames)], axis=0).round().astype(np.uint8)
        raw = arm.raw()
        move_gripper(arm, nodes[node], opening, travel)
        name = f'{len(stops):03d}'
        cv2.imwrite(str(out / f'wrist_{name}.png'), wrist_image)
        cv2.imwrite(str(out / f'top_{name}.png'), top_image)
        stops[str(node)] = {'name': name, 'opening_pct': opening, 'raw': raw}
        path.write_text(json.dumps(stops, indent=1) + '\n')
        print(f'[{len(stops)}/{len(chosen)}] jaw {opening:.0f}%', flush=True)
        wrist.status(blank, [f'{len(stops)}/{len(chosen)}', f'jaw {opening:.0f}%'])
        return top.poll_line() == 'q'

    try:
        drive(args.port, args.speed, planned, route, visit, f'{len(todo)} poses for the arm model. Stay out of both camera views.', prepare)
        print(f'{len(stops)} stops in {out}.')
    finally:
        wrist.release()
        top_camera.release()
        preview.close()
        scan.set_controls(args.wrist, auto_exposure=3, white_balance_automatic=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    planner = commands.add_parser('plan')
    planner.add_argument('--out', required=True)
    planner.add_argument('--dataset', default=scan.DATASET)
    planner.add_argument('--goal', type=float, default=0.95)
    planner.add_argument('--max-views', type=int, default=260)
    planner.add_argument('--speed', type=float, default=25.0)
    planner.set_defaults(run=plan)
    runner = commands.add_parser('run')
    runner.add_argument('--wrist', required=True)
    runner.add_argument('--port', required=True)
    runner.add_argument('--out', required=True)
    runner.add_argument('--dataset', default=scan.DATASET)
    runner.add_argument('--speed', type=float, default=25.0, help='peak joint speed, degrees per second')
    runner.add_argument('--limit', type=int, default=0, help='stop after this many poses and return to rest')
    runner.set_defaults(run=run)
    background = commands.add_parser('top', help='top camera images with the arm in poses that together reveal the table')
    background.add_argument('--top', required=True)
    background.add_argument('--port', required=True)
    background.add_argument('--out', required=True)
    background.add_argument('--poses', type=int, default=8)
    background.add_argument('--frames', type=int, default=30)
    background.add_argument('--speed', type=float, default=25.0)
    background.set_defaults(run=top_backgrounds)
    arm = commands.add_parser('arm', help='top and wrist images of the arm at spread poses and jaw openings')
    arm.add_argument('--wrist', required=True)
    arm.add_argument('--top', required=True)
    arm.add_argument('--port', required=True)
    arm.add_argument('--source', required=True, help='the scan directory with auto_plan.json and the locked exposure')
    arm.add_argument('--out', required=True)
    arm.add_argument('--dataset', default=scan.DATASET)
    arm.add_argument('--stops', type=int, default=80)
    arm.add_argument('--frames', type=int, default=20)
    arm.add_argument('--speed', type=float, default=25.0)
    arm.set_defaults(run=arm_views)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
