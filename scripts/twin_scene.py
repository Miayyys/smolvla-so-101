"""MuJoCo scene of the digital twin: the arm with its identified servos, the table, the box and the key ring.

Everything here is in the base frame of the arm. The wrist-view reconstruction (table, background splats, scanned
objects) has its own frame, mapped by calib_refine_handeye.py (to_reconstruction); without that file the reconstruction
is taken as the base frame and the checkerboard mount is used.

The arm is so101_new_calib.xml with the servo parameters, command delay and joint limits of sysid_actuators.py
(runs/sysid/v4/actuators.json). The table is the plane fitted by recon_gs.py. The box is five thin plates with the
measured size (recon_box.py). The key ring is one rigid body made of columns standing on the table: over its scanned
height map (recon_object3d.py, heights.npz) when there is one, else thin plates over the opaque part of its flat
texture (recon_object.py); the columns are joined into convex pieces along the ring's length (object_pieces).

Only the wrist, the gripper and the two jaws touch the table and the objects; the rest of the arm touches nothing,
because the MJCF collision meshes already report contact with the table in the rest pose. Each jaw's collision mesh is
replaced by convex slices along its length: the convex hull of a whole jaw part fills the space the object is gripped in.

Object frames match twin_render.py: the box has its origin at the centre of its base, the key ring at the centre of
its opaque texture cells on the table surface, both with z up.

This file only needs mujoco, numpy and OpenCV, so it runs in both environments.
"""

import functools
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

ACTUATORS = 'runs/sysid/v4/actuators.json'
HANDEYE = 'runs/calib/wrist/handeye.json'
REFINED = 'runs/calib/wrist/refined.json'  # calib_refine_handeye.py: mount and the reconstruction's place
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
LINKS = ['base', 'shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper', 'moving_jaw_so101_v1']  # as recon_table.py
FRAME_S = 1 / 30
SUBSTEPS = 16
TOUCHING_BODIES = ['wrist', 'gripper', 'moving_jaw_so101_v1']
SLICED_MESHES = ['wrist_roll_follower_so101_v1', 'moving_jaw_so101_v1']
SLICE_M = 0.006
SURFACE_SAMPLES = 60000
ARM_BIT, OBJECT_BIT = 1, 2
JAW_FRICTION = 1.0
KEY_THICKNESS_M = 0.003
KEY_BLOCK_CELLS = 8  # texture cells per collision plate side (3.2 mm at 0.4 mm cells)
KEY_BLOCK_COVERAGE = 0.4
FOOTPRINT_COVERAGE = 0.35  # a height-map cell belongs to the object when this much of it is opaque in the flat texture
HEIGHT_STEP_M = 0.002  # neighbouring columns this close in height are merged into one
PIECE_M = 0.010  # length of each convex piece of the key ring along its long axis
# Not measured yet: weigh the box and the key ring and replace these.
BOX_MASS_KG = 0.030
KEY_MASS_KG = 0.030
SETTLE_S = 0.5
# Contacts of generated objects (they take precedence by priority): MuJoCo's default (0.02 s, 0.9-0.95) lets the jaws
# sink millimetres into an object of a few grams under the gripper's squeeze, and it slips out under the fixed jaw.
# These are the MJCF's finger pad values.
HARD_SOLREF = [0.005, 1.0]
HARD_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2]
GRIPPER_TORQUE_SHARE = 0.5  # LeRobot's so_follower sets the gripper's Max_Torque_Limit to 50 %
SOFT_DMIN = 0.3  # constraint impedance at first touch when the key is soft; 0.95 once pressed in by the softness width


def gripper_rad(percent, low, high):
    return low + np.asarray(percent) / 100 * (high - low)


def on_plane(plane, x, y, yaw):
    """Pose (4 x 4) of a frame lying on the plane z = a x + b y + c at (x, y), z along its normal, turned by yaw."""
    a, b, c = plane
    normal = np.array([-a, -b, 1.0]) / np.linalg.norm([a, b, 1.0])
    quat, rotation = np.zeros(4), np.zeros(9)
    mujoco.mju_quatZ2Vec(quat, normal)
    mujoco.mju_quat2Mat(rotation, quat)
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = rotation.reshape(3, 3) @ turn
    pose[:3, 3] = x, y, a * x + b * y + c
    return pose


def surface_points(model, data, geom, count, rng):
    """Points spread over the surface of a mesh geom, in world coordinates of the current pose."""
    mesh = model.geom_dataid[geom]
    vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
    faces = model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
    corners = vertices[faces]
    area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    chosen = corners[rng.choice(len(corners), count, p=area / area.sum())]
    w = rng.dirichlet(np.ones(3), count)[..., None]
    points = np.vstack([(chosen * w).sum(1), vertices])
    return (data.geom_xmat[geom].reshape(3, 3) @ points.T).T + data.geom_xpos[geom]


@functools.lru_cache
def jaw_slices(mjcf):
    """For each jaw mesh: its body and convex pieces (point sets in the body frame), cut across the jaw's length."""
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    rng = np.random.default_rng(0)
    gripper = model.body('gripper').id
    along = data.xmat[gripper].reshape(3, 3)[:, 2]  # the jaws point along -z of the gripper body
    pieces = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 3:
            continue
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[geom]) not in SLICED_MESHES:
            continue
        body = model.geom_bodyid[geom]
        world = surface_points(model, data, geom, SURFACE_SAMPLES, rng)
        local = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
        depth = world @ along
        for start in np.arange(depth.min(), depth.max(), SLICE_M):
            part = local[(depth >= start - 1e-4) & (depth <= start + SLICE_M + 1e-4)]
            if len(part) >= 20 and np.linalg.matrix_rank(part - part.mean(0), tol=1e-5) == 3:
                pieces.append((model.body(body).name, part))
    return pieces


def flat_plates(directory):
    """Plates covering the opaque texture cells of a flat object: (centre x, centre y, half x, half y) in its frame."""
    meta = json.loads((Path(directory) / 'object/object.json').read_text())
    alpha = cv2.imread(str(Path(directory) / 'object/texture.png'), cv2.IMREAD_UNCHANGED)[..., 3] > 127
    rows, columns = np.nonzero(alpha)
    centre = rows.mean(), columns.mean()
    n, cell = KEY_BLOCK_CELLS, meta['cell_m']
    trimmed = alpha[:alpha.shape[0] // n * n, :alpha.shape[1] // n * n]
    blocks = trimmed.reshape(trimmed.shape[0] // n, n, trimmed.shape[1] // n, n).mean((1, 3)) > KEY_BLOCK_COVERAGE
    plates = []
    for i, row in enumerate(blocks):  # texture rows run along x, columns along y; merge runs of blocks along y
        j = 0
        while j < len(row):
            if not row[j]:
                j += 1
                continue
            end = j
            while end < len(row) and row[end]:
                end += 1
            plates.append((((i + 0.5) * n - 0.5 - centre[0]) * cell, ((j + end) / 2 * n - 0.5 - centre[1]) * cell,
                           n * cell / 2, (end - j) * n * cell / 2))
            j = end
    return plates


def object_columns(directory):
    """Boxes standing on the table that make up an object: (centre x, centre y, half x, half y, height) in its frame.

    From the height map of recon_object3d.py: its cells over the flat outline or with a scanned height, each as tall as
    the upper quartile of the scanned heights around it (the map has holes where few splats were solid), runs of cells
    of similar height along y merged. Without a height map, plates of KEY_THICKNESS_M over the flat outline."""
    directory = Path(directory)
    if not (directory / 'object/heights.npz').exists():
        return [(x, y, hx, hy, KEY_THICKNESS_M) for x, y, hx, hy in flat_plates(directory)]
    meta = json.loads((directory / 'object/object.json').read_text())
    alpha = cv2.imread(str(directory / 'object/texture.png'), cv2.IMREAD_UNCHANGED)[..., 3] > 127
    rows, columns = np.nonzero(alpha)
    saved = np.load(directory / 'object/heights.npz')
    scanned, cell, half = saved['heights'], float(saved['cell']), saved['half']
    centres = [-half[k] + (np.arange(scanned.shape[k]) + 0.5) * cell for k in (0, 1)]
    member = scanned > 0
    for i, x in enumerate(centres[0]):
        r0, r1 = (int(round(rows.mean() + (x + s * cell / 2) / meta['cell_m'])) for s in (-1, 1))
        for j, y in enumerate(centres[1]):
            c0, c1 = (int(round(columns.mean() + (y + s * cell / 2) / meta['cell_m'])) for s in (-1, 1))
            window = alpha[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)]
            member[i, j] |= window.size > 0 and window.mean() > FOOTPRINT_COVERAGE
    heights = np.zeros_like(scanned)
    for i, j in zip(*np.nonzero(member)):
        for reach in (1, 2, 3):
            around = scanned[max(i - reach, 0):i + reach + 1, max(j - reach, 0):j + reach + 1]
            if (around > 0).sum() >= 3:
                heights[i, j] = np.percentile(around[around > 0], 75)
                break
        else:
            heights[i, j] = KEY_THICKNESS_M
    out = []
    for i, x in enumerate(centres[0]):
        j = 0
        while j < len(centres[1]):
            if heights[i, j] <= 0:
                j += 1
                continue
            end = j + 1
            while end < len(centres[1]) and heights[i, end] > 0 and \
                    abs(heights[i, end] - heights[i, j:end].mean()) < HEIGHT_STEP_M:
                end += 1
            out.append((x, (centres[1][j] + centres[1][end - 1]) / 2, cell / 2, (end - j) * cell / 2,
                        float(heights[i, j:end].max())))
            j = end
    return out


def object_pieces(directory):
    """Convex pieces (point sets in the object frame) of the object's columns, cut across its long axis every PIECE_M.

    The columns themselves are 3 mm wide: thinner than a jaw slice, so a squeezing jaw pushes them inside itself and
    then holds them from both sides. Pieces this size cannot get in, and their sloped sides let a jaw slide off."""
    columns = np.array(object_columns(directory))
    corners = np.array([[x + sx * hx, y + sy * hy, z] for x, y, hx, hy, height in columns
                        for z in (0, height) for sx in (-1, 1) for sy in (-1, 1)])
    centred = columns[:, :2] - columns[:, :2].mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    along = centred @ axis
    pieces = []
    for start in np.arange(along.min(), along.max() + 1e-9, PIECE_M):
        chosen = np.nonzero((along >= start - 1e-9) & (along < start + PIECE_M))[0]
        if len(chosen):
            pieces.append(corners.reshape(-1, 8, 3)[chosen].reshape(-1, 3))
    return pieces


class Scene:
    """The twin's physics. Objects are given as {'box': directory, 'key': directory} of their scans, or as
    twin_shapes.Shape (convex parts in the object frame) for generated ones."""

    def __init__(self, table, objects=None, actuators=ACTUATORS, table_drop_m=0.0, key_softness_m=0.0):
        """key_softness_m > 0 lets the jaws press that far into the key ring before it pushes back fully, as the real
        bundle gives under the jaws; its contacts with the table and the box stay hard."""
        fitted = json.loads(Path(actuators).read_text())
        self.params = fitted['params']
        self.offsets_deg = np.array([fitted['joint_offset_deg'][j] for j in JOINTS])
        if Path(REFINED).exists():
            refined = json.loads(Path(REFINED).read_text())
            self.mount = np.array(refined['T_gripper_cam'])
            self.recon_R, self.recon_t, self.recon_scale = np.array(refined['R']), np.array(refined['t']), refined['scale']
        else:
            self.mount = np.array(json.loads(Path(HANDEYE).read_text())['T_gripper_cam'])
            self.recon_R, self.recon_t, self.recon_scale = np.eye(3), np.zeros(3), 1.0
        a, b, c = np.load(Path(table) / 'gs/table.npz')['plane'].astype(float)
        grid = np.array([[x, y, a * x + b * y + c] for x in (0.1, 0.4) for y in (-0.3, 0.3)])
        base = (self.recon_R.T @ (self.recon_scale * grid - self.recon_t).T).T
        self.plane = np.linalg.lstsq(np.column_stack([base[:, :2], np.ones(4)]), base[:, 2], rcond=None)[0]
        self.plane[2] -= table_drop_m  # an extra drop of the physics table, if wanted
        spec = mujoco.MjSpec.from_file(fitted['mjcf'])
        spec.option.timestep = FRAME_S / SUBSTEPS
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        touching = {spec.body(name) for name in TOUCHING_BODIES}
        for geom in spec.geoms:
            hits = geom.parent in touching and geom.contype != 0 and \
                not (geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname in SLICED_MESHES) and \
                'finger_pad' not in geom.name
            geom.contype, geom.conaffinity = (ARM_BIT if hits else 0), 0
        for index, (body, points) in enumerate(jaw_slices(fitted['mjcf'])):
            spec.add_mesh(name=f'jaw_slice_{index}', uservert=points.ravel().tolist())
            spec.body(body).add_geom(name=f'jaw_slice_{index}', type=mujoco.mjtGeom.mjGEOM_MESH,
                                     meshname=f'jaw_slice_{index}', contype=ARM_BIT, conaffinity=0, group=3,
                                     friction=[JAW_FRICTION, 0.005, 0.0001], mass=0)

        a, b, c = self.plane
        normal = np.array([-a, -b, 1.0]) / np.linalg.norm([a, b, 1.0])
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, normal)
        spec.worldbody.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01], pos=[0, 0, c],
                                quat=quat, contype=0, conaffinity=ARM_BIT | OBJECT_BIT, rgba=[0.1, 0.1, 0.1, 1])
        self.table_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(self.table_rotation, quat)
        self.table_rotation = self.table_rotation.reshape(3, 3)

        self.object_names = []
        for name, directory in (objects or {}).items():
            body = spec.worldbody.add_body(name=name)
            body.add_freejoint(name=f'{name}_free')
            common = dict(contype=OBJECT_BIT, conaffinity=ARM_BIT | OBJECT_BIT, group=3)
            if hasattr(directory, 'parts'):  # a twin_shapes.Shape
                shape = directory
                pieces = shape.pieces()
                volumes = [max(np.ptp(p, 0).prod(), 1e-9) for p in pieces]
                for k, (points, volume) in enumerate(zip(pieces, volumes)):
                    spec.add_mesh(name=f'{name}_{k}', uservert=np.asarray(points).ravel().tolist())
                    body.add_geom(name=f'{name}_{k}', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'{name}_{k}',
                                  mass=shape.mass * volume / sum(volumes), friction=[shape.friction, 0.005, 0.0001],
                                  rgba=[*shape.colour, 1], priority=1, solref=HARD_SOLREF, solimp=HARD_SOLIMP, **common)
            elif (Path(directory) / 'object/box.json').exists():
                meta = json.loads((Path(directory) / 'object/box.json').read_text())
                h, t, z = meta['size_m'] / 2, meta['wall_m'] / 2, meta['height_m'] / 2
                plates = [((0, 0, t), (h, h, t)), ((h - t, 0, z), (t, h, z)), ((-h + t, 0, z), (t, h, z)),
                          ((0, h - t, z), (h, t, z)), ((0, -h + t, z), (h, t, z))]
                volume = sum(8 * np.prod(size) for _, size in plates)
                for k, (pos, size) in enumerate(plates):
                    body.add_geom(name=f'{name}_{k}', type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                                  density=BOX_MASS_KG / volume, rgba=[0.9, 0.9, 0.9, 1], **common)
            else:
                pieces = object_pieces(directory)
                for k, points in enumerate(pieces):
                    spec.add_mesh(name=f'{name}_{k}', uservert=points.ravel().tolist())
                    body.add_geom(name=f'{name}_{k}', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'{name}_{k}',
                                  mass=KEY_MASS_KG / len(pieces), rgba=[0.7, 0.7, 0.6, 1], **common)
            self.object_names.append(name)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        limits = np.array([fitted['joint_limits_rad'][j] for j in JOINTS])
        m.jnt_range[:5] = limits
        m.actuator_ctrlrange[:5] = limits
        self.jaw_low, self.jaw_high = m.jnt_range[5]
        p = self.params
        m.actuator_gainprm[:6, 0] = p['kp']
        m.actuator_biasprm[:6, 1] = -p['kp']
        m.actuator_biasprm[:6, 2] = -p['kv']
        m.actuator_forcerange[:6] = [-p['forcerange'], p['forcerange']]
        m.actuator_forcerange[5] *= GRIPPER_TORQUE_SHARE
        m.dof_damping[:6], m.dof_frictionloss[:6], m.dof_armature[:6] = p['damping'], p['frictionloss'], p['armature']
        self.link_ids = [m.body(name).id for name in LINKS]
        self.object_ids = [m.body(name).id for name in self.object_names]
        self.object_qpos = [m.jnt_qposadr[m.joint(f'{name}_free').id] for name in self.object_names]
        if key_softness_m > 0 and 'key' in self.object_names:
            # with differing priorities a contact takes the parameters of the higher one: jaw-key soft, the rest hard
            soft, hard = m.body('key').id, {0} | {m.body(n).id for n in self.object_names if n != 'key'}
            for geom in range(m.ngeom):
                if m.geom_bodyid[geom] == soft:
                    m.geom_priority[geom] = 1
                    m.geom_solimp[geom] = [SOFT_DMIN, 0.95, key_softness_m, 0.5, 2]
                elif m.geom_bodyid[geom] in hard:
                    m.geom_priority[geom] = 2

    # Units: datasets hold LeRobot degrees for the five arm joints and percent for the gripper.
    def to_rad(self, values):
        values = np.asarray(values, float)
        return np.concatenate([np.radians(values[..., :5] + self.offsets_deg),
                               gripper_rad(values[..., 5:], self.jaw_low, self.jaw_high)], -1)

    def from_rad(self, q):
        """Joint angles (rad, the model's) in dataset units: the inverse of to_rad."""
        q = np.asarray(q, float)
        return np.concatenate([np.degrees(q[..., :5]) - self.offsets_deg,
                               (q[..., 5:] - self.jaw_low) / (self.jaw_high - self.jaw_low) * 100], -1)

    def opening_percent(self):
        return float((self.data.qpos[5] - self.jaw_low) / (self.jaw_high - self.jaw_low) * 100)

    def on_table(self, x, y, yaw):
        """Pose (4 x 4) of an object frame lying on the table at (x, y), turned by yaw about the table normal."""
        return on_plane(self.plane, x, y, yaw)

    def reset(self, joints, placements=None):
        """Arm at the joint values (dataset units), objects at {'name': (x, y, yaw)}; lets the objects settle."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        q = self.to_rad(joints)
        d.qpos[:6] = q
        for name, start in zip(self.object_names, self.object_qpos):
            pose = self.on_table(*(placements or {}).get(name, (0.25, 0.0, 0.0)))
            d.qpos[start:start + 3] = pose[:3, 3] + pose[:3, 2] * 0.0005
            mujoco.mju_mat2Quat(d.qpos[start + 3:start + 7], pose[:3, :3].ravel())
        d.ctrl[:] = q
        mujoco.mj_forward(m, d)
        for _ in range(int(SETTLE_S / m.opt.timestep)):
            d.qpos[:6], d.qvel[:6] = q, 0  # the arm is held while the objects settle
            mujoco.mj_step(m, d)
        d.qpos[:6], d.qvel[:6] = q, 0
        mujoco.mj_forward(m, d)
        self.command = q.copy()
        self.history = [q.copy()]

    def step(self, action):
        """One 30 Hz frame with the action (dataset units) as the servo target, applied after the identified delay."""
        m, d = self.model, self.data
        self.history.append(self.to_rad(action))
        previous, current = self.history[-2], self.history[-1]
        for sub in range(SUBSTEPS):
            # target as a function of time is the linear interpolation of the frame targets, shifted by the delay
            phase = (sub / SUBSTEPS) - self.params['delay_s'] / FRAME_S
            if phase >= 0:
                d.ctrl[:6] = previous + phase * (current - previous)
            else:
                older = self.history[-3] if len(self.history) > 2 else previous
                d.ctrl[:6] = older + (1 + phase) * (previous - older)
            mujoco.mj_step(m, d)

    def set_arm(self, joints):
        """Places the arm at the joint values without dynamics (for rendering recorded states)."""
        self.data.qpos[:6] = self.to_rad(joints)
        mujoco.mj_kinematics(self.model, self.data)

    def pose(self, body):
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = self.data.xmat[body].reshape(3, 3), self.data.xpos[body]
        return out

    def link_poses(self):
        return np.stack([self.pose(body) for body in self.link_ids])

    def wrist_camera(self):
        return self.pose(self.link_ids[5]) @ self.mount

    def to_reconstruction(self, pose):
        """A pose (4 x 4) in the base frame expressed in the wrist-view reconstruction's frame, for twin_render.py."""
        out = np.eye(4)
        out[:3, :3] = self.recon_R @ pose[:3, :3]
        out[:3, 3] = (self.recon_R @ pose[:3, 3] + self.recon_t) / self.recon_scale
        return out

    def object_poses(self):
        return {name: self.pose(body) for name, body in zip(self.object_names, self.object_ids)}
