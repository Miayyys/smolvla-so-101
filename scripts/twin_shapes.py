"""Random irregular, roughly elongated rigid objects for the twin: what the scripted expert learns to grasp.

    .venv/bin/python scripts/twin_shapes.py --count 6        # prints a few shapes and their grasp widths

Each shape is a few convex parts, each an irregular cuboid: eight corners with their own jitter, so that the width and
the height change along the part and the top is not flat. Families:

    bar     one long part
    key     a squat head and a long, thinner blade (a key, a lighter, a USB stick)
    bent    two long parts at an angle
    hammer  a long handle and a crosswise head
    dumbbell  a handle between two heads

The object frame has its origin on the table under the middle of the footprint, z up, the long side roughly along x.
The bottom corners stay at z = 0 so the object lies flat.
"""

import argparse
from dataclasses import dataclass, field

import numpy as np

FAMILIES = ['bar', 'key', 'bent', 'hammer', 'dumbbell']
MAX_LENGTH_M = 0.11  # longest footprint side: the box inside is 8.6 cm across, its diagonal 12 cm
DENSITY_KG_M3 = (300, 1500)
MASS_KG = (0.008, 0.080)
FRICTION = (0.5, 1.0)
STABLE_HEIGHT = 0.9
PIECE_M = 0.012


@dataclass
class Shape:
    family: str
    parts: list  # convex point sets (N x 3) in the object frame
    mass: float
    friction: float
    colour: tuple = (0.6, 0.6, 0.6)
    meta: dict = field(default_factory=dict)

    def pieces(self):
        """Collision pieces: each part cut every PIECE_M along its length. One long convex mesh gets only one or two
        contacts with the table in MuJoCo and rocks over onto an edge; short pieces each get their own."""
        out = []
        for corners in self.parts:
            near, far = corners[:4], corners[4:]  # the two ends, in the order irregular_cuboid makes them
            count = max(1, int(np.ceil(np.linalg.norm(far.mean(0) - near.mean(0)) / PIECE_M)))
            for k in range(count):
                a, b = k / count, (k + 1) / count
                out.append(np.vstack([near + a * (far - near), near + b * (far - near)]))
        return out

    def height(self):
        return max(float(p[:, 2].max()) for p in self.parts)

    def centre_of_mass(self):
        volumes = np.array([hull_volume(p) for p in self.parts])
        centres = np.array([p.mean(0) for p in self.parts])
        return (centres * volumes[:, None]).sum(0) / volumes.sum()


def hull_volume(points):
    """Volume of an irregular cuboid's corners: its bounding prism scaled by the share of the footprint it fills."""
    footprint = polygon_area(convex_hull_2d(points[:, :2]))
    return footprint * float(points[:, 2].max()) * 0.8


def convex_hull_2d(points):
    """Counter-clockwise hull (monotone chain)."""
    points = np.unique(np.round(np.asarray(points, float), 9), axis=0)
    if len(points) < 3:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def half(sequence):
        out = []
        for p in sequence:
            while len(out) >= 2 and cross(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(points), half(points[::-1])
    return np.array(lower[:-1] + upper[:-1])


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def polygon_area(polygon):
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def irregular_cuboid(rng, centre, yaw, length, width, height, jitter=0.15, taper=0.0):
    """Eight corners of a cuboid along its own x; each corner moved by up to jitter of the size, the width narrowed
    towards +x by taper (share), the top corners given their own heights."""
    corners = []
    for sx in (-1, 1):
        half_width = width / 2 * (1 - taper * (sx + 1) / 2)
        for sy in (-1, 1):
            x = sx * length / 2 + rng.uniform(-jitter, jitter) * min(length, 3 * width) / 3
            y = sy * half_width * (1 + rng.uniform(-jitter, jitter))
            corners.append((x, y, 0.0))
            # the top corner over it, leaning a little: sides that slope much make a wedge the jaws squeeze out
            corners.append((x + rng.uniform(-0.1, 0.1) * jitter * width, y * (1 + rng.uniform(-0.3, 0.3) * jitter),
                            height * (1 + rng.uniform(-jitter, jitter))))
    corners = np.array(corners)
    c, s = np.cos(yaw), np.sin(yaw)
    corners[:, :2] = corners[:, :2] @ np.array([[c, s], [-s, c]]) + centre
    return corners


def random_shape(rng, family=None):
    family = family or rng.choice(FAMILIES)
    u = rng.uniform
    if family == 'bar':
        length, width = u(0.05, 0.11), u(0.008, 0.030)
        parts = [irregular_cuboid(rng, (0, 0), 0, length, width, u(0.005, 0.025), taper=u(0, 0.5))]
    elif family == 'key':
        head, blade = u(0.015, 0.035), u(0.03, 0.07)
        head_height = u(0.006, 0.025)
        parts = [irregular_cuboid(rng, (-blade / 2, 0), u(-0.3, 0.3), head * u(0.8, 1.2), head, head_height, 0.2),
                 irregular_cuboid(rng, (head / 2 - 0.003, u(-0.004, 0.004)), u(-0.25, 0.25), blade + 0.006,
                                  u(0.007, min(0.020, head * 0.8)), head_height * u(0.3, 0.9), taper=u(0, 0.5))]
    elif family == 'bent':
        a, b, width, height, angle = u(0.03, 0.06), u(0.03, 0.06), u(0.008, 0.025), u(0.006, 0.020), u(0.3, 1.0)
        parts = [irregular_cuboid(rng, (-a / 2, 0), 0, a + width / 2, width, height),
                 irregular_cuboid(rng, (np.cos(angle) * b / 2, np.sin(angle) * b / 2), angle, b + width / 2, width,
                                  height * u(0.8, 1.2))]
    elif family == 'hammer':
        handle, head = u(0.05, 0.09), u(0.025, 0.05)
        thick = u(0.010, 0.022)
        parts = [irregular_cuboid(rng, (0, 0), 0, handle, u(0.008, 0.020), u(0.006, 0.018), taper=u(0, 0.4)),
                 irregular_cuboid(rng, (-handle / 2 - thick / 2 + 0.002, 0), np.pi / 2 + u(-0.2, 0.2), head, thick,
                                  u(0.010, 0.025))]
    elif family == 'dumbbell':
        handle, head = u(0.03, 0.06), u(0.018, 0.035)
        height = u(0.008, 0.025)
        parts = [irregular_cuboid(rng, (0, 0), 0, handle + 0.006, u(0.007, 0.016), height * u(0.4, 0.8))]
        for side in (-1, 1):
            parts.append(irregular_cuboid(rng, (side * (handle + head) / 2, 0), u(-0.3, 0.3), head * u(0.6, 1.0),
                                          head, height, 0.2))
    else:
        raise ValueError(family)

    # no part taller than STABLE_HEIGHT of its narrow side, or it falls over onto that side when put down
    for p in parts:
        hull = convex_hull_2d(p[:, :2])
        narrow = min(np.ptp(hull @ np.array([np.cos(a), np.sin(a)])) for a in np.linspace(0, np.pi, 36, endpoint=False))
        p[:, 2] *= min(1.0, STABLE_HEIGHT * narrow / p[:, 2].max())

    # centre the footprint, turn its long side onto x
    flat = np.vstack(parts)[:, :2]
    middle = (flat.max(0) + flat.min(0)) / 2
    axis = np.linalg.svd(flat - flat.mean(0), full_matrices=False)[2][0]
    turn = np.array([[axis[0], -axis[1]], [axis[1], axis[0]]])  # columns: new x, new y in the old frame
    parts = [np.column_stack([(p[:, :2] - middle) @ turn, p[:, 2]]) for p in parts]
    flat = np.vstack(parts)[:, :2]
    if np.ptp(flat, 0).max() > MAX_LENGTH_M:
        scale = MAX_LENGTH_M / np.ptp(flat, 0).max()
        parts = [np.column_stack([p[:, :2] * scale, p[:, 2]]) for p in parts]
    volume = sum(hull_volume(p) for p in parts)
    mass = float(np.clip(volume * u(*DENSITY_KG_M3), *MASS_KG))
    colour = tuple(float(c) for c in rng.uniform(0.15, 0.95, 3))
    return Shape(family, parts, mass, float(u(*FRICTION)), colour)


def main():
    import twin_expert

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    for _ in range(args.count):
        shape = random_shape(rng)
        size = np.ptp(np.vstack(shape.parts), 0) * 1000
        grasp = twin_expert.plan_grasp(shape)
        print(f'{shape.family:9s} {size[0]:4.0f} x {size[1]:3.0f} x {size[2]:3.0f} mm, {shape.mass * 1000:3.0f} g; '
              f'grasp width {grasp.width * 1000:4.1f} mm at ({grasp.centre[0] * 1000:+.0f}, {grasp.centre[1] * 1000:+.0f}) mm, '
              f'across {np.degrees(grasp.angle):+4.0f} deg')


if __name__ == '__main__':
    main()
