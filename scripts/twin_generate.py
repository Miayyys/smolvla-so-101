"""Generate pick-and-place episodes in the digital twin: the scripted expert's motion with rendered camera images.
Runs in .venv-gs (physics and rendering).

    .venv-gs/bin/python scripts/twin_generate.py --preview --seed 3 --out runs/twin/generate_preview
    .venv-gs/bin/python scripts/twin_generate.py --episodes 200 --out runs/twin/generated_v1

Each seed gives a random elongated object and placement (twin_expert.py). The physics is run once without images;
only episodes that end with the object in the box are run again (the physics is deterministic) and rendered: the top
camera at 1280 x 720 and the wrist camera at 640 x 480, as recorded on the real arm. Per kept episode the output
directory gets ep_<seed>/ with top.mp4 and wrist.mp4 (H.264, near lossless), frames.npz (action, state in dataset
units; expert = False on the frames of a deliberate miss) and meta.json. twin_to_lerobot.py turns the directory into a LeRobot dataset.

--preview writes one image of a few frames of one episode, real-size tiles side by side.
"""

import argparse
import json
import time
from pathlib import Path

import av
import cv2
import numpy as np

import twin_expert
import twin_render
import twin_shapes

BOX_TOP_GAIN = [0.99, 1.03, 1.13]  # the box's colour from the wrist scan to the top camera (median of the replays)
CRF = 16
FPS = 30


def similarity_to_reconstruction(scene, pose):
    """A pose of a generated object (metres) in the wrist reconstruction's frame, its size scaled with the frame."""
    out = scene.to_reconstruction(pose)
    out[:3, :3] /= scene.recon_scale
    return out


class Renderer:
    def __init__(self, scan, box):
        self.twin = twin_render.Twin(scan, Path(scan) / 'arm')
        self.box, _ = self.twin.load_object(box)

    def prepare(self, shape):
        self.twin.add_shape('object', shape.pieces(), shape.colour)

    def frame(self, scene):
        poses = scene.object_poses()
        wrist = self.twin.wrist(scene.to_reconstruction(scene.wrist_camera()), scene.opening_percent(),
                                [(self.box, scene.to_reconstruction(poses['box'])),
                                 ('object', similarity_to_reconstruction(scene, poses['object']))])
        top = self.twin.top(scene.link_poses(), [(self.box, poses['box']), ('object', poses['object'])],
                            {self.box: BOX_TOP_GAIN})
        return top, wrist


class Video:
    def __init__(self, path, width, height):
        self.container = av.open(str(path), 'w')
        self.stream = self.container.add_stream('libx264', rate=FPS)
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, 'yuv420p'
        self.stream.options = {'crf': str(CRF)}

    def add(self, image):
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(image, format='bgr24')):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def settings(args):
    return {'scan': args.scan, 'box': args.box, 'family': args.family, 'key': None,
            'random_start': args.random_start, 'regrasp': args.regrasp, 'wide': args.wide}


def shape_of(seed, family):
    """The same object twin_expert.episode makes for the seed (its first draw from the seed's generator)."""
    return twin_shapes.random_shape(np.random.default_rng(seed), family)


def preview(args, renderer):
    renderer.prepare(shape_of(args.seed, args.family))
    tiles = []

    def watch(scene, k, phases):
        names = ('miss_close', 'miss_end', 'approach', 'descend', 'close', 'lift', 'carry', 'open')
        if k == 0 or k in [phases[name] - 1 for name in names if name in phases]:
            top, wrist = renderer.frame(scene)
            tile = np.hstack([cv2.resize(top, (853, 480)), wrist])
            cv2.putText(tile, f'frame {k}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            tiles.append(tile)

    result = twin_expert.episode((args.seed, {**settings(args), 'watch': watch}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f'preview_{args.seed}.jpg'), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print({k: v for k, v in result.items() if k not in ('actions', 'states', 'held', 'phases')})


def generate(args, renderer):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept, tried, seed = 0, 0, args.seed
    started = time.time()
    while kept < args.episodes:
        tried += 1
        trial = twin_expert.episode((seed, settings(args)))
        if not trial['success'] or (out / f'ep_{seed:06d}/meta.json').exists():
            kept += bool(trial['success'])
            seed += 1
            continue
        shape = shape_of(seed, args.family)
        renderer.prepare(shape)
        directory = out / f'ep_{seed:06d}'
        directory.mkdir(exist_ok=True)
        videos = {'top': Video(directory / 'top.mp4', 1280, 720), 'wrist': Video(directory / 'wrist.mp4', 640, 480)}

        def watch(scene, k, phases):
            top, wrist = renderer.frame(scene)
            videos['top'].add(top)
            videos['wrist'].add(wrist)

        result = twin_expert.episode((seed, {**settings(args), 'watch': watch, 'keep': True}))
        for video in videos.values():
            video.close()
        if not result['success']:  # should not happen: the physics is deterministic
            raise RuntimeError(f'seed {seed} succeeded once and failed when rendered')
        # expert: False on a deliberate miss (twin_expert.plan_miss), which is not to be imitated
        expert = np.arange(len(result['actions'])) >= result['phases'].get('miss_end', 0)
        np.savez(directory / 'frames.npz', action=result['actions'].astype(np.float32),
                 state=result['states'].astype(np.float32), expert=expert)
        meta = {k: v for k, v in result.items() if k not in ('actions', 'states', 'held')}
        meta.update(task=args.task, colour=shape.colour, mass_kg=shape.mass, friction=shape.friction)
        (directory / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
        kept += 1
        seed += 1
        print(f'{kept}/{args.episodes}: seed {seed - 1} {shape.family} {result["side"]}, {result["frames"]} frames '
              f'({time.time() - started:.0f} s, {tried} tried)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes', type=int, default=10, help='episodes to keep (ending in the box)')
    parser.add_argument('--seed', type=int, default=0, help='first seed')
    parser.add_argument('--family', choices=twin_shapes.FAMILIES)
    parser.add_argument('--task', default='Put the object in the white box.')
    parser.add_argument('--scan', default='runs/scan/table_curtain')
    parser.add_argument('--box', default='runs/scan/box')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--random-start', action='store_true', help='start near one of twin_expert.START_POSES')
    parser.add_argument('--regrasp', type=float, default=0.0, help='share of episodes whose first attempt misses')
    parser.add_argument('--wide', action='store_true', help='placements also near the base and further out')
    args = parser.parse_args()
    renderer = Renderer(args.scan, args.box)
    (preview if args.preview else generate)(args, renderer)


if __name__ == '__main__':
    main()
