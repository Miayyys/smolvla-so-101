"""Local image-based closed-loop policy evaluation in the reconstructed twin; never opens robot devices.

Run with .venv-gs/bin/python scripts/twin_policy_eval.py --out runs/twin/policy_recap_v2.
Rendering/physics use .venv-gs, policy inference uses .venv in a second local process. Raw uint8 images cross
an anonymous pipe losslessly. The policy receives images, measured joint state and task text only.
Synchronous replanning executes --execute actions at 30 Hz; this is not the real-arm RTC inference engine.
"""

import argparse
import contextlib
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_packet(stream):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            chunk = stream.read(size - len(result))
            if not chunk:
                raise EOFError('Local policy worker pipe closed')
            result.extend(chunk)
        return bytes(result)
    size, = struct.unpack('!Q', exact(8))
    return pickle.loads(exact(size))


def write_packet(stream, value):
    payload = pickle.dumps(value, protocol=5)
    stream.write(struct.pack('!Q', len(payload)))
    stream.write(payload)
    stream.flush()


def worker(args):
    # Keep stdout exclusively for the local binary protocol, including during third-party model loading.
    channel = sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from types import SimpleNamespace
        from so101vla.policy import load_policy
        from so101vla.rollout import install
        torch.set_num_threads(4)
        if args.smooth:
            install(args.smooth)
        policy, pre, post = load_policy(SimpleNamespace(model_dir=Path(args.model),
            vlm_assets_dir=ROOT / 'artifacts/transfer/smolvlm2_assets', device='cuda'))
    write_packet(channel, {'ready': True, 'model': args.model})
    while True:
        request = read_packet(sys.stdin.buffer)
        if request['op'] == 'quit':
            return
        if request['op'] == 'reset':
            policy.reset()
            torch.manual_seed(request['seed'])
            write_packet(channel, {'reset': True})
            continue
        raw = {'observation.state': torch.from_numpy(request['state'][None].astype('float32')),
               'task': [args.task]}
        for key, name in [('camera1', 'top'), ('camera2', 'wrist')]:
            rgb = request[name][..., ::-1].copy()
            raw[f'observation.images.{key}'] = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            actions = post(policy.predict_action_chunk(pre(raw))).float().cpu().numpy()[0]
        if actions.shape != (50, 6) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid policy actions: {actions.shape}')
        write_packet(channel, {'actions': actions})


def cases(args):
    import twin_shapes
    result = []
    if args.cases_json:
        for row in json.loads(Path(args.cases_json).read_text())['cases']:
            directory = Path(row['expert_recording'])
            meta = json.loads((directory / 'meta.json').read_text())
            shape = twin_shapes.random_shape(np.random.default_rng(meta['seed']))
            if (shape.family != meta['family'] or not np.allclose(shape.colour, meta['colour'])
                    or not np.isclose(shape.mass, meta['mass_kg'])
                    or not np.isclose(shape.friction, meta['friction'])):
                raise ValueError(f'Generated shape differs from recording: {directory}')
            result.append({**row, 'shape': shape, 'start': meta['start'],
                           'placements': meta['placements'], 'pair': meta['seed'],
                           'side': meta['side'], 'family': meta['family']})
        if not result:
            raise ValueError('Empty generated case manifest')
        return result
    # Exact initial states/geometry from successful demonstrations used in training.
    for seed in (5011, 5017):
        directory = Path(f'runs/twin/generated_v2b/ep_{seed:06d}')
        meta = json.loads((directory / 'meta.json').read_text())
        shape = twin_shapes.random_shape(np.random.default_rng(seed))
        result.append({'name': f'seen_{seed}', 'shape': shape, 'start': meta['start'],
                       'placements': meta['placements'], 'pair': seed, 'group': 'seen',
                       'expert_recording': str(directory), 'side': meta['side']})
    corners = np.array([[sx * .040, sy * .0075, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0, .012)])
    # Identical geometry/physics/noise per colour pair; 8 x 1.5 x 1.2 cm is a diagnostic proxy, not a measured real object.
    for side, y in [('left', .12), ('right', -.12)]:
        for yaw in (0, 90):
            for colour, rgb in [('light', (.65, .65, .65)), ('black', (.035, .035, .035))]:
                shape = twin_shapes.Shape('bar', [corners.copy()], .025, .7, rgb)
                result.append({'name': f'{side}_{yaw}_{colour}', 'shape': shape,
                    'start': [36, -73, 58, 79, -101, 1],
                    'placements': {'object': [.24, y, float(np.radians(yaw))], 'box': [.28, -y, 0]},
                    'pair': (100 if side == 'left' else 200) + yaw,
                    'group': colour, 'side': side, 'yaw_deg': yaw})
    if args.only:
        result = [r for r in result if r['name'] in args.only]
    if not result:
        raise ValueError('No cases selected')
    return result


def diagnostics(scene, shape, grasp):
    import twin_expert
    obj = scene.object_poses()['object']
    gripper = scene.pose(scene.link_ids[5])
    expected = obj[:3, :3] @ np.array([*grasp.centre, 0]) + obj[:3, 3]
    point = gripper[:3, :3] @ np.array([twin_expert.FIXED_FACE_X + twin_expert.PUSH_M + grasp.width / 2,
                                      0, twin_expert.TIP_Z]) + gripper[:3, 3]
    across = obj[:3, :3] @ np.array([np.cos(grasp.angle), np.sin(grasp.angle), 0])
    a, b = gripper[:2, 0], across[:2]
    cosine = abs(float(a @ b)) / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b))
    error = float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
    box = scene.object_poses()['box']
    local = np.linalg.inv(box) @ obj
    vertices = np.vstack(shape.parts) @ local[:3, :3].T + local[:3, 3]
    return {'object_xyz': obj[:3, 3].copy(), 'jaw_open': scene.opening_percent(),
            'distance_xy_mm': float(np.linalg.norm(point[:2] - expected[:2]) * 1000),
            'distance_z_mm': float(abs(point[2] - expected[2]) * 1000), 'angle_error_deg': error,
            'centre_in_box': twin_expert.object_in_box(scene.object_poses()),
            'fully_in_box': bool(twin_expert.object_in_box(scene.object_poses())
                                 and (np.abs(vertices[:, :2]) < .0433).all())}


def expert_check(scene, case, grasp):
    import twin_expert
    # Verify the matching world can actually be solved by the scripted expert.
    if 'expert_recording' in case:
        actions = np.load(Path(case['expert_recording']) / 'frames.npz')['action']
    else:
        kinematics = twin_expert.Kinematics(scene)
        path, _, _ = twin_expert.plan_motion(scene, kinematics, grasp, scene.to_rad(case['start']))
        actions = np.asarray([scene.from_rad(a) for a in path])
    for action in actions:
        scene.step(action)
    for _ in range(15):
        scene.step(actions[-1])
    d = diagnostics(scene, case['shape'], grasp)
    return {key: d[key] for key in ('centre_in_box', 'fully_in_box')}


def grasp_metrics(trace, initial_z):
    """Closing diagnostics without an arbitrary 20% opening threshold (thin objects need less opening)."""
    lifting = np.flatnonzero(trace[:, 2] - initial_z > .01)
    closing = np.diff(trace[:, 3], prepend=trace[0, 3]) < -.05
    nearby = np.flatnonzero(closing & (trace[:, 4] < 40) & (trace[:, 5] < 40))
    nearest = (int(nearby[np.argmin(np.linalg.norm(trace[nearby, 4:6], axis=1))]) if len(nearby) else None)
    return {'first_1cm_lift_frame': int(lifting[0]) if len(lifting) else None,
            'angle_at_first_1cm_lift_deg': float(trace[lifting[0], 6]) if len(lifting) else None,
            'near_closing_frames': len(nearby), 'nearest_closing_frame': nearest,
            'nearest_closing_angle_deg': float(trace[nearest, 6]) if nearest is not None else None,
            'nearest_closing_xy_mm': float(trace[nearest, 4]) if nearest is not None else None,
            'nearest_closing_z_mm': float(trace[nearest, 5]) if nearest is not None else None}


def teacher_forced(child, case, std):
    """Predict on recorded expert observations, without allowing policy errors to change the input."""
    import cv2
    directory = Path(case['expert_recording'])
    meta = json.loads((directory / 'meta.json').read_text())
    data = np.load(directory / 'frames.npz')
    indices = sorted({meta['phases'][phase] - 1 for phase in
                      ('approach', 'descend', 'close', 'lift', 'carry')})
    images = {}
    for camera in ('top', 'wrist'):
        cap = cv2.VideoCapture(str(directory / f'{camera}.mp4'))
        try:
            for i in range(indices[-1] + 1):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode {directory}/{camera}.mp4 frame {i}')
                if i in indices:
                    images[camera, i] = frame
        finally:
            cap.release()
    errors = []
    for i in indices:
        write_packet(child.stdin, {'op': 'reset', 'seed': 19000 + case['pair'] + i})
        read_packet(child.stdout)
        write_packet(child.stdin, {'op': 'predict', 'state': data['state'][i],
                                  'top': images['top', i], 'wrist': images['wrist', i]})
        prediction = read_packet(child.stdout)['actions']
        target = data['action'][i:i + 50]
        error = np.abs(prediction[:len(target)] - target)
        errors.append({'frame': i, 'normalized_mae': float((error / std).mean()),
                       'per_joint_mae': error.mean(0).tolist(), 'valid_actions': len(target)})
    return {'observations': errors, 'normalized_mae': float(np.mean([e['normalized_mae'] for e in errors])),
            'per_joint_mae': np.mean([e['per_joint_mae'] for e in errors], axis=0).tolist(),
            'scope': 'recorded expert images/state; smoothed physical action chunks; not closed-loop success'}


def evaluate(args):
    import cv2
    import twin_expert
    import twin_scene
    from twin_generate import Renderer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError(f'{out}/summary.json already exists')
    selected = cases(args)
    description = {k: v for k, v in vars(args).items() if k != 'worker'}
    (out / 'config.json').write_text(json.dumps({**description, 'local_only': True,
        'control': 'synchronous receding horizon, no wall-clock inference delay in physics',
        'image_transport': 'lossless raw uint8 over local anonymous pipes',
        'test_cases': [c['name'] for c in selected]}, indent=2) + '\n')
    stderr = (out / 'policy_worker.log').open('a')
    child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()),
        '--worker', '--model', args.model, '--smooth', str(args.smooth), '--task', args.task],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, cwd=ROOT)
    summaries = []
    try:
        print('policy', read_packet(child.stdout), flush=True)
        renderer = Renderer(args.scan, args.box)
        for case in selected:
            name = case['name']
            directory = out / name
            directory.mkdir(exist_ok=True)
            scene = twin_scene.Scene(args.scan, {'box': args.box, 'object': case['shape']})
            scene.reset(case['start'], case['placements'])
            grasp = twin_expert.plan_grasp(case['shape'])
            oracle = expert_check(scene, case, grasp)
            scene.reset(case['start'], case['placements'])
            renderer.prepare(case['shape'])
            fit = None
            if args.teacher_forced_stats:
                std = np.asarray(json.loads(Path(args.teacher_forced_stats).read_text())['action']['std'])
                fit = teacher_forced(child, case, std)
            write_packet(child.stdin, {'op': 'reset', 'seed': 9000 + case['pair']})
            read_packet(child.stdout)
            initial_z = scene.object_poses()['object'][2, 3]
            trace, states, actions, closing = [], [], [], []
            video = cv2.VideoWriter(str(directory / 'preview.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                    30 / args.execute, (1280, 480))
            if not video.isOpened():
                raise RuntimeError('Video encoder did not open')
            started = time.monotonic()
            total = int(args.seconds * 30)
            previous_open = scene.opening_percent()
            try:
                for frame in range(0, total, args.execute):
                    top, wrist = renderer.frame(scene)
                    tile = np.hstack([cv2.resize(top, (640, 480)), wrist])
                    cv2.putText(tile, f'{name}  t={frame / 30:.1f}s', (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                    video.write(tile)
                    if frame == 0:
                        cv2.imwrite(str(directory / 'initial.jpg'), tile)
                    state = scene.from_rad(scene.data.qpos[:6].copy()).astype('float32')
                    write_packet(child.stdin, {'op': 'predict', 'top': top, 'wrist': wrist, 'state': state})
                    chunk = read_packet(child.stdout)['actions']
                    for offset, action in enumerate(chunk[:min(args.execute, total - frame)]):
                        scene.step(action)
                        d = diagnostics(scene, case['shape'], grasp)
                        if previous_open >= 20 and d['jaw_open'] < 20:
                            closing.append({'frame': frame + offset, **{k: d[k] for k in
                                ('distance_xy_mm', 'distance_z_mm', 'angle_error_deg')}})
                        previous_open = d['jaw_open']
                        trace.append([*d['object_xyz'], d['jaw_open'], d['distance_xy_mm'],
                                      d['distance_z_mm'], d['angle_error_deg'], d['centre_in_box'], d['fully_in_box']])
                        states.append(scene.from_rad(scene.data.qpos[:6].copy()))
                        actions.append(action)
                    if frame % (args.execute * 10) == 0:
                        print(json.dumps({'case': name, 'sim_s': (frame + args.execute) / 30,
                                          'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)
                top, wrist = renderer.frame(scene)
                cv2.imwrite(str(directory / 'final.jpg'), np.hstack([cv2.resize(top, (640, 480)), wrist]))
            finally:
                video.release()
            trace = np.asarray(trace)
            last = diagnostics(scene, case['shape'], grasp)
            near = [c for c in closing if c['distance_xy_mm'] < 40 and c['distance_z_mm'] < 40]
            summary = {k: v for k, v in case.items() if k not in ('shape',)}
            summary.update(oracle=oracle, fully_in_box=last['fully_in_box'], centre_in_box=last['centre_in_box'],
                lifted_3cm=bool((trace[:, 2] - initial_z).max() > .03),
                max_lift_mm=float((trace[:, 2] - initial_z).max() * 1000), closing_events=closing,
                near_closing_events=near, min_distance_xy_mm=float(trace[:, 4].min()),
                elapsed_s=round(time.monotonic() - started, 2), colour=list(case['shape'].colour))
            summary['grasp_diagnostics'] = grasp_metrics(trace, initial_z)
            if fit is not None:
                summary['teacher_forced'] = fit
            np.savez(directory / 'trace.npz', states=np.asarray(states), actions=np.asarray(actions), metrics=trace)
            (directory / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
            summaries.append(summary)
            (out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
            print('RESULT', json.dumps(summary), flush=True)
    finally:
        if child.poll() is None:
            try:
                write_packet(child.stdin, {'op': 'quit'})
                child.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                child.terminate()
                child.wait(timeout=10)
        stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model', default='runs/so101_recap_v2/best')
    parser.add_argument('--out', default='runs/twin/policy_recap_v2')
    parser.add_argument('--scan', default='runs/scan/table_curtain')
    parser.add_argument('--box', default='runs/scan/box')
    parser.add_argument('--task', default='Put the object in the white box.')
    parser.add_argument('--smooth', type=float, default=2)
    parser.add_argument('--execute', type=int, default=30)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--cases-json', help='Manifest of original generated episode directories and dataset splits')
    parser.add_argument('--teacher-forced-stats', help='Also compare 5 recorded expert observations per episode; action std JSON')
    args = parser.parse_args()
    if not 1 <= args.execute <= 50 or args.seconds <= 0:
        parser.error('--execute must be 1..50 and --seconds must be positive')
    os.chdir(ROOT)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    (worker if args.worker else evaluate)(args)


if __name__ == '__main__':
    main()
