"""Real-demonstration SmolVLA fine-tuning with episode-isolated validation.

Uses LeRobot's policy, processors, flow-matching loss and padding contract.
The packed JPEG cache avoids repeated AV1 seeking; alignment is checked against
LeRobot's own decoder before any training. No test frames enter training/stats.
"""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def emit(out, event):
    event['time_unix'] = time.time()
    with (out / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    print(json.dumps(event), flush=True)


def camera_map(cfg):
    """Model camera key -> dataset video key (default: the old single top camera as camera1)."""
    return cfg.get('camera_map', {'observation.images.camera1': 'observation.images.top'})


def cache_name(video_key):
    return video_key.split('.')[-1] + '_frames.jpeg.bin'


def prepare(cfg, out):
    """Cache a LeRobot v3 dataset (any number of data/video files) as per-camera JPEG streams.

    Every episode's frames are decoded from its own video file segment (``from_timestamp`` in the episode
    metadata) and checked frame by frame against the table timestamps; a few rows are compared with LeRobot's
    own reader. Normalization statistics use training episodes only.

    With ``trim_static_tail`` = {tolerance_deg, keep_frames}, each episode's trailing frames in which every joint
    stays within ``tolerance_deg`` of the final state are cut down to ``keep_frames``: the cut frames are cached
    but never sampled, never enter the statistics, and action chunks are padded after the kept end. Without
    ``task`` in the config, each frame's language instruction is its own task from ``meta/tasks.parquet``.

    With an ``intervention`` column (human-correction rollouts, so101vla/correction.py, merged with demos tagged all
    True by scripts/merge_corrections.py), only human-driven frames are sampled and enter the statistics, and each
    one's action chunk is padded after the end of its own correction run, so policy actions never become targets.
    Validation episodes must be fully human-driven. ``correction_repeat`` = k lists the human frames of correction
    episodes (episodes with any policy frame) k times in the sampled rows (uniform sampling, so k times as likely);
    statistics and the reported frame counts use each frame once. ``stats_from`` = path of an earlier run's stats.json
    reuses its normalization (fine-tuning from that run with ``init_weights`` keeps the inputs/outputs the weights were
    trained for, and keeps the normalized validation error comparable).
    """
    import av
    import pandas as pd
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from so101vla import correction as so101_correction

    root = Path(cfg['dataset_root'])
    info = json.loads((root / 'meta/info.json').read_text())
    fps = info['fps']
    partition = json.loads(Path(cfg['partition']).read_text())
    splits = partition['splits']
    assert all(not set(splits[a]) & set(splits[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    data_files = sorted((root / 'data').rglob('*.parquet'))
    table = pd.concat([pd.read_parquet(p) for p in data_files]).sort_values('index').reset_index(drop=True)
    if not np.array_equal(table['index'].to_numpy(), np.arange(len(table))):
        raise ValueError('Dataset rows are not a contiguous 0..N-1 index')
    eps = table['episode_index'].to_numpy()
    if not set(eps) == set(splits['train']) | set(splits['validation']) | set(splits['test']):
        raise ValueError('Partition does not cover exactly the dataset episodes')
    states = np.stack(table['observation.state'].to_numpy()).astype('float32')
    actions = np.stack(table['action'].to_numpy()).astype('float32')
    timestamps = table['timestamp'].to_numpy(dtype='float64')
    tasks = pd.read_parquet(root / 'meta/tasks.parquet')
    task_names = np.empty(len(tasks), dtype=object)
    task_names[tasks['task_index'].to_numpy()] = tasks.index.to_numpy()
    task_index = table['task_index'].to_numpy(dtype='int64')
    trim = cfg.get('trim_static_tail')
    human = (np.stack(table['intervention'].to_numpy()).reshape(-1).astype(bool) if 'intervention' in table
             else np.ones(len(eps), dtype=bool))
    kept = np.ones(len(eps), dtype=bool)
    end_rows = np.empty(len(eps), dtype='int64')
    validation_rows = []; trimmed = {}
    for ep in sorted(set(eps.tolist())):
        rows = np.flatnonzero(eps == ep)
        assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1))
        end = rows[-1]
        if trim:
            still = (np.abs(states[rows] - states[rows[-1]]) <= trim['tolerance_deg']).all(1)
            first = len(rows)
            while first > 0 and still[first - 1]: first -= 1
            end = min(rows[-1], rows[0] + first + trim['keep_frames'] - 1)
            kept[end + 1:rows[-1] + 1] = False
            trimmed[int(ep)] = int(rows[-1] - end)
        end_rows[rows] = end
        if not human[rows].all():
            if ep in splits['validation']:
                raise ValueError(f'Validation episode {ep} has policy-driven frames')
            kept[rows[~human[rows]]] = False
            for start, stop in so101_correction.runs(human[rows]):
                end_rows[rows[start:stop]] = min(rows[stop - 1], end)
        if ep in splits['validation']:
            validation_rows.extend(np.linspace(rows[0], max(rows[0], end - 49), cfg['validation_frames_per_episode'], dtype=int).tolist())
    train = np.isin(eps, splits['train']) & kept
    train_rows = np.flatnonzero(train)
    correction_episodes = np.unique(eps[~human])
    correction_rows = train_rows[np.isin(eps[train_rows], correction_episodes)]
    sampled_rows = np.concatenate([train_rows] + [correction_rows] * (cfg.get('correction_repeat', 1) - 1))
    stats = {key: {'mean': a[train].mean(0).tolist(), 'std': np.maximum(a[train].std(0), 1e-6).tolist()}
             for key, a in [('observation.state', states), ('action', actions)]}
    if cfg.get('stats_from'):
        stats = json.loads(Path(cfg['stats_from']).read_text())
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((root / 'meta/episodes').rglob('*.parquet'))]).set_index('episode_index')

    selected_checks = [0, len(eps) // 2, len(eps) - 1]
    arrays = {}
    caches = {}
    max_alignment_error = 0.0
    for video_key in camera_map(cfg).values():
        cache_path = out / cache_name(video_key)
        partial = cache_path.with_suffix('.partial')
        offsets = np.zeros(len(eps), dtype='int64'); lengths = np.zeros(len(eps), dtype='int64')
        with partial.open('wb') as target:
            for ep in sorted(set(eps.tolist())):
                rows = np.flatnonzero(eps == ep)
                meta = episodes.loc[ep]
                path = root / info['video_path'].format(video_key=video_key, chunk_index=int(meta[f'videos/{video_key}/chunk_index']),
                                                        file_index=int(meta[f'videos/{video_key}/file_index']))
                expected = float(meta[f'videos/{video_key}/from_timestamp']) + timestamps[rows]
                count = 0
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    stream.thread_type = 'AUTO'
                    container.seek(int(max(expected[0] - 1.0, 0) / stream.time_base), stream=stream)
                    for frame in container.decode(stream):
                        t = float(frame.pts * stream.time_base)
                        if t < expected[0] - 0.5 / fps:
                            continue
                        if count == len(rows):
                            break
                        error = abs(t - expected[count])
                        max_alignment_error = max(max_alignment_error, error)
                        if error > 1e-3:
                            raise ValueError(f'{video_key} episode {ep}: video/table timestamp mismatch at frame {count}: {error}')
                        array = frame.to_ndarray(format='rgb24')
                        buffer = io.BytesIO()
                        Image.fromarray(array).save(buffer, format='JPEG', quality=95, subsampling=0)
                        data = buffer.getvalue()
                        row = rows[count]
                        offsets[row] = target.tell(); lengths[row] = len(data); target.write(data)
                        if row in selected_checks:
                            arrays[(video_key, row)] = array
                        count += 1
                if count != len(rows):
                    raise ValueError(f'{video_key} episode {ep}: {count} video frames for {len(rows)} table rows')
                emit(out, {'stage': 'cache', 'camera': video_key, 'episode': int(ep), 'frames': int(rows[-1]) + 1, 'total': len(eps)})
        partial.replace(cache_path)
        caches[video_key] = {'offsets': offsets, 'lengths': lengths, 'sha256': sha(cache_path), 'bytes': cache_path.stat().st_size}

    # LeRobot's own reader independently validates RGB decoding and time offsets.
    dataset = LeRobotDataset(info.get('repo_id', 'local/dataset'), root=root.resolve(), video_backend='pyav', return_uint8=True)
    rgb_errors = {}
    for video_key in camera_map(cfg).values():
        rgb_errors[video_key] = []
        with (out / cache_name(video_key)).open('rb') as stream:
            for row in selected_checks:
                official = dataset[row][video_key].permute(1, 2, 0).numpy()
                if not np.array_equal(official, arrays[(video_key, row)]):
                    raise ValueError(f'{video_key}: sequential decoder differs from official LeRobot RGB frame at row {row}')
                stream.seek(caches[video_key]['offsets'][row])
                image = np.asarray(Image.open(io.BytesIO(stream.read(int(caches[video_key]['lengths'][row])))).convert('RGB'))
                rgb_errors[video_key].append(float(np.abs(image.astype('float32') - official).mean()))
    np.savez(out / 'data.npz', states=states, actions=actions, episodes=eps, end_rows=end_rows,
             train_rows=sampled_rows, validation_rows=validation_rows, task_index=task_index, task_names=task_names.astype(str),
             **{f'{k.split(".")[-1]}_{name}': c[name] for k, c in caches.items() for name in ('offsets', 'lengths')})
    (out / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    manifest = {'config': cfg, 'splits': splits, 'frames': len(eps), 'train_frames': len(train_rows),
                'validation_rows': validation_rows, 'tasks': task_names.tolist(),
                'train_frames_by_task': {task_names[i]: int((task_index[train_rows] == i).sum()) for i in range(len(task_names))},
                'trimmed_tail_frames': trimmed, 'trimmed_train_frames': int((np.isin(eps, splits['train']) & ~kept & human).sum()),
                'policy_frames_excluded': int((~human).sum()),
                'correction_train_frames': len(correction_rows), 'correction_repeat': cfg.get('correction_repeat', 1),
                'sampled_rows': len(sampled_rows),
                'data_sha256': {str(p.relative_to(root)): sha(p) for p in data_files},
                'partition_sha256': sha(cfg['partition']),
                'caches': {k: {'sha256': c['sha256'], 'bytes': c['bytes']} for k, c in caches.items()},
                'max_timestamp_error_s': max_alignment_error, 'jpeg_rgb_mae_0_255': rgb_errors,
                'official_decoder_exact_checks': selected_checks,
                'stats_sha256': sha(out / 'stats.json'),
                'stats_from_sha256': sha(cfg['stats_from']) if cfg.get('stats_from') else None}
    (out / 'prepared.json').write_text(json.dumps(manifest, indent=2) + '\n')
    emit(out, {'stage': 'prepared', 'train_frames': len(train_rows), 'validation_rows': len(validation_rows),
               'trimmed_train_frames': manifest['trimmed_train_frames'], 'policy_frames_excluded': manifest['policy_frames_excluded'],
               'correction_train_frames': len(correction_rows), 'sampled_rows': len(sampled_rows),
               'train_frames_by_task': manifest['train_frames_by_task'],
               'cache_bytes': {k: c['bytes'] for k, c in caches.items()}, 'max_timestamp_error_s': max_alignment_error})


class Frames:
    def __init__(self, out, cfg):
        self.data = {k: v for k, v in np.load(out / 'data.npz').items()}
        self.camera_map = camera_map(cfg)
        self.streams = {key: (out / cache_name(video_key)).open('rb') for key, video_key in self.camera_map.items()}
        # A fixed config task (experiment 1) or each frame's own dataset task.
        self.task = cfg.get('task')

    def batch(self, rows):
        import torch
        from PIL import Image
        images = {key: [] for key in self.camera_map}; actions, masks = [], []
        for row in rows:
            for key, video_key in self.camera_map.items():
                cam = video_key.split('.')[-1]
                self.streams[key].seek(int(self.data[f'{cam}_offsets'][row]))
                image = Image.open(io.BytesIO(self.streams[key].read(int(self.data[f'{cam}_lengths'][row])))).convert('RGB')
                images[key].append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
            indices = row + np.arange(50)
            masks.append(indices > self.data['end_rows'][row])
            actions.append(self.data['actions'][np.minimum(indices, self.data['end_rows'][row])])
        return {**{key: torch.stack(v).float() / 255 for key, v in images.items()},
                'observation.state': torch.from_numpy(self.data['states'][rows].copy()),
                'action': torch.from_numpy(np.stack(actions)),
                'action_is_pad': torch.from_numpy(np.stack(masks)),
                'task': [self.task or str(self.data['task_names'][self.data['task_index'][row]]) for row in rows]}


def train(cfg, out, smoke, resume=False, stop_step=None):
    import torch
    from lerobot.policies import make_pre_post_processors
    from so101vla.policy import load_policy
    from safetensors.torch import save_file

    prepared = json.loads((out / 'prepared.json').read_text())
    if prepared['config'] != cfg or prepared['partition_sha256'] != sha(cfg['partition']):
        raise ValueError('Prepared data/config identity differs')
    if (out / 'progress.json').exists() and not smoke and not resume:
        raise ValueError('Training already exists; preserve it instead of overwriting')
    stats = json.loads((out / 'stats.json').read_text())
    if sha(out / 'stats.json') != prepared['stats_sha256']:
        raise ValueError('Normalization statistics changed')
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    policy, _, _ = load_policy(SimpleNamespace(model_dir=Path(cfg['model_dir']).resolve(),
        vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(), device='cuda'))
    policy.float()
    policy.config.resize_imgs_with_padding = (cfg['image_size'], cfg['image_size'])
    # Keep only the configured cameras in the model interface; missing views are not duplicated.
    cameras = list(cfg.get('camera_map', {'observation.images.camera1': None}))
    policy.config.input_features = {k: v for k, v in policy.config.input_features.items()
                                    if k in ['observation.state', *cameras]}
    policy.config.empty_cameras = 0
    if cfg.get('init_weights'):
        # Continue from an earlier fine-tuned checkpoint of the same structure (fresh optimizer and schedule).
        from safetensors.torch import load_file
        policy.load_state_dict(load_file(cfg['init_weights'], device='cuda'), strict=True)
    goal_cfg = cfg.get('goal_conditioning')
    if goal_cfg:
        from so101vla import goal as so101_goal
        so101_goal.attach(policy)
    pre, post = make_pre_post_processors(policy.config, cfg['model_dir'],
        preprocessor_overrides={
            'tokenizer_processor': {'tokenizer_name': str(Path(cfg['vlm_assets_dir']).resolve())},
            'device_processor': {'device': 'cuda'}, 'normalizer_processor': {'stats': stats}},
        postprocessor_overrides={'unnormalizer_processor': {'stats': stats}})
    if cfg.get('evaluation') == 'scripted_sim':
        from so101vla.scripted import ScriptedFrames
        frames = ScriptedFrames(cfg, out)
    else:
        frames = Frames(out, cfg)
    parameters = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    if any('.vlm.' in n for n, _ in parameters):
        raise ValueError('VLM unexpectedly trainable')
    trained_layers = sorted({int(n.split('.layers.')[1].split('.')[0]) for n, _ in parameters if 'lm_expert.layers.' in n})
    assert trained_layers == list(range(16))
    (out / 'trainable.json').write_text(json.dumps({'names': [n for n, _ in parameters],
        'parameters': sum(p.numel() for _, p in parameters), 'expert_layers': trained_layers,
        'model_source_sha256': sha(Path(cfg['model_dir']) / 'model.safetensors'),
        'init_weights_sha256': sha(cfg['init_weights']) if cfg.get('init_weights') else None}, indent=2) + '\n')
    if goal_cfg:
        # Expert side (incl. goal_embedding) and planner are separate groups: disjoint parameters, own clipping.
        expert_parameters = [(n, p) for n, p in parameters if not so101_goal.planner_parameter(n)]
        planner_parameters = [(n, p) for n, p in parameters if so101_goal.planner_parameter(n)]
        groups = [{'params': [p for _, p in expert_parameters], 'lr_scale': 1.0},
                  {'params': [p for _, p in planner_parameters], 'lr_scale': goal_cfg['planner_learning_rate_scale']}]
        for part in frames.parts:
            if not {'phase', 'goal_pos', 'goal_rot', 'tcp_pose'} <= set(part):
                raise ValueError('Every training source of a goal-conditioned run needs goal labels')
        rows = frames.data['train_rows']; part_of = np.searchsorted(frames.bases, rows, side='right') - 1
        train_phases = np.concatenate([frames.parts[p]['phase'][rows[part_of == p] - frames.bases[p]] for p in np.unique(part_of)])
        class_weights = torch.tensor(so101_goal.phase_weights(train_phases), dtype=torch.float32, device='cuda')
        emit(out, {'stage': 'goal_setup', 'expert_parameters': sum(p.numel() for _, p in expert_parameters),
                   'planner_parameters': sum(p.numel() for _, p in planner_parameters),
                   'phase_class_weights': dict(zip(so101_goal.PHASES, class_weights.tolist()))})
    else:
        expert_parameters = parameters
        groups = [{'params': [p for _, p in parameters], 'lr_scale': 1.0}]
    optimizer = torch.optim.AdamW(groups, lr=cfg['learning_rate'], betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg['weight_decay'])

    def evaluate(step):
        old_rng = torch.get_rng_state(); old_cuda_rng = torch.cuda.get_rng_state()
        joint_errors = []; planner_rows = []; row_tasks = []
        policy.eval()
        try:
            for row in frames.data['validation_rows']:
                raw = frames.batch([int(row)])
                goal = raw.pop('goal', None)
                target = raw['action'].numpy()[0].copy()
                mask = ~raw['action_is_pad'].numpy()[0]
                torch.manual_seed(100000 + int(row))
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    batch = pre(raw)
                    if goal_cfg:
                        goal = so101_goal.labels_to_device(goal, 'cuda')
                        # Planner first: predict_action_chunk rewrites observation entries of ``batch`` in place.
                        prefix_out, prefix_pad_masks, state = so101_goal.prefix_features(policy, batch)
                        planned = policy.model.goal_planner(policy.model.goal_planner.slow_features(prefix_out, prefix_pad_masks), state)
                        planner_rows.append(so101_goal.planner_metrics(planned, goal))
                        # Offline action check uses the recorded (noise-free) goal, i.e. experiment B's condition.
                        so101_goal.set_goal(policy, so101_goal.goal_vector_from_labels(goal))
                    try:
                        prediction = post(policy.predict_action_chunk(batch)).float().cpu().numpy()[0]
                    finally:
                        if goal_cfg: so101_goal.set_goal(policy, None)
                joint_errors.append(np.abs(prediction[mask] - target[mask])); row_tasks.append(raw['task'][0])
            mae = np.concatenate(joint_errors).mean(0)
            normalized = float(np.mean(mae / np.asarray(stats['action']['std'])))
            result = {'stage': 'validation', 'step': step, 'normalized_action_mae': normalized,
                      'per_joint_mae': mae.tolist(), 'units': 'degrees for 5 arm joints; gripper percentage points',
                      'observations': len(joint_errors), 'scope': 'offline held-out demonstrations; not task success'}
            if len(set(row_tasks)) > 1:
                result['per_task_normalized_action_mae'] = {
                    t: float(np.mean(np.concatenate([e for e, r in zip(joint_errors, row_tasks) if r == t]).mean(0) / np.asarray(stats['action']['std'])))
                    for t in sorted(set(row_tasks))}
            if goal_cfg:
                merged = {k: np.concatenate([r[k] for r in planner_rows]) for k in planner_rows[0]}
                result['action_condition'] = 'recorded goal, no noise, no null'
                result['planner'] = {'phase_accuracy': float(merged['phase_correct'].mean()),
                    'phase_accuracy_by_phase': {p: float(merged['phase_correct'][merged['phase'] == i].mean())
                                                for i, p in enumerate(so101_goal.PHASES) if (merged['phase'] == i).any()},
                    'position_error_mm_median': float(np.median(merged['position_error_mm'])),
                    'position_error_mm_p90': float(np.percentile(merged['position_error_mm'], 90)),
                    'rotation_error_deg_median': float(np.median(merged['rotation_error_deg'])),
                    'aperture_error_mean': float(merged['aperture_error'].mean())}
            emit(out, result)
            return normalized
        finally:
            torch.set_rng_state(old_rng); torch.cuda.set_rng_state(old_cuda_rng); policy.train()

    def save_model(name):
        folder = out / name; folder.mkdir(exist_ok=True)
        temporary = folder / 'model.partial.safetensors'
        save_file({k: v.detach().cpu().contiguous() for k, v in policy.state_dict().items()}, str(temporary))
        temporary.replace(folder / 'model.safetensors')
        policy.config.save_pretrained(folder)
        pre.save_pretrained(folder); post.save_pretrained(folder)

    started = time.perf_counter()
    best_step = stale = start_step = 0
    if resume:
        from safetensors.torch import load_file
        checkpoint = torch.load(out / 'training_state.pt', map_location='cuda', weights_only=False)
        if checkpoint['config'] != cfg or checkpoint['last_weights_sha256'] != sha(out / 'last/model.safetensors'):
            raise ValueError('Resume identity or saved weights changed')
        policy.load_state_dict(load_file(str(out / 'last/model.safetensors'), device='cuda'), strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        rng.bit_generator.state = checkpoint['rng']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu())
        best, best_step = checkpoint['best_metric'], checkpoint['best_step']
        stale, start_step = checkpoint['stale_evaluations'], checkpoint['step']
        emit(out, {'stage': 'resumed', 'step': start_step})
    else:
        best = float('inf') if smoke else evaluate(0)
    # Prefetch: the next micro-batch is decoded on a worker thread while the GPU runs the current one.
    # Rows are still drawn from ``rng`` on this thread in the original order, and checkpoints store the
    # RNG state from before the prefetched draw, so training and resumption are unchanged.
    from concurrent.futures import ThreadPoolExecutor
    loader = ThreadPoolExecutor(max_workers=1)

    def draw():
        state = rng.bit_generator.state
        rows = rng.choice(frames.data['train_rows'], cfg['batch_size'], replace=True)
        return state, loader.submit(frames.batch, rows)

    pending = draw()
    first_param = next(p for n, p in parameters if 'action_out_proj.weight' in n)
    initial = first_param.detach().clone()
    last_step = 2 if smoke else min(cfg['steps'], stop_step or cfg['steps'])
    for step in range(start_step + 1, last_step + 1):
        policy.train(); optimizer.zero_grad(set_to_none=True)
        progress = min(1.0, max(0.0, (step - cfg['warmup_steps']) / (cfg['steps'] - cfg['warmup_steps'])))
        lr = (cfg['minimum_learning_rate'] + 0.5 * (cfg['learning_rate'] - cfg['minimum_learning_rate']) * (1 + math.cos(math.pi * progress)))
        if step <= cfg['warmup_steps']:
            lr = cfg['learning_rate'] * step / cfg['warmup_steps']
        for group in optimizer.param_groups: group['lr'] = lr * group.get('lr_scale', 1.0)
        losses = []; planner_losses = []; planner_terms = []
        accumulation = 1 if smoke else cfg['gradient_accumulation']
        for _ in range(accumulation):
            raw = pending[1].result()
            pending = draw()
            goal = raw.pop('goal', None)
            batch = pre(raw)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if goal_cfg:
                    loss, planner_loss, terms = so101_goal.training_loss(
                        policy, batch, so101_goal.labels_to_device(goal, 'cuda'), goal_cfg, class_weights)
                else:
                    loss, _ = policy(batch)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            if goal_cfg:
                if not torch.isfinite(planner_loss): raise RuntimeError('Nonfinite planner loss')
                # Disjoint parameters and a detached planner input: the sum backpropagates each loss to its own side.
                ((loss + goal_cfg['planner_loss_weight'] * planner_loss) / accumulation).backward()
                planner_losses.append(float(planner_loss.detach())); planner_terms.append(terms)
            else:
                (loss / accumulation).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in expert_parameters], cfg['gradient_clip'])
        if not torch.isfinite(norm) or float(norm) == 0: raise RuntimeError('Invalid gradient norm')
        if goal_cfg:
            planner_norm = torch.nn.utils.clip_grad_norm_([p for _, p in planner_parameters], goal_cfg['planner_gradient_clip'])
            if not torch.isfinite(planner_norm) or float(planner_norm) == 0: raise RuntimeError('Invalid planner gradient norm')
        if step == 1:
            stage_grads = {str(i): sum(float(p.grad.abs().sum()) for n, p in parameters
                if f'lm_expert.layers.{i}.' in n and p.grad is not None) for i in range(16)}
            if any(v <= 0 or not math.isfinite(v) for v in stage_grads.values()):
                raise RuntimeError('Missing expert layer gradient')
            if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
                raise RuntimeError('Frozen weights have gradients')
            audit = {'stage': 'gradient_audit', 'all_16_expert_layers_positive': True,
                     'per_layer_gradient_l1': stage_grads, 'frozen_gradients_absent': True}
            if goal_cfg:
                # With the zero-initialised output layer, only that layer (and the null embedding) receives
                # gradient at step 1; the input layer starts learning once the output weights are non-zero.
                goal_grads = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.0
                              for n, p in parameters if '.goal_embedding.' in f'.{n}'}
                if goal_grads['model.goal_embedding.out.weight'] <= 0:
                    raise RuntimeError('Goal embedding output layer has no gradient')
                if any(p.grad is None or float(p.grad.abs().sum()) <= 0 for _, p in planner_parameters):
                    raise RuntimeError('Missing planner gradient')
                audit.update({'goal_embedding_gradient_l1': goal_grads, 'planner_all_positive': True})
            emit(out, audit)
        optimizer.step()
        record = {'stage': 'smoke' if smoke else 'training', 'step': step, 'steps': cfg['steps'],
                  'loss': float(np.mean(losses)), 'learning_rate': lr, 'gradient_norm': float(norm),
                  'elapsed_seconds': time.perf_counter() - started,
                  'peak_cuda_bytes': torch.cuda.max_memory_allocated()}
        if goal_cfg:
            record.update({'planner_loss': float(np.mean(planner_losses)), 'planner_gradient_norm': float(planner_norm),
                           'planner_terms': {k: float(np.mean([t[k] for t in planner_terms])) for k in planner_terms[0]}})
        if smoke or step % 10 == 0 or step == 1:
            emit(out, record)
        if not smoke:
            (out / 'progress.json').write_text(json.dumps(record, indent=2) + '\n')
        if smoke:
            continue
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            pending[1].result()  # no decoding in flight while evaluation reads the same frame source
            metric = evaluate(step)
            if metric < best:
                best, best_step, stale = metric, step, 0
                save_model('best')
            else:
                stale += 1
            if not cfg.get('save_last', True):
                # Disk-limited runs keep only the selected checkpoint; no resume state.
                if step >= cfg['minimum_steps'] and stale >= cfg['patience_evaluations']:
                    emit(out, {'stage': 'early_stop', 'step': step, 'best_step': best_step})
                    break
                continue
            save_model('last')
            temporary = out / 'training_state.partial.pt'
            torch.save({'step': step, 'optimizer': optimizer.state_dict(), 'rng': pending[0],
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                        'best_metric': best, 'best_step': best_step, 'stale_evaluations': stale, 'config': cfg,
                        'last_weights_sha256': sha(out / 'last/model.safetensors')}, temporary)
            temporary.replace(out / 'training_state.pt')
            if step >= cfg['minimum_steps'] and stale >= cfg['patience_evaluations']:
                emit(out, {'stage': 'early_stop', 'step': step, 'best_step': best_step})
                break
    loader.shutdown(wait=True)
    delta = float((first_param.detach() - initial).norm())
    assert delta > 0 and math.isfinite(delta)
    emit(out, {'stage': 'smoke_completed' if smoke else ('stopped' if last_step < cfg['steps'] else 'completed'), 'steps': step,
               'action_out_weight_delta_l2': delta, 'best_step': best_step, 'best_metric': None if smoke else best,
               'quantization_started': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'train'], required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-step', type=int,
                        help='stop after this step (a multiple of eval_every, so best/last/resume state are saved there); '
                             'the learning-rate schedule still follows the configured total steps')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if args.resume and args.stage != 'train':
        parser.error('--resume is only available for training')
    cfg = json.loads(args.config.read_text())
    if args.stop_step is not None and (args.stage != 'train' or args.stop_step % cfg['eval_every']):
        parser.error('--stop-step needs --stage train and a multiple of eval_every')
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare' and cfg.get('evaluation') == 'scripted_sim':
        from so101vla.scripted import prepare_scripted
        prepare_scripted(cfg, out)
    elif args.stage == 'prepare':
        prepare(cfg, out)
    else:
        train(cfg, out, args.stage == 'smoke', resume=args.resume, stop_step=args.stop_step)


if __name__ == '__main__':
    main()
