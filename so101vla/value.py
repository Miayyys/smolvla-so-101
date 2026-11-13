"""RECAP value function and advantage labels (pi*0.6, arXiv 2511.14759, Sec. III, IV-A, IV-B; docs/notes/recap.md).

    python -m so101vla.value --config configs/so101_value_v1.json --stage prepare   # frame cache (train.prepare)
    python -m so101vla.value --config configs/so101_value_v1.json --stage smoke     # 2 steps
    python -m so101vla.value --config configs/so101_value_v1.json --stage train
    python -m so101vla.value --config configs/so101_value_v1.json --stage label     # -> <output>/labels.npz

Reward (paper Eq. 5): -1 per step, 0 at the end of a successful episode, -C_fail at the end of a failed one; no
discount. The return from frame t of an episode ending at T is therefore -(T - t) (success) or -(T - t) - C_fail
(failure). It is divided by (H + C_fail), H = ``horizon_s`` x fps the longest episode, C_fail = H, so successes lie in
[-0.5, 0] and failures in [-1, -0.5]. The episode end stands in for the moment of success: most real rollouts were
stopped by hand when the object was in the box, but some ran to their time limit (pilot v2/v3 evaluations).

Model: SmolVLA's VLM (images + instruction + state, the policy prefix, weights from ``init_weights``), the final
hidden state of the last prefix token (the state token, which attends to all others) -> LayerNorm -> 201 logits over
bins evenly spaced on [-1, 0]; cross-entropy against the bin of the return (paper: categorical, B = 201). The value is
the expectation over the bins. The action expert is loaded but not used.

Labels (paper Sec. III, IV-B): advantage A_t = (G_t - G_{t+N}) + V(t+N) - V(t) for t + N within the episode (the
first term is the N rewards in between), otherwise G_t - V(t), with G the normalised return. Per task, the threshold
is the (1 - positive_share) quantile of the advantages of that task's training frames; I_t = A_t > threshold. Frames
a human drove in correction episodes are always positive (the paper's interventions); the deliberate misses of
twin episodes (``intervention`` False there, scripts/recap_extend.py) are always negative and left out of the
thresholds. Every dataset row gets a
label (rows outside the training split too; train.py only samples the training split).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from so101vla.train import Frames, emit, prepare, prepared_identity, sha

BINS = 201
CENTRES = np.linspace(-1.0, 0.0, BINS)


def outcomes_of(cfg):
    path = Path(cfg.get('outcomes') or Path(cfg['dataset_root']) / 'meta/recap_outcomes.json')
    return {row['episode']: row for row in json.loads(path.read_text())}


def returns(cfg, data, outcomes, fps=30):
    """Normalised return G per row (NaN where the episode's outcome is unknown) and the per-step reward scale."""
    horizon = cfg['value']['horizon_s'] * fps
    fail = horizon
    scale = horizon + fail
    eps = data['episodes']
    out = np.full(len(eps), np.nan)
    for ep in np.unique(eps):
        rows = np.flatnonzero(eps == ep)
        success = outcomes[int(ep)]['success']
        if success is None:
            continue
        to_go = rows[-1] - rows
        out[rows] = -(to_go + (0 if success else fail)) / scale
    if np.nanmin(out) < -1:
        raise ValueError(f'an episode is longer than horizon_s ({cfg["value"]["horizon_s"]} s)')
    return out, 1.0 / scale


class ValueModel:
    """Wraps a SmolVLA policy (prefix encoder) and a bin head; ``logits(batch)`` takes a preprocessed batch."""

    def __init__(self, policy):
        import torch
        self.policy = policy
        hidden = policy.model.vlm_with_expert.config.text_config.hidden_size
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, BINS)).to('cuda')
        self.centres = torch.tensor(CENTRES, dtype=torch.float32, device='cuda')

    def logits(self, batch):
        import torch
        from lerobot.policies.common.vla_utils import make_att_2d_masks
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        policy, model = self.policy, self.policy.model
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        embs, pad, att = model.embed_prefix(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                            batch[OBS_LANGUAGE_ATTENTION_MASK], state=state)
        # use_cache=True: every VLM layer runs self-attention over the prefix (as at inference); without it the
        # cross-attention layers would expect expert tokens.
        (out, _), _ = model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad, att),
                                                    position_ids=torch.cumsum(pad, dim=1) - 1, past_key_values=None,
                                                    inputs_embeds=[embs, None], use_cache=True)
        return self.head(out[:, -1].float())

    def value(self, logits):
        return (logits.softmax(-1) * self.centres).sum(-1)

    def state_dict(self):
        return {**{f'policy.{k}': v for k, v in self.policy.state_dict().items()},
                **{f'head.{k}': v for k, v in self.head.state_dict().items()}}

    def load_state_dict(self, weights):
        self.policy.load_state_dict({k[7:]: v for k, v in weights.items() if k.startswith('policy.')}, strict=True)
        self.head.load_state_dict({k[5:]: v for k, v in weights.items() if k.startswith('head.')}, strict=True)


def build(cfg, out):
    """Value model, processors and the prepared frames."""
    import torch
    from lerobot.policies import make_pre_post_processors
    from so101vla.policy import load_policy

    prepared = json.loads((out / 'prepared.json').read_text())
    if prepared_identity(prepared['config']) != prepared_identity(cfg) or prepared['partition_sha256'] != sha(cfg['partition']):
        raise ValueError('Prepared data/config identity differs')
    stats = json.loads((out / 'stats.json').read_text())
    policy, _, _ = load_policy(SimpleNamespace(model_dir=Path(cfg['model_dir']).resolve(),
                                               vlm_assets_dir=Path(cfg['vlm_assets_dir']).resolve(), device='cuda'))
    policy.float()
    policy.config.resize_imgs_with_padding = (cfg['image_size'], cfg['image_size'])
    cameras = list(cfg['camera_map'])
    policy.config.input_features = {k: v for k, v in policy.config.input_features.items() if k in ['observation.state', *cameras]}
    policy.config.empty_cameras = 0
    if cfg.get('init_weights'):
        from safetensors.torch import load_file
        policy.load_state_dict(load_file(cfg['init_weights'], device='cuda'), strict=True)
    pre, _ = make_pre_post_processors(policy.config, cfg['model_dir'], preprocessor_overrides={
        'tokenizer_processor': {'tokenizer_name': str(Path(cfg['vlm_assets_dir']).resolve())},
        'device_processor': {'device': 'cuda'}, 'normalizer_processor': {'stats': stats}},
        postprocessor_overrides={'unnormalizer_processor': {'stats': stats}})
    for p in policy.parameters():
        p.requires_grad = False
    vlm = policy.model.vlm_with_expert.vlm
    trained = [p for n, p in vlm.named_parameters()
               if not n.startswith('lm_head') and (cfg['value'].get('train_vision', True) or '.vision_model.' not in n)]
    for p in trained:
        p.requires_grad = True
    model = ValueModel(policy)
    return model, pre, Frames(out, cfg), stats


def inputs(frames, rows):
    raw = frames.batch(rows)
    for key in ('action', 'action_is_pad', 'advantage'):
        raw.pop(key, None)
    return raw


def train(cfg, out, smoke):
    import torch
    from safetensors.torch import save_file

    vcfg = cfg['value']
    if (out / 'progress.json').exists() and not smoke:
        raise ValueError('Training already exists; preserve it instead of overwriting')
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    model, pre, frames, _ = build(cfg, out)
    target, _ = returns(cfg, frames.data, outcomes_of(cfg))
    train_rows = frames.data['train_rows'][~np.isnan(target[frames.data['train_rows']])]
    validation_rows = np.asarray(frames.data['validation_rows'])
    validation_rows = validation_rows[~np.isnan(target[validation_rows])]
    bins = np.clip(np.rint((target + 1) * (BINS - 1)), 0, BINS - 1)
    parameters = [p for p in model.policy.parameters() if p.requires_grad] + list(model.head.parameters())
    emit(out, {'stage': 'value_setup', 'trained_parameters': sum(p.numel() for p in parameters),
               'train_rows': len(train_rows), 'validation_rows': len(validation_rows),
               'train_target_mean': float(target[train_rows].mean()),
               'train_failure_share': float((target[train_rows] < -0.5).mean())})
    optimizer = torch.optim.AdamW([{'params': [p for p in model.policy.parameters() if p.requires_grad], 'lr_scale': 1.0},
                                   {'params': list(model.head.parameters()), 'lr_scale': vcfg.get('head_lr_scale', 10.0)}],
                                  lr=cfg['learning_rate'], betas=(0.9, 0.95), weight_decay=cfg['weight_decay'])
    kinds = {e: r['kind'] + ('' if r['success'] else ' failed') for e, r in outcomes_of(cfg).items()}

    def evaluate(step):
        model.policy.eval(); model.head.eval()
        predictions, losses = [], []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, len(validation_rows), cfg['batch_size']):
                rows = validation_rows[start:start + cfg['batch_size']]
                logits = model.logits(pre(inputs(frames, rows)))
                losses.append(torch.nn.functional.cross_entropy(logits, torch.from_numpy(bins[rows]).long().cuda(), reduction='sum').item())
                predictions.append(model.value(logits).float().cpu().numpy())
        predictions = np.concatenate(predictions)
        error = np.abs(predictions - target[validation_rows])
        groups = np.array([kinds[int(e)] for e in frames.data['episodes'][validation_rows]])
        result = {'stage': 'validation', 'step': step, 'cross_entropy': float(np.sum(losses) / len(validation_rows)),
                  'value_mae': float(error.mean()),
                  'value_mae_by_kind': {g: float(error[groups == g].mean()) for g in sorted(set(groups))},
                  'success_vs_failure_accuracy': float(((predictions < -0.5) == (target[validation_rows] < -0.5)).mean())}
        emit(out, result)
        model.policy.train(); model.head.train()
        return result['cross_entropy']

    def save(name):
        folder = out / name
        folder.mkdir(exist_ok=True)
        temporary = folder / 'value.partial.safetensors'
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if v.dtype.is_floating_point}, str(temporary))
        temporary.replace(folder / 'value.safetensors')

    best = float('inf') if smoke else evaluate(0)
    best_step, stale = 0, 0
    started = time.perf_counter()
    model.policy.train(); model.head.train()
    last = 2 if smoke else cfg['steps']
    for step in range(1, last + 1):
        progress = min(1.0, max(0.0, (step - cfg['warmup_steps']) / (cfg['steps'] - cfg['warmup_steps'])))
        lr = cfg['minimum_learning_rate'] + 0.5 * (cfg['learning_rate'] - cfg['minimum_learning_rate']) * (1 + math.cos(math.pi * progress))
        if step <= cfg['warmup_steps']:
            lr = cfg['learning_rate'] * step / cfg['warmup_steps']
        for group in optimizer.param_groups:
            group['lr'] = lr * group['lr_scale']
        optimizer.zero_grad(set_to_none=True)
        losses = []
        accumulation = 1 if smoke else cfg['gradient_accumulation']
        for _ in range(accumulation):
            rows = rng.choice(train_rows, cfg['batch_size'], replace=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model.logits(pre(inputs(frames, rows)))
            loss = torch.nn.functional.cross_entropy(logits.float(), torch.from_numpy(bins[rows]).long().cuda())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'])
        optimizer.step()
        if smoke or step % 10 == 0 or step == 1:
            emit(out, {'stage': 'smoke' if smoke else 'training', 'step': step, 'loss': float(np.mean(losses)), 'learning_rate': lr,
                       'gradient_norm': float(norm), 'elapsed_seconds': time.perf_counter() - started,
                       'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
        if smoke:
            continue
        (out / 'progress.json').write_text(json.dumps({'step': step, 'loss': float(np.mean(losses))}) + '\n')
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            metric = evaluate(step)
            if metric < best:
                best, best_step, stale = metric, step, 0
                save('best')
            else:
                stale += 1
            if stale >= cfg['patience_evaluations']:
                emit(out, {'stage': 'early_stop', 'step': step, 'best_step': best_step})
                break
    emit(out, {'stage': 'smoke_completed' if smoke else 'completed', 'best_step': best_step, 'best_cross_entropy': best})


def reuse_value_prefix(source, out, cfg, data):
    """Reuse predictions only after proving identical weights, normalisation, rows and cached RGB inputs.

    This reuses V(s), never the old advantages/indicators: thresholds must be recomputed on the enlarged dataset.
    The original cache files need not still exist; their recorded SHA256 is checked against the new cache prefix.
    """
    import hashlib
    from so101vla.train import cache_path

    source, out = Path(source), Path(out)
    old_prepared = json.loads((source / 'prepared.json').read_text())
    old_cfg = old_prepared['config']
    for key in ('model_dir', 'vlm_assets_dir', 'image_size', 'cache_resize', 'camera_map'):
        if old_cfg.get(key) != cfg.get(key):
            raise ValueError(f'Cannot reuse values: {key} changed')
    if sha(source / 'stats.json') != sha(out / 'stats.json'):
        raise ValueError('Cannot reuse values: normalisation changed')
    old_checkpoint, checkpoint = source / 'best/value.safetensors', out / 'best/value.safetensors'
    if old_checkpoint.resolve() != checkpoint.resolve() and sha(old_checkpoint) != sha(checkpoint):
        raise ValueError('Cannot reuse values: value weights changed')
    if old_checkpoint.stat().st_mtime > (source / 'labels.npz').stat().st_mtime:
        raise ValueError('Cannot reuse values: weights were modified after the source predictions')
    labels_sha = sha(source / 'labels.npz')
    if labels_sha != json.loads((source / 'labels.json').read_text())['labels_sha256']:
        raise ValueError('Cannot reuse values: source predictions checksum differs')
    with np.load(source / 'labels.npz') as labels:
        values = labels['value'].copy()
    count = len(values)
    if values.ndim != 1 or count == 0 or count > len(data['episodes']) or not np.isfinite(values).all():
        raise ValueError('Cannot reuse values: invalid source prediction shape or values')
    with np.load(source / 'data.npz') as old:
        if len(old['episodes']) != count:
            raise ValueError('Cannot reuse values: source rows and predictions differ')
        for key in ('states', 'actions', 'episodes', 'task_index'):
            if not np.array_equal(old[key], data[key][:count]):
                raise ValueError(f'Cannot reuse values: {key} prefix changed')
        if not np.array_equal(old['task_names'], data['task_names']):
            raise ValueError('Cannot reuse values: task text changed')
        for camera, record in old_prepared['caches'].items():
            prefix = camera.split('.')[-1]
            for name in ('offsets', 'lengths'):
                key = f'{prefix}_{name}'
                if not np.array_equal(old[key], data[key][:count]):
                    raise ValueError(f'Cannot reuse values: {key} changed')
            length = int(data[f'{prefix}_offsets'][count - 1] + data[f'{prefix}_lengths'][count - 1])
            if length != record['bytes']:
                raise ValueError(f'Cannot reuse values: {camera} prefix size changed')
            h = hashlib.sha256()
            with cache_path(cfg, out, camera).open('rb') as stream:
                remaining = length
                while remaining:
                    block = stream.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError(f'Cannot reuse values: {camera} cache is truncated')
                    h.update(block)
                    remaining -= len(block)
            if h.hexdigest() != record['sha256']:
                raise ValueError(f'Cannot reuse values: {camera} cached image bytes changed')
    old_outcomes, new_outcomes = outcomes_of(old_cfg), outcomes_of(cfg)
    if any(new_outcomes.get(ep) != row for ep, row in old_outcomes.items()):
        raise ValueError('Cannot reuse values: source episode outcomes changed')
    return values, {'source': str(source), 'source_labels_sha256': labels_sha, 'reused_value_rows': count,
                    'new_value_rows': len(data['episodes']) - count, 'cache_prefix_sha256_verified': True}


def label(cfg, out, reuse_values_from=None):
    """Value of every row with the best checkpoint, then advantages and indicators (module docstring)."""
    import torch
    from safetensors.torch import load_file

    vcfg = cfg['value']
    model, pre, frames, _ = build(cfg, out)
    model.load_state_dict(load_file(str(out / 'best/value.safetensors'), device='cuda'))
    model.policy.eval(); model.head.eval()
    data = frames.data
    outcomes = outcomes_of(cfg)
    target, _ = returns(cfg, data, outcomes)
    values = np.zeros(len(target), dtype=np.float32)
    reused = 0
    reuse_info = {}
    if reuse_values_from is not None:
        prefix, reuse_info = reuse_value_prefix(reuse_values_from, out, cfg, data)
        reused = len(prefix)
        values[:reused] = prefix
        emit(out, {'stage': 'value_predictions_reused', **reuse_info})
    started = time.perf_counter()
    batch = vcfg.get('label_batch_size', 32)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(reused, len(values), batch):
            rows = np.arange(start, min(start + batch, len(values)))
            values[rows] = model.value(model.logits(pre(inputs(frames, rows)))).float().cpu().numpy()
            if (start - reused) // batch % 200 == 0:
                emit(out, {'stage': 'labelling', 'rows': int(rows[-1]) + 1, 'total': len(values),
                           'inferred_rows': int(rows[-1]) + 1 - reused, 'reused_rows': reused,
                           'elapsed_seconds': time.perf_counter() - started})
    n = vcfg['advantage_steps']
    eps = data['episodes']
    advantage = np.full(len(values), np.nan, dtype=np.float32)
    for ep in np.unique(eps):
        rows = np.flatnonzero(eps == ep)
        if outcomes[int(ep)]['success'] is None:
            continue
        g, v = target[rows], values[rows]
        a = g - v  # t + N beyond the end: the Monte Carlo return
        if n < len(rows):
            a[:-n] = (g[:-n] - g[n:]) + v[n:] - v[:-n]
        advantage[rows] = a
    task_names = data['task_names'][data['task_index']]
    train_mask = np.zeros(len(values), dtype=bool)
    split = json.loads(Path(cfg['partition']).read_text())['splits']
    train_mask[np.isin(eps, split['train'])] = True
    human_correction = np.zeros(len(values), dtype=bool)
    import pandas as pd
    table = pd.concat([pd.read_parquet(p) for p in sorted((Path(cfg['dataset_root']) / 'data').rglob('*.parquet'))]).sort_values('index')
    human = (np.stack(table['intervention'].to_numpy()).reshape(-1).astype(bool) if 'intervention' in table
             else np.zeros(len(values), dtype=bool))
    twin_miss = np.zeros(len(values), dtype=bool)  # deliberate misses of twin data v2 (scripts/recap_extend.py)
    for ep in np.unique(eps):
        rows = np.flatnonzero(eps == ep)
        if outcomes[int(ep)]['kind'] == 'correction':
            human_correction[rows] = human[rows]
        elif outcomes[int(ep)]['kind'] == 'twin':
            twin_miss[rows] = ~human[rows]
    indicator = np.zeros(len(values), dtype=np.int8)
    thresholds = {}
    for task in np.unique(task_names):
        mask = (task_names == task) & ~np.isnan(advantage)
        threshold = float(np.quantile(advantage[mask & train_mask & ~twin_miss], 1 - vcfg['positive_share']))
        thresholds[str(task)] = threshold
        indicator[mask] = advantage[mask] > threshold
    indicator[human_correction] = 1
    indicator[twin_miss] = 0
    path = out / 'labels.npz'
    np.savez(path, indicator=indicator, value=values, advantage=advantage, target=target.astype(np.float32),
             human_correction=human_correction, twin_miss=twin_miss)
    kinds = np.array([outcomes[int(e)]['kind'] + ('' if outcomes[int(e)]['success'] else ' failed') for e in eps])
    summary = {'stage': 'labelled', 'labels': str(path), 'labels_sha256': sha(path), 'thresholds': thresholds,
               **reuse_info,
               'advantage_steps': n, 'positive_share_target': vcfg['positive_share'],
               'positive_share_by_kind': {k: float(indicator[(kinds == k) & train_mask].mean()) for k in sorted(set(kinds))},
               'human_correction_frames': int(human_correction.sum()), 'twin_miss_frames': int(twin_miss.sum()),
               'value_mae_train': float(np.nanmean(np.abs(values - target)[train_mask])),
               'value_mae_validation': float(np.nanmean(np.abs(values - target)[np.isin(eps, split['validation'])]))}
    (out / 'labels.json').write_text(json.dumps(summary, indent=2) + '\n')
    emit(out, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'train', 'label'], required=True)
    parser.add_argument('--reuse-values-from', type=Path,
                        help='label only: verified prior run whose value predictions form a prefix of this dataset')
    args = parser.parse_args()
    if args.reuse_values_from is not None and args.stage != 'label':
        parser.error('--reuse-values-from requires --stage label')
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    cfg = json.loads(args.config.read_text())
    if not cfg.get('include_policy_frames'):
        raise ValueError('the value function needs include_policy_frames (failed rollouts are its negatives)')
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        prepare(cfg, out)
    elif args.stage == 'label':
        label(cfg, out, args.reuse_values_from)
    else:
        train(cfg, out, args.stage == 'smoke')


if __name__ == '__main__':
    main()
