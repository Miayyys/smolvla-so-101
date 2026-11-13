"""Real/twin-balanced RECAP sampling and verified reuse of an existing image cache."""

import copy
import json
from pathlib import Path

import numpy as np


class BalancedSampler:
    """Exactly balance each microbatch; only human correction rows receive extra within-real weight."""

    def __init__(self, data, cfg):
        sampling = cfg['sampling']
        if sampling['real_fraction'] != .5:
            raise ValueError('This sampler implements exactly 50:50 real/twin microbatches')
        self.rows = np.asarray(data['train_rows'])
        if len(np.unique(self.rows)) != len(self.rows):
            raise ValueError('Balanced sampling requires correction_repeat=1 and unique train rows')
        group = data['sampling_group'][self.rows]
        if not np.isin(group, [0, 1]).all():
            raise ValueError('Unknown sampling group')
        self.real, self.twin = self.rows[group == 0], self.rows[group == 1]
        if not len(self.real) or not len(self.twin):
            raise ValueError('Both real and twin training groups must be nonempty')
        boost = sampling['human_correction_weight']
        if not np.isfinite(boost) or boost < 1:
            raise ValueError('Correction weight must be finite and at least 1')
        weight = np.where(data['human_correction'][self.real], boost, 1.)
        self.real_probability = weight / weight.sum()

    def draw(self, rng, batch_size):
        if batch_size % 2:
            raise ValueError('Balanced microbatches require an even batch size')
        half = batch_size // 2
        rows = np.concatenate([rng.choice(self.real, half, p=self.real_probability),
                               rng.choice(self.twin, half)])
        rng.shuffle(rows)
        return rows


def prepare_cached(cfg, out):
    """Reuse validated images and eligible rows, allowing a subset of training episodes to become held-out.

    Pixel-affecting settings, data hashes, labels and cache hashes must match. Existing validation remains held-out;
    new validation may contain only clean successful twin expert episodes. Nothing in the source run is rewritten.
    """
    from so101vla.train import cache_path, emit, sha

    out, source = Path(out), Path(cfg['cache_from'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'prepared.json').exists():
        raise ValueError('Prepared output already exists')
    old = json.loads((source / 'prepared.json').read_text())
    for key in ('dataset_root', 'camera_map', 'image_size', 'cache_resize', 'trim_static_head',
                'trim_static_tail', 'stats_from', 'correction_repeat', 'advantage'):
        if old['config'].get(key) != cfg.get(key):
            raise ValueError(f'Cached preparation is incompatible: {key}')
    if cfg.get('correction_repeat', 1) != 1:
        raise ValueError('Use human_correction_weight instead of correction_repeat')
    root = Path(cfg['dataset_root'])
    actual_files = {str(p.relative_to(root)) for p in (root / 'data').rglob('*.parquet')}
    if actual_files != set(old['data_sha256']):
        raise ValueError('Dataset parquet file set changed')
    for name, checksum in old['data_sha256'].items():
        if sha(root / name) != checksum:
            raise ValueError(f'Dataset changed: {name}')
    if sha(cfg['advantage']['labels']) != old['advantage']['labels_sha256']:
        raise ValueError('Advantage labels changed')
    if sha(source / 'stats.json') != old['stats_sha256'] or sha(cfg['stats_from']) != old['stats_sha256']:
        raise ValueError('Normalization statistics changed')
    for camera, record in old['caches'].items():
        if cache_path(cfg, out, camera).resolve() != cache_path(old['config'], source, camera).resolve():
            raise ValueError('Cached preparation requires sharing the existing cache files')
        if sha(cache_path(cfg, out, camera)) != record['sha256']:
            raise ValueError(f'Cache checksum changed: {camera}')
    with np.load(source / 'data.npz') as archive:
        data = dict(archive)
    outcomes = json.loads((root / 'meta/recap_outcomes.json').read_text())
    kinds = {r['episode']: r for r in outcomes}
    partition = json.loads(Path(cfg['partition']).read_text())
    split = partition['splits']
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if set(split[a]) & set(split[b]):
            raise ValueError('Partition overlap')
    if set.union(*(set(split[k]) for k in ['train', 'validation', 'test'])) != set(data['episodes']):
        raise ValueError('Partition does not cover all episodes')
    if not set(split['train']) <= set(old['splits']['train']):
        raise ValueError('Cached preparation cannot add new train episodes')
    if not set(old['splits']['validation']) <= set(split['validation']) or split['test'] != old['splits']['test']:
        raise ValueError('Existing held-out splits must be preserved')
    added_validation = set(split['validation']) - set(old['splits']['validation'])
    for ep in added_validation:
        row = kinds[ep]
        if row['kind'] != 'twin' or not row['success'] or row.get('miss_frames', 0):
            raise ValueError('New validation must contain clean successful twin expert episodes')
    train = data['train_rows'][np.isin(data['episodes'][data['train_rows']], split['train'])]
    validation = list(data['validation_rows'])
    for ep in sorted(added_validation):
        rows = np.flatnonzero(data['episodes'] == ep)
        moving = np.flatnonzero((np.abs(data['states'][rows] - data['states'][rows[0]]) >
                                cfg['trim_static_head']['tolerance_deg']).any(1))
        begin = rows[0] + max(0, (moving[0] if len(moving) else len(rows)) - cfg['trim_static_head']['keep_frames'])
        end = int(data['end_rows'][rows[0]])
        validation.extend(np.linspace(begin, max(begin, end - 49), cfg['validation_frames_per_episode'], dtype=int))
    with np.load(cfg['advantage']['labels']) as labels:
        human = labels['human_correction'].copy()
        miss = labels['twin_miss'].copy()
    groups = np.array([int(kinds[int(ep)]['kind'] == 'twin') for ep in data['episodes']], dtype='int8')
    if (human & (groups != 0)).any():
        raise ValueError('Twin frames cannot count as human corrections')
    data.update(train_rows=train, validation_rows=np.asarray(validation, dtype='int64'),
                sampling_group=groups, human_correction=human)
    sampler = BalancedSampler(data, cfg)
    np.savez(out / 'data.npz', **data)
    (out / 'stats.json').write_text((source / 'stats.json').read_text())
    manifest = copy.deepcopy(old)
    manifest.update(config=cfg, splits=split, partition_sha256=sha(cfg['partition']), train_frames=len(train),
                    sampled_rows=len(train), validation_rows=[int(r) for r in validation],
                    train_frames_by_task={str(t): int((data['task_names'][data['task_index'][train]] == t).sum())
                                          for t in data['task_names']},
                    correction_train_frames=int(human[train].sum()), correction_repeat=1,
                    cache_reused_from=str(source), prepared_source_sha256=sha(source / 'prepared.json'),
                    outcomes_sha256=sha(root / 'meta/recap_outcomes.json'))
    audit_rng = np.random.default_rng(20261112)
    drawn = np.concatenate([sampler.draw(audit_rng, cfg['batch_size']) for _ in range(10000)])
    real = drawn[groups[drawn] == 0]
    correction_count = int(human[sampler.real].sum())
    observed_correction = float(human[real].mean())
    expected_correction = float(sampler.real_probability[human[sampler.real]].sum())
    if np.mean(groups[drawn] == 0) != .5 or abs(observed_correction - expected_correction) > .015:
        raise ValueError('Sampling distribution audit failed')
    manifest['sampling_audit'] = {'real_train_frames': len(sampler.real), 'twin_train_frames': len(sampler.twin),
        'human_correction_frames': correction_count, 'draws': len(drawn), 'real_share': .5,
        'correction_share_within_real': observed_correction,
        'expected_correction_share_within_real': expected_correction,
        'human_correction_weight': cfg['sampling']['human_correction_weight']}
    intervention = np.array([kinds[int(ep)]['kind'] in ('demo', 'twin') for ep in data['episodes']])
    intervention = (intervention & ~miss) | human
    eligible = np.zeros(len(data['episodes']), dtype=bool)
    eligible[train] = True
    manifest['trimmed_train_frames'] = int((np.isin(data['episodes'], split['train']) & ~eligible & intervention).sum())
    manifest['advantage']['policy_frames_sampled'] = int((~intervention[drawn]).sum())
    manifest['sampling_audit']['task_share'] = {
        str(t): float((data['task_names'][data['task_index'][drawn]] == t).mean()) for t in data['task_names']}
    manifest['advantage']['positive_share_sampled'] = float(data['indicator'][drawn].mean())
    manifest['advantage']['sampling_share_method'] = 'balanced sampler audit, 160000 draws'
    (out / 'prepared.json').write_text(json.dumps(manifest, indent=2) + '\n')
    emit(out, {'stage': 'prepared', 'cache_reused': True, 'train_frames': len(train),
               'validation_rows': len(validation), 'sampling_audit': manifest['sampling_audit']})
