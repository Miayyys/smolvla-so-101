"""Merge everything with a known outcome into one dataset for RECAP (docs/notes/recap.md): the demos and human
corrections of so101_pilot_v3, the real policy rollouts that have a success label, and the twin's scripted episodes.

    .venv/bin/python scripts/recap_dataset.py --out so101_recap_v1

Writes, beside the inputs in artifacts/datasets:
  <out>/                           the merged LeRobot dataset (every source gets the ``intervention`` column:
                                   rollouts all False, twin episodes all True, so101_pilot_v3 keeps its own)
  <out>/meta/recap_outcomes.json   one row per merged episode: source, source episode, kind, task, success
  configs/<out>_partition_policy.json   validation = so101_pilot_v3's 11 validation demos; all else with an
                                   outcome in train
  configs/<out>_partition_value.json    validation also holds out VALUE_HOLDOUT of the rollout and twin episodes
                                   (seeded), so the value function is checked on episodes it has not fitted

Kinds: demo (teleoperated, success), correction (policy with human takeovers; success from runs/dagger/*.csv),
rollout (policy only; success from runs/real_eval/<name>.csv), twin (scripted expert in the twin, success). Rollout
episodes without a row in their CSV get success null and go to the "test" split, which nothing trains on.
Tagged copies of the inputs (a full copy each, needed because LeRobot only merges identical features) are kept in
artifacts/datasets/recap_tagged/ and reused.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import merge_corrections

ROOT = Path('artifacts/datasets')
BASE = 'so101_pilot_v3'
BASE_PARTITION = 'configs/so101_pilot_v3_partition.json'
ROLLOUTS = ['rollout_full_v5_key_raised', 'rollout_full_v5_doll_raised', 'rollout_full_v4_doll_raised',
            'rollout_pilot_v3_key', 'rollout_pilot_v3_doll', 'rollout_key_v1_best',
            'rollout_pilot_v2_final_doll_rtc', 'rollout_pilot_v2_final_doll_rtc_s2', 'rollout_pilot_v2_final_key_rtc',
            'rollout_pilot_v2_final_key_rtc_s2', 'rollout_pilot_v2_s6500_doll', 'rollout_pilot_v2_s6500_doll_rtc',
            'rollout_pilot_v2_s6500_key']
TWIN = ['so101_twin_gen_v1']
VALUE_HOLDOUT = 0.15


def csv_outcomes(path):
    return {int(row['episode']): row['success'] == '1' for row in csv.DictReader(open(path))}


def tagged(name, value):
    """A copy of dataset ``name`` with an ``intervention`` column of ``value`` (made once)."""
    from lerobot.datasets.dataset_tools import add_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = ROOT / 'recap_tagged' / name
    if not out.exists():
        partial = out.with_name(name + '.partial')
        if partial.exists():
            raise SystemExit(f'{partial} is left from an interrupted run; remove it')
        source = LeRobotDataset(f'local/{name}', root=ROOT / name)
        add_features(source, {'intervention': (np.full((source.meta.total_frames, 1), value, dtype=bool),
                                               {'dtype': 'bool', 'shape': (1,), 'names': None})},
                     output_dir=partial, repo_id=f'local/{name}')
        merge_corrections.episode_meta_mismatches(partial, fix=True)
        partial.rename(out)
    if merge_corrections.episode_meta_mismatches(out):
        raise SystemExit(f'{out}: meta/episodes rows name the wrong file; remove it and rerun')
    return out


def fixed_copy(name):
    """``name`` with its meta/episodes file locations corrected (merge_corrections.py's docstring): data and videos
    hard-linked, meta copied, so the original is not touched."""
    import shutil

    out = ROOT / 'recap_tagged' / name
    if not out.exists():
        partial = out.with_name(name + '.partial')
        if partial.exists():
            raise SystemExit(f'{partial} is left from an interrupted run; remove it')
        shutil.copytree(ROOT / name, partial, copy_function=os.link, ignore=shutil.ignore_patterns('meta'))
        shutil.copytree(ROOT / name / 'meta', partial / 'meta')
        print(f'{name}: rewrote the metadata file location of {merge_corrections.episode_meta_mismatches(partial, fix=True)} episodes')
        partial.rename(out)
    if merge_corrections.episode_meta_mismatches(out):
        raise SystemExit(f'{out}: meta/episodes rows name the wrong file; remove it and rerun')
    return out


def tasks_of(root):
    import pandas as pd
    episodes = pd.concat(pd.read_parquet(p) for p in sorted(Path(root, 'meta/episodes').rglob('*.parquet')))
    return {int(e): list(t) for e, t in zip(episodes['episode_index'], episodes['tasks'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = ROOT / args.out
    partitions = {k: Path(f'configs/{args.out}_partition_{k}.json') for k in ('policy', 'value')}
    for path in (out, *partitions.values()):
        if path.exists():
            raise SystemExit(f'{path} already exists')

    rows = []  # one per merged episode, in merge order
    base_partition = json.loads(Path(BASE_PARTITION).read_text())
    base_tasks = tasks_of(ROOT / BASE)
    for source in base_partition['sources']:
        first, last = source['episodes']
        name = Path(source['dataset']).name
        outcomes = csv_outcomes(f'runs/dagger/{name}.csv') if 'dagger' in name else None
        for k, episode in enumerate(range(first, last + 1)):
            rows.append({'source': name, 'source_episode': k, 'kind': 'correction' if outcomes else 'demo',
                         'success': outcomes[k] if outcomes else True, 'task': base_tasks[episode][0]})
    inputs = [fixed_copy(BASE)]
    for name in ROLLOUTS:
        outcomes = csv_outcomes(f'runs/real_eval/{name}.csv')
        tasks = tasks_of(ROOT / name)
        inputs.append(tagged(name, False))
        rows += [{'source': name, 'source_episode': e, 'kind': 'rollout', 'success': outcomes.get(e), 'task': tasks[e][0]}
                 for e in sorted(tasks)]
    for name in TWIN:
        tasks = tasks_of(ROOT / name)
        inputs.append(tagged(name, True))
        rows += [{'source': name, 'source_episode': e, 'kind': 'twin', 'success': True, 'task': tasks[e][0]}
                 for e in sorted(tasks)]

    datasets = [LeRobotDataset(f'local/{p.name}', root=p) for p in inputs]
    merged = merge_datasets(datasets, output_repo_id=f'local/{args.out}', output_dir=out)
    if merged.meta.total_episodes != len(rows):
        raise SystemExit(f'merged {merged.meta.total_episodes} episodes, expected {len(rows)}')
    merged_tasks = tasks_of(out)
    for index, row in enumerate(rows):
        row['episode'] = index
        if merged_tasks[index] != [row['task']]:
            raise SystemExit(f'episode {index}: task {merged_tasks[index]} != {row["task"]} ({row["source"]})')
    (out / 'meta/recap_outcomes.json').write_text(json.dumps(rows, indent=1) + '\n')

    validation = set(base_partition['splits']['validation'])
    unknown = {r['episode'] for r in rows if r['success'] is None}
    rng = np.random.default_rng(args.seed)
    candidates = [r['episode'] for r in rows if r['kind'] in ('rollout', 'twin') and r['success'] is not None]
    held_out = set(rng.choice(candidates, int(round(VALUE_HOLDOUT * len(candidates))), replace=False).tolist())
    for kind, extra in (('policy', set()), ('value', held_out)):
        valid = validation | extra
        splits = {'train': [r['episode'] for r in rows if r['episode'] not in valid | unknown],
                  'validation': sorted(valid), 'test': sorted(unknown)}
        partitions[kind].write_text(json.dumps({
            'dataset': f'{out} (scripts/recap_dataset.py)',
            'rule': f'{BASE} validation demos held out' + (f'; plus {len(extra)} rollout/twin episodes (seed {args.seed})'
                                                         if extra else '') + '; episodes without an outcome in test',
            'splits': splits}, indent=2) + '\n')
    summary = {}
    for r in rows:
        key = f"{r['kind']} / {r['task']}"
        s = summary.setdefault(key, {'episodes': 0, 'success': 0, 'unknown': 0})
        s['episodes'] += 1
        s['success'] += r['success'] is True
        s['unknown'] += r['success'] is None
    print(json.dumps({'dataset': str(out), 'episodes': len(rows), 'frames': merged.meta.total_frames,
                      'value_validation_extra': len(held_out), 'by_kind_and_task': summary}, indent=1))


if __name__ == '__main__':
    main()
