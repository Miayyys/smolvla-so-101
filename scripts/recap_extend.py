"""Append new twin episodes to a merged RECAP dataset (scripts/recap_dataset.py) without rebuilding it, so only the
new episodes have to be uploaded to the training server (docs/notes/recap.md).

Twin data v2 (scripts/twin_night_v2.sh, twin_topup_v2.sh) has deliberate misses before the real grasp (twin_expert
plan_miss); those frames must not count as expert frames. Two steps:

    # where runs/twin/generated_* is (the local machine): per-episode miss ends beside the converted dataset
    .venv/bin/python scripts/recap_extend.py sidecar --dataset so101_twin_gen_v2_512 --generated runs/twin/generated_v2

    # anywhere (the server): tag, merge, outcomes and partitions
    .venv/bin/python scripts/recap_extend.py merge --base so101_recap_v1_512 --base-name so101_recap_v1 \
        --twin so101_twin_gen_v2_512 so101_twin_gen_v2b_512 --name so101_recap_v2 --out-dir /root/sj-fs/datasets/so101_recap_v2_512

``sidecar`` writes <dataset>/meta/twin_expert.json: per dataset episode its generated directory, frame count and
``miss_end`` (frames before it are the deliberate miss; 0 = none), after checking that each episode's actions equal
frames.npz. ``merge`` tags each twin dataset with ``intervention`` = expert (True except the miss frames), merges it
after the base (base episode indices and rows are unchanged), appends kind ``twin`` rows (with ``miss_frames``) to
meta/recap_outcomes.json and writes configs/<name>_partition_{policy,value}.json: the base partitions with the new
episodes in train. so101vla/value.py labels the miss frames negative.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import merge_corrections

ROOT = Path('artifacts/datasets')


def episode_table(root):
    return pd.concat(pd.read_parquet(p) for p in sorted(Path(root, 'meta/episodes').rglob('*.parquet'))).sort_values('episode_index')


def data_table(root):
    return pd.concat(pd.read_parquet(p) for p in sorted(Path(root, 'data').rglob('*.parquet'))).sort_values('index')


def sidecar(args):
    root = ROOT / args.dataset
    episodes = sorted(p for p in Path(args.generated).glob('ep_*') if (p / 'meta.json').exists())
    table = episode_table(root)
    data = data_table(root)
    if len(table) != len(episodes):
        raise SystemExit(f'{root}: {len(table)} episodes, {args.generated}: {len(episodes)}')
    rows = []
    for (_, ep), directory in zip(table.iterrows(), episodes):
        frames = np.load(directory / 'frames.npz')
        actions = np.stack(data['action'][data['episode_index'] == ep['episode_index']].to_numpy())
        if len(actions) != len(frames['action']) or not np.allclose(actions, frames['action'], atol=1e-4):
            raise SystemExit(f'episode {ep["episode_index"]}: actions differ from {directory}/frames.npz')
        expert = frames['expert'] if 'expert' in frames else np.ones(len(actions), dtype=bool)
        miss_end = int(np.argmax(expert)) if expert.any() else len(expert)
        if not expert[miss_end:].all():
            raise SystemExit(f'{directory}: expert frames are not one block at the end')
        rows.append({'episode': int(ep['episode_index']), 'generated': directory.name, 'frames': len(actions), 'miss_end': miss_end})
    (root / 'meta/twin_expert.json').write_text(json.dumps(rows, indent=1) + '\n')
    print(f'{root}: {len(rows)} episodes, {sum(r["miss_end"] > 0 for r in rows)} with a deliberate miss '
          f'({sum(r["miss_end"] for r in rows)} frames)')


def tagged(name):
    """A copy of twin dataset ``name`` with ``intervention`` = expert per frame (made once, then reused)."""
    from lerobot.datasets.dataset_tools import add_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rows = json.loads((ROOT / name / 'meta/twin_expert.json').read_text())
    out = ROOT / 'recap_tagged' / name
    if not out.exists():
        partial = out.with_name(name + '.partial')
        if partial.exists():
            raise SystemExit(f'{partial} is left from an interrupted run; remove it')
        source = LeRobotDataset(f'local/{name}', root=ROOT / name)
        lengths = episode_table(ROOT / name)['length'].tolist()
        if lengths != [r['frames'] for r in rows]:
            raise SystemExit(f'{name}: meta/twin_expert.json does not match the episode lengths')
        expert = np.concatenate([np.arange(r['frames']) >= r['miss_end'] for r in rows])[:, None]
        add_features(source, {'intervention': (expert, {'dtype': 'bool', 'shape': (1,), 'names': None})},
                     output_dir=partial, repo_id=f'local/{name}')
        merge_corrections.episode_meta_mismatches(partial, fix=True)
        partial.rename(out)
    if merge_corrections.episode_meta_mismatches(out):
        raise SystemExit(f'{out}: meta/episodes rows name the wrong file; remove it and rerun')
    return out, rows


def fixed_base(base):
    """``base`` with its meta/episodes file locations corrected (scripts/merge_corrections.py; LeRobot's merge writes
    them wrong): data and videos hard-linked beside the real directory, meta copied, the original untouched."""
    import shutil

    if not merge_corrections.episode_meta_mismatches(base):
        return base
    real = base.resolve()
    out = real.with_name(real.name + '.fixed')
    if not out.exists():
        partial = real.with_name(real.name + '.fixed.partial')
        if partial.exists():
            raise SystemExit(f'{partial} is left from an interrupted run; remove it')
        shutil.copytree(real, partial, copy_function=os.link, ignore=shutil.ignore_patterns('meta'))
        shutil.copytree(real / 'meta', partial / 'meta')
        print(f'{base}: rewrote the metadata file location of {merge_corrections.episode_meta_mismatches(partial, fix=True)} episodes')
        partial.rename(out)
    if merge_corrections.episode_meta_mismatches(out):
        raise SystemExit(f'{out}: meta/episodes rows name the wrong file; remove it and rerun')
    return out


def merge(args):
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    base = ROOT / args.base
    out = Path(args.out_dir) if args.out_dir else ROOT / f'{args.name}_512'
    partitions = {k: Path(f'configs/{args.name}_partition_{k}.json') for k in ('policy', 'value')}
    for path in (out, *partitions.values()):
        if path.exists():
            raise SystemExit(f'{path} already exists')
    base = fixed_base(base)
    rows = json.loads((base / 'meta/recap_outcomes.json').read_text())
    inputs = [base]
    for name in args.twin:
        path, expert = tagged(name)
        tasks = episode_table(ROOT / name)['tasks'].tolist()
        inputs.append(path)
        rows += [{'source': name, 'source_episode': r['episode'], 'kind': 'twin', 'success': True, 'task': list(t)[0],
                  'miss_frames': r['miss_end']} for r, t in zip(expert, tasks)]
    merged = merge_datasets([LeRobotDataset(f'local/{p.name}', root=p) for p in inputs],
                            output_repo_id=f'local/{args.name}', output_dir=out)
    if merged.meta.total_episodes != len(rows):
        raise SystemExit(f'merged {merged.meta.total_episodes} episodes, expected {len(rows)}')
    merged_tasks = {int(e): list(t) for e, t in zip(*episode_table(out)[['episode_index', 'tasks']].to_numpy().T)}
    for index, row in enumerate(rows):
        if row.get('episode', index) != index or merged_tasks[index] != [row['task']]:
            raise SystemExit(f'episode {index}: {merged_tasks[index]} / {row}')
        row['episode'] = index
    # the base rows must be where they were: same episode indices and the same table rows
    old, new = data_table(base), data_table(out)
    for column in ('action', 'observation.state', 'episode_index', 'intervention'):
        if not np.array_equal(np.stack(old[column].to_numpy()), np.stack(new[column].to_numpy()[:len(old)])):
            raise SystemExit(f'{column} of the base rows changed in the merge')
    (out / 'meta/recap_outcomes.json').write_text(json.dumps(rows, indent=1) + '\n')
    added = [r['episode'] for r in rows[len(json.loads((base / 'meta/recap_outcomes.json').read_text())):]]
    for kind, path in partitions.items():
        partition = json.loads(Path(f'configs/{args.base_name}_partition_{kind}.json').read_text())
        partition['splits']['train'] += added
        partition['dataset'] = f'{out} (scripts/recap_extend.py)'
        partition['rule'] += f'; plus {len(added)} twin episodes of {", ".join(args.twin)} in train'
        path.write_text(json.dumps(partition, indent=2) + '\n')
    print(json.dumps({'dataset': str(out), 'episodes': len(rows), 'added': len(added), 'frames': merged.meta.total_frames,
                      'miss_frames': int(sum(r.get('miss_frames', 0) for r in rows)),
                      'partitions': [str(p) for p in partitions.values()]}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('sidecar')
    s.add_argument('--dataset', required=True)
    s.add_argument('--generated', required=True)
    m = sub.add_parser('merge')
    m.add_argument('--base', required=True, help='merged dataset in artifacts/datasets (with meta/recap_outcomes.json)')
    m.add_argument('--base-name', required=True, help='configs/<base-name>_partition_{policy,value}.json')
    m.add_argument('--twin', nargs='+', required=True)
    m.add_argument('--name', required=True)
    m.add_argument('--out-dir')
    args = parser.parse_args()
    (sidecar if args.command == 'sidecar' else merge)(args)


if __name__ == '__main__':
    main()
