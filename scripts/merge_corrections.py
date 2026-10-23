"""Merge teleoperated demos with human-correction rollouts (scripts/dagger.fish) into one training dataset + partition.

  uv run --frozen python scripts/merge_corrections.py --demo artifacts/datasets/so101_pilot_v2 \\
      --demo-partition configs/so101_pilot_v2_partition.json --out so101_dagger_v1 \\
      --partition-out configs/so101_dagger_v1_partition.json artifacts/datasets/rollout_dagger_doll_r1 [...]

LeRobot only merges datasets with identical features, so the demos first get an all-True ``intervention`` column
(written once to ``<demo>_intervention`` beside the demo; a full copy). Episode order: demos first with unchanged
indices, so the demo partition's validation episodes keep their numbers; then each correction dataset in the given
order. Every correction episode goes to train (policy frames in them are never sampled, see so101vla/train.py).

so101_pilot_v2 (itself a lerobot-edit-dataset merge) keeps all episode metadata in meta/episodes/chunk-000/file-000
but 75 rows still name file-001..003 in ``meta/episodes/{chunk,file}_index``. Loading globs the files and does not
care; aggregation opens the named files and fails. The tagged copy gets those columns rewritten to the file each row
is actually in; the correction datasets are only checked.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def episode_meta_mismatches(root, fix=False):
    """Rows of meta/episodes/*/*.parquet whose meta/episodes/{chunk,file}_index name another file; rewritten if fix."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    bad = 0
    for path in sorted(Path(root, 'meta/episodes').glob('chunk-*/file-*.parquet')):
        chunk, file = int(path.parent.name.split('-')[1]), int(path.stem.split('-')[1])
        table = pq.read_table(path)
        wrong = (np.asarray(table['meta/episodes/chunk_index']) != chunk) | (np.asarray(table['meta/episodes/file_index']) != file)
        bad += int(wrong.sum())
        if fix and wrong.any():
            for name, value in (('meta/episodes/chunk_index', chunk), ('meta/episodes/file_index', file)):
                column = table.schema.get_field_index(name)
                table = table.set_column(column, table.schema.field(column), pa.array([value] * len(table), table.schema.field(column).type))
            pq.write_table(table, path)
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', required=True)
    parser.add_argument('--demo-partition', required=True)
    parser.add_argument('--out', required=True, help='name of the merged dataset, created beside the demo dataset')
    parser.add_argument('--partition-out', required=True)
    parser.add_argument('corrections', nargs='+')
    args = parser.parse_args()

    from lerobot.datasets.dataset_tools import add_features, merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    demo = Path(args.demo)
    out = demo.parent / args.out
    for path in (out, Path(args.partition_out)):
        if path.exists():
            raise SystemExit(f'{path} already exists')
    corrections = [Path(p) for p in args.corrections]
    for path in corrections:
        if 'intervention' not in json.loads((path / 'meta/info.json').read_text())['features']:
            raise SystemExit(f'{path} has no intervention feature; record it with scripts/dagger.fish')
        if episode_meta_mismatches(path):
            raise SystemExit(f'{path}: meta/episodes rows name the wrong file (see the docstring)')

    tagged = demo.with_name(demo.name + '_intervention')
    if not tagged.exists():
        partial = tagged.with_name(tagged.name + '.partial')
        if partial.exists():
            raise SystemExit(f'{partial} is left from an interrupted run; remove it')
        source = LeRobotDataset(f'local/{demo.name}', root=demo)
        add_features(source, {'intervention': (np.ones((source.meta.total_frames, 1), dtype=bool),
                                               {'dtype': 'bool', 'shape': (1,), 'names': None})},
                     output_dir=partial, repo_id=f'local/{tagged.name}')
        fixed = episode_meta_mismatches(partial, fix=True)
        print(f'{tagged}: rewrote the metadata file location of {fixed} episodes')
        partial.rename(tagged)
    if episode_meta_mismatches(tagged):
        raise SystemExit(f'{tagged}: meta/episodes rows name the wrong file; remove it and rerun')
    datasets = [LeRobotDataset(f'local/{p.name}', root=p) for p in [tagged, *corrections]]
    merged = merge_datasets(datasets, output_repo_id=f'local/{args.out}', output_dir=out)

    counts = [d.meta.total_episodes for d in datasets]
    if merged.meta.total_episodes != sum(counts):
        raise SystemExit(f'merged {merged.meta.total_episodes} episodes, expected {sum(counts)}')
    demo_partition = json.loads(Path(args.demo_partition).read_text())
    splits = {k: list(v) for k, v in demo_partition['splits'].items()}
    splits['train'] += list(range(counts[0], sum(counts)))
    sources, start = [], 0
    for path, n in zip([demo, *corrections], counts):
        sources.append({'dataset': str(path), 'episodes': [start, start + n - 1]})
        start += n
    partition = {'dataset': f'{out} (scripts/merge_corrections.py)', 'sources': sources,
                 'rule': f'demo splits from {args.demo_partition} unchanged; all correction episodes in train',
                 'splits': splits}
    Path(args.partition_out).write_text(json.dumps(partition, indent=2) + '\n')
    print(json.dumps({'dataset': str(out), 'episodes': sum(counts), 'sources': sources}))


if __name__ == '__main__':
    main()
