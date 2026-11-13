"""Check RECAP labels (so101vla/value.py --stage label) before training a policy on them.

    .venv/bin/python scripts/recap_label_report.py --run runs/so101_value_v1 --config configs/so101_value_v1.json

Reads <run>/labels.npz and <run>/data.npz, the outcomes beside the dataset and the value partition. Reports, for the
episodes the value function was trained on and for the held-out ones separately:
  - value error against the normalised return (overfitting: much smaller on trained episodes),
  - the spread of the advantages (a value function that has memorised its episodes gives advantages near 0 there, so
    the labels would only be noise),
  - the share of positive frames by kind of episode (failed rollouts should get fewer than successful ones),
  - how well the value at the first frame separates successful from failed episodes (rank AUC).
Writes the numbers to <run>/label_report.json as well.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def auc(scores, labels):
    """Probability that a random positive scores above a random negative (ties count half)."""
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    cfg = json.loads(Path(args.config).read_text())
    labels = np.load(run / 'labels.npz')
    data = np.load(run / 'data.npz')
    outcomes = {r['episode']: r for r in json.loads((Path(cfg['dataset_root']) / 'meta/recap_outcomes.json').read_text())}
    splits = json.loads(Path(cfg['partition']).read_text())['splits']
    eps = data['episodes']
    value, target, adv, ind = labels['value'], labels['target'], labels['advantage'], labels['indicator']
    human_correction = labels['human_correction']
    kind = np.array([outcomes[int(e)]['kind'] + ('' if outcomes[int(e)]['success'] else ' failed')
                     if outcomes[int(e)]['success'] is not None else 'unknown' for e in eps])
    kind = np.where(human_correction, 'correction (human frames)', np.where(kind == 'correction', 'correction (policy frames)', kind))
    if 'twin_miss' in labels:
        kind = np.where(labels['twin_miss'], 'twin (deliberate miss)', kind)
    report = {}
    for split in ('train', 'validation'):
        mask = np.isin(eps, splits[split]) & ~np.isnan(adv)
        part = {'frames': int(mask.sum()),
                'value_mae': float(np.abs(value - target)[mask].mean()),
                'advantage_std': float(adv[mask].std()),
                'advantage_abs_mean': float(np.abs(adv[mask]).mean()),
                'by_kind': {}}
        for k in sorted(set(kind[mask])):
            m = mask & (kind == k)
            part['by_kind'][k] = {'frames': int(m.sum()), 'positive_share': float(ind[m].mean()),
                                  'value_mae': float(np.abs(value - target)[m].mean()),
                                  'advantage_mean': float(adv[m].mean()), 'advantage_std': float(adv[m].std())}
        episodes = [e for e in splits[split] if outcomes[e]['kind'] in ('rollout', 'correction') and outcomes[e]['success'] is not None]
        first = [np.flatnonzero(eps == e)[0] for e in episodes]
        part['rollout_start_value_auc'] = auc(value[first], [outcomes[e]['success'] for e in episodes])
        part['rollout_episodes'] = len(episodes)
        share = [ind[eps == e].mean() for e in episodes]
        part['episode_positive_share_auc'] = auc(share, [outcomes[e]['success'] for e in episodes])
        report[split] = part
    (run / 'label_report.json').write_text(json.dumps(report, indent=2) + '\n')
    for split, part in report.items():
        print(f"== {split}: {part['frames']} frames, value MAE {part['value_mae']:.3f}, advantage std {part['advantage_std']:.4f} "
              f"(|A| mean {part['advantage_abs_mean']:.4f}); rollout episodes {part['rollout_episodes']}: "
              f"AUC of the first-frame value {part['rollout_start_value_auc']}, of the positive share {part['episode_positive_share_auc']}")
        for k, v in part['by_kind'].items():
            print(f"   {k:28s} {v['frames']:7d} frames  positive {v['positive_share']:.2f}  value MAE {v['value_mae']:.3f}  "
                  f"A mean {v['advantage_mean']:+.4f} std {v['advantage_std']:.4f}")


if __name__ == '__main__':
    main()
