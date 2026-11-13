"""Finish/upload twin v2 and prepare RECAP v2 on GPUServer, without starting a training run.

Run locally with .venv/bin/python scripts/recap_prepare_v2.py. The existing top-up job must finish first.
The server uses the frozen value v1 checkpoint to label the merged data, then prepares and smoke-tests the policy.
Progress and errors are persisted in runs/twin/prepare_recap_v2_{local,server}.json; safe failures never delete data.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = '/root/sj-tmp/so101-vla-edge'
SHARED = Path('/root/sj-fs/datasets')
SSH = ['ssh', '-F', '/home/loser/.ssh/config', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', 'GPUServer']
DATASETS = {'so101_twin_gen_v2_512': (236, 76458), 'so101_twin_gen_v2b_512': (64, 20784)}
VALUE_CONFIG = 'configs/so101_value_v2_labels.json'
POLICY_CONFIG = 'configs/so101_recap_v2.json'
STATE = None


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.tmp')
    partial.write_text(json.dumps(data, indent=2) + '\n')
    partial.replace(path)


def status(stage, **extra):
    record = {'stage': stage, 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra}
    write(STATE, record)
    print(json.dumps(record), flush=True)


def run(args, log=None):
    if log:
        with Path(log).open('a') as output:
            subprocess.run(args, check=True, stdout=output, stderr=subprocess.STDOUT)
    else:
        subprocess.run(args, check=True)


def ssh(command):
    return subprocess.check_output([*SSH, command], text=True)


def sync(source, destination):
    run(['rsync', '-rt', '--checksum', '--partial', '--partial-dir=.rsync-partial',
         '-e', shlex.join(SSH[:-1]), str(source), destination])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_dataset(path, expected):
    info = read(path / 'meta/info.json')
    if (info['total_episodes'], info['total_frames']) != expected:
        raise ValueError(f'{path}: incomplete dataset')
    rows = read(path / 'meta/twin_expert.json')
    if len(rows) != expected[0] or sum(r['frames'] for r in rows) != expected[1]:
        raise ValueError(f'{path}: expert sidecar does not cover the dataset')
    report = read(path / 'meta/shrink_report.json')
    if report['worst_file_mae'] >= 4:
        raise ValueError(f'{path}: failed compression quality check')


def local():
    status('syncing_code')
    ssh(f'mkdir -p {SERVER_ROOT}/runs/twin')
    files = [*Path('so101vla').glob('*.py'), Path(__file__).relative_to(ROOT),
             *map(Path, ['scripts/recap_extend.py', 'scripts/merge_corrections.py',
                         'scripts/recap_label_report.py', VALUE_CONFIG, POLICY_CONFIG,
                         'configs/so101_recap_v1_partition_policy.json',
                         'configs/so101_recap_v1_partition_value.json'])]
    run(['rsync', '-rt', '--checksum', '--relative', '-e', shlex.join(SSH[:-1]),
         *map(str, files), f'GPUServer:{SERVER_ROOT}/'])
    # Upload the completed batch now, including its previously missing miss-frame annotations.
    sync('artifacts/datasets/so101_twin_gen_v2_512/',
         'GPUServer:/root/sj-fs/datasets/so101_twin_gen_v2_512/')
    deadline = time.monotonic() + 3 * 3600
    while True:
        done = Path('runs/twin/topup_v2.log')
        report = Path('artifacts/datasets/so101_twin_gen_v2b_512/meta/shrink_report.json')
        if done.exists() and 'TWIN_V2B_DONE' in done.read_text() and report.exists():
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Local top-up did not finish within three hours; inspect its logs')
        if not subprocess.run(['pgrep', '-f', '^/bin/bash scripts/twin_topup_v2.sh$'],
                              stdout=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError('Local top-up process stopped before writing TWIN_V2B_DONE')
        status('waiting_for_topup', log='runs/twin/to_lerobot_v2b.log')
        time.sleep(30)
    manifest = {}
    for name, expected in DATASETS.items():
        status('checking_and_uploading', dataset=name)
        generated = 'runs/twin/generated_v2b' if 'v2b' in name else 'runs/twin/generated_v2'
        run([sys.executable, 'scripts/recap_extend.py', 'sidecar', '--dataset', name, '--generated', generated])
        path = Path('artifacts/datasets') / name
        check_dataset(path, expected)
        manifest[name] = {str(p.relative_to(path)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                          for p in sorted(path.rglob('*')) if p.is_file()}
        sync(str(path) + '/', f'GPUServer:/root/sj-fs/datasets/{name}/')
    write('runs/twin/upload_v2_manifest.json', manifest)
    sync('runs/twin/upload_v2_manifest.json', f'GPUServer:{SERVER_ROOT}/runs/twin/upload_v2_manifest.json')
    status('starting_server_preparation')
    print(ssh(f'cd {shlex.quote(SERVER_ROOT)} && nohup .venv/bin/python -u scripts/recap_prepare_v2.py --server '
              '> runs/twin/prepare_recap_v2_server.log 2>&1 < /dev/null &'), flush=True)
    # The server job is detached; monitor it without tying its lifetime to SSH.
    deadline = time.monotonic() + 8 * 3600
    while time.monotonic() < deadline:
        time.sleep(30)
        try:
            remote = json.loads(ssh(f'cat {SERVER_ROOT}/runs/twin/prepare_recap_v2_server.json'))
        except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
            status('monitor_connection_retry', error=str(error))
            continue
        status('server_preparation', server=remote)
        if remote['stage'] == 'failed':
            raise RuntimeError(f'Server preparation failed: {remote}')
        if remote['stage'] == 'ready':
            for filename in ['so101_recap_v2_partition_policy.json', 'so101_recap_v2_partition_value.json']:
                sync(f'GPUServer:{SERVER_ROOT}/configs/{filename}', 'configs/')
            for dirname, names in {
                'so101_value_v2_labels': ['labels.npz', 'labels.json', 'label_report.json', 'prepared.json', 'stats.json', 'provenance.json'],
                'so101_recap_v2': ['prepared.json', 'stats.json', 'trainable.json', 'smoke.log'],
            }.items():
                out = Path('runs') / dirname
                out.mkdir(exist_ok=True)
                for name in names:
                    sync(f'GPUServer:{SERVER_ROOT}/runs/{dirname}/{name}', str(out / name))
            sync(f'GPUServer:{SERVER_ROOT}/runs/twin/prepare_recap_v2_server.json',
                 'runs/twin/prepare_recap_v2_server.json')
            status('ready', server=remote)
            return
    raise TimeoutError('Server preparation exceeded eight hours; the detached job may still be running')


def server(resume_label=False):
    from merge_corrections import episode_meta_mismatches

    for proc in subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines():
        if ' -m so101vla.train ' in proc or ' -m so101vla.value ' in proc:
            raise RuntimeError(f'Another model job is active: {proc}')
    if resume_label:
        value_cfg, policy_cfg = read(VALUE_CONFIG), read(POLICY_CONFIG)
        out = Path(value_cfg['output'])
        if not (out / 'prepared.json').is_file() or (out / 'labels.npz').exists():
            raise ValueError('Label resume requires prepared data and no completed labels')
        return finish_server_preparation(value_cfg, policy_cfg)
    status('verifying_uploads')
    manifest = read('runs/twin/upload_v2_manifest.json')
    for name, expected in DATASETS.items():
        path = SHARED / name
        check_dataset(path, expected)
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
        if actual != set(manifest[name]):
            raise ValueError(f'{path}: uploaded file set differs')
        for relative, info in manifest[name].items():
            if (path / relative).stat().st_size != info['bytes'] or digest(path / relative) != info['sha256']:
                raise ValueError(f'{path / relative}: upload checksum mismatch')
    for name in ['so101_recap_v1_512', *DATASETS]:
        link = Path('artifacts/datasets') / name
        target = SHARED / name
        if link.is_symlink() or link.exists():
            if link.resolve() != target:
                raise ValueError(f'{link}: unexpected dataset location')
        else:
            link.symlink_to(target, target_is_directory=True)
    status('merging')
    run([sys.executable, 'scripts/recap_extend.py', 'merge', '--base', 'so101_recap_v1_512',
         '--base-name', 'so101_recap_v1', '--twin', *DATASETS, '--name', 'so101_recap_v2',
         '--out-dir', str(SHARED / 'so101_recap_v2_512')], 'runs/twin/merge_recap_v2.log')
    root = SHARED / 'so101_recap_v2_512'
    episode_meta_mismatches(root, fix=True)
    link = Path('artifacts/datasets/so101_recap_v2_512')
    if link.exists() or link.is_symlink():
        raise ValueError(f'{link} already exists')
    link.symlink_to(root, target_is_directory=True)
    info = read(root / 'meta/info.json')
    if (info['total_episodes'], info['total_frames']) != (840, 455730):
        raise ValueError('Unexpected merged dataset counts')
    for kind in ['policy', 'value']:
        old = read(f'configs/so101_recap_v1_partition_{kind}.json')['splits']
        new = read(f'configs/so101_recap_v2_partition_{kind}.json')['splits']
        if old['validation'] != new['validation'] or old['test'] != new['test']:
            raise ValueError('Held-out splits changed')
        if new['train'] != old['train'] + list(range(540, 840)):
            raise ValueError('New twin episodes are not all in train')
    value_cfg, policy_cfg = read(VALUE_CONFIG), read(POLICY_CONFIG)
    value_out, policy_out = Path(value_cfg['output']), Path(policy_cfg['output'])
    value_out.mkdir(exist_ok=True)
    policy_out.mkdir(exist_ok=True)
    checkpoint = Path(value_cfg['value']['checkpoint_source']).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    (value_out / 'best').symlink_to(checkpoint.parent, target_is_directory=True)
    write(value_out / 'provenance.json', {'checkpoint': str(checkpoint), 'sha256': digest(checkpoint),
                                         'value_retrained': False, 'dataset': str(root)})
    status('preparing_value_cache')
    run([sys.executable, '-m', 'so101vla.value', '--config', VALUE_CONFIG, '--stage', 'prepare'], value_out / 'prepare.log')
    finish_server_preparation(value_cfg, policy_cfg)


def finish_server_preparation(value_cfg, policy_cfg):
    import numpy as np

    root = SHARED / 'so101_recap_v2_512'
    value_out, policy_out = Path(value_cfg['output']), Path(policy_cfg['output'])
    status('labelling_with_value_v1')
    run([sys.executable, '-m', 'so101vla.value', '--config', VALUE_CONFIG, '--stage', 'label',
         '--reuse-values-from', 'runs/so101_value_v1'], value_out / 'label.log')
    status('checking_labels')
    run([sys.executable, 'scripts/recap_label_report.py', '--run', str(value_out), '--config', VALUE_CONFIG], value_out / 'report.log')
    with np.load(value_out / 'labels.npz') as labels:
        ind = labels['indicator']
        if ind.shape != (455730,) or not np.isin(ind, [0, 1]).all() or not np.isfinite(labels['value']).all():
            raise ValueError('Invalid RECAP labels')
        if not (ind[labels['twin_miss']] == 0).all() or not (ind[labels['human_correction']] == 1).all():
            raise ValueError('Deliberate-miss/correction labels are wrong')
        outcomes = read(root / 'meta/recap_outcomes.json')
        if int(labels['twin_miss'].sum()) != sum(r.get('miss_frames', 0) for r in outcomes):
            raise ValueError('Miss-frame counts differ from the generated annotations')
    status('preparing_policy_cache')
    run([sys.executable, '-m', 'so101vla.train', '--config', POLICY_CONFIG, '--stage', 'prepare'], policy_out / 'prepare.log')
    if read(value_out / 'prepared.json')['caches'] != read(policy_out / 'prepared.json')['caches']:
        raise ValueError('Shared value/policy cache bytes differ')
    status('policy_smoke_test')
    run([sys.executable, '-m', 'so101vla.train', '--config', POLICY_CONFIG, '--stage', 'smoke'], policy_out / 'smoke.log')
    if not any(json.loads(line).get('stage') == 'smoke_completed' for line in (policy_out / 'events.jsonl').read_text().splitlines()):
        raise ValueError('Policy smoke test did not finish')
    if (policy_out / 'progress.json').exists():
        raise ValueError('A formal policy training run was unexpectedly started')
    status('backing_up_preparation')
    backup = Path('/root/sj-fs/so101_recap_v2_preparation')
    backup.mkdir(exist_ok=True)
    for folder in [value_out, policy_out, Path('configs')]:
        destination = backup / folder
        destination.mkdir(parents=True, exist_ok=True)
        if folder == Path('configs'):
            names = [Path(VALUE_CONFIG).name, Path(POLICY_CONFIG).name,
                     'so101_recap_v2_partition_policy.json', 'so101_recap_v2_partition_value.json']
        else:
            names = [p.name for p in folder.iterdir() if p.is_file()]
        import shutil
        for name in names:
            shutil.copy2(folder / name, destination / name)
    status('ready', episodes=840, frames=455730, value_retrained=False, policy_training_started=False,
           training_command=f'.venv/bin/python -m so101vla.train --config {POLICY_CONFIG} --stage train',
           note='Top camera cache is in /dev/shm/recap_v2 and must be rebuilt after a server reboot')


def main():
    global STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--resume-waiting', action='store_true',
                        help='resume only a local workflow that was still waiting for the top-up')
    parser.add_argument('--resume-server-label', action='store_true',
                        help='resume the server workflow from its interrupted labelling stage')
    args = parser.parse_args()
    if args.resume_server_label and not args.server:
        parser.error('--resume-server-label requires --server')
    os.chdir(ROOT)
    os.environ['PYTHONUNBUFFERED'] = '1'
    mode = 'server' if args.server else 'local'
    STATE = Path(f'runs/twin/prepare_recap_v2_{mode}.json')
    STATE.parent.mkdir(exist_ok=True)
    with STATE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        can_resume = (args.resume_waiting and not args.server and STATE.exists()
                      and read(STATE)['stage'] == 'waiting_for_topup')
        if args.resume_server_label and STATE.exists():
            previous = read(STATE)
            if previous['stage'] == 'failed':
                previous = previous.get('previous') or {}
            can_resume = previous.get('stage') == 'labelling_with_value_v1'
        if STATE.exists() and not can_resume:
            raise SystemExit(f'{STATE} already exists; inspect it before restarting this workflow')
        try:
            if args.server:
                server(resume_label=args.resume_server_label)
            else:
                local()
        except Exception:
            previous = read(STATE) if STATE.exists() else None
            status('failed', previous=previous, error=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
