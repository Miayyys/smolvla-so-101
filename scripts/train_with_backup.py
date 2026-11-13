"""Run one training job and copy each new best checkpoint to persistent storage (no automatic shutdown)."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def backup(out, target):
    weight = out / 'best/model.safetensors'
    if not weight.exists():
        return None
    # save_model atomically replaces the weight. An open descriptor remains on a consistent version while copying.
    destination = target / 'best'
    destination.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    partial = destination / 'model.partial.safetensors'
    with weight.open('rb') as source, partial.open('wb') as dest:
        stamp = os.fstat(source.fileno()).st_mtime_ns
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
            dest.write(block)
    if checksum(partial) != h.hexdigest():
        raise RuntimeError('Persistent checkpoint backup checksum differs')
    partial.replace(destination / weight.name)
    for file in (out / 'best').iterdir():
        if file.is_file() and file.name != weight.name and '.partial.' not in file.name:
            shutil.copy2(file, destination / file.name)
    for file in out.iterdir():
        if file.is_file() and file.suffix in ('.json', '.jsonl', '.log'):
            shutil.copy2(file, target / file.name)
    (target / 'backup.json').write_text(json.dumps({'source': str(out), 'sha256': h.hexdigest(),
        'source_mtime_ns': stamp, 'copied_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=2) + '\n')
    print('BACKUP_VERIFIED', h.hexdigest(), flush=True)
    return stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--backup-root', type=Path, default=Path('/root/sj-fs'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['output'])
    target = args.backup_root / out.name
    target.mkdir(parents=True, exist_ok=True)
    with (out / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'progress.json').exists() or (out / 'train.log').exists():
            raise RuntimeError('A formal run already exists; refusing to overwrite')
        for process in subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines():
            if ' -m so101vla.train ' in process:
                raise RuntimeError('Another policy job is active')
        command = [sys.executable, '-u', '-m', 'so101vla.train', '--config', str(args.config), '--stage', 'train']
        with (out / 'train.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
        record = {'pid': child.pid, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  'command': command, 'config_sha256': checksum(args.config),
                  'initialization': cfg.get('init_weights') or cfg['model_dir'], 'backup': str(target),
                  'automatic_shutdown': False}
        (out / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
        shutil.copy2(args.config, target / args.config.name)
        shutil.copy2(cfg['partition'], target / Path(cfg['partition']).name)
        print(json.dumps(record), flush=True)
        last_stamp = None
        while child.poll() is None:
            weight = out / 'best/model.safetensors'
            if weight.exists() and weight.stat().st_mtime_ns != last_stamp:
                try:
                    last_stamp = backup(out, target)
                except Exception as error:
                    print('BACKUP_RETRY', repr(error), flush=True)
            time.sleep(30)
        ok = False
        for _ in range(3):
            try:
                copied = backup(out, target)
                ok = copied is not None
                if ok:
                    break
            except Exception as error:
                print('BACKUP_RETRY', repr(error), flush=True)
            time.sleep(10)
        result = {'training_exit_code': child.returncode, 'backup_verified': ok,
                  'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        (out / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
        shutil.copy2(out / 'completion.json', target / 'completion.json')
        print(json.dumps(result), flush=True)
        if child.returncode or not ok:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
