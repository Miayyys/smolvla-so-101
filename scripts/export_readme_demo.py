"""Export one CSV-verified successful LeRobot episode for the README (no robot connection).

uv run python scripts/export_readme_demo.py --dataset artifacts/datasets/rollout_full_v5_key_raised \
    --success-csv runs/real_eval/rollout_full_v5_key_raised.csv --episode 1 --out docs/media
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--success-csv', type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.success_csv.open(newline='') as stream:
        matches = [r for r in csv.DictReader(stream) if int(r['episode']) == args.episode]
    if len(matches) != 1 or matches[0]['success'] != '1':
        raise ValueError('Expected one explicitly successful episode in the outcome CSV')
    table = pd.concat([pd.read_parquet(p) for p in sorted((args.dataset / 'meta/episodes').rglob('*.parquet'))])
    rows = table[table['episode_index'] == args.episode]
    if len(rows) != 1:
        raise ValueError('Episode metadata must be unique')
    row = rows.iloc[0]
    inputs, sources = [], {}
    duration = float('inf')
    for camera in ('top', 'wrist'):
        key = f'videos/observation.images.{camera}'
        source = args.dataset / 'videos' / f'observation.images.{camera}' / (
            f"chunk-{int(row[key + '/chunk_index']):03d}/file-{int(row[key + '/file_index']):03d}.mp4")
        start = float(row[key + '/from_timestamp'])
        duration = min(duration, float(row[key + '/to_timestamp']) - start)
        inputs += ['-ss', str(start), '-i', str(source)]
        sources[camera] = {'path': str(source), 'from_timestamp': start}
    args.out.mkdir(parents=True, exist_ok=True)
    movie = args.out / 'v5-key-success.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *inputs,
                    '-filter_complex', '[0:v]scale=576:324,setsar=1[a];[1:v]scale=432:324,setsar=1[b];[a][b]hstack=shortest=1[v]',
                    '-map', '[v]', '-t', str(duration), '-an', '-r', '30', '-c:v', 'libx264',
                    '-crf', '25', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie)], check=True)
    # Short normal-speed preview; the linked MP4 keeps the complete episode.
    preview_start = max(0, duration - 16)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(preview_start),
                    '-i', str(movie), '-filter_complex',
                    'fps=8,scale=756:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer',
                    '-loop', '0', str(args.out / 'v5-key-success.gif')], check=True)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(max(0, duration - 3)),
                    '-i', str(movie), '-frames:v', '1', str(args.out / 'v5-key-success.jpg')], check=True)
    record = {'dataset': str(args.dataset), 'episode': args.episode, 'outcome': matches[0],
              'sources': sources, 'duration_seconds': duration, 'playback_speed': 1,
              'preview_from_seconds': preview_start, 'audio': False,
              'selection': 'Selected successful V5 real-robot rollout; not an estimate of overall success rate',
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in args.out.glob('v5-key-success.*') if p.suffix in ('.mp4', '.gif', '.jpg')}}
    (args.out / 'v5-key-success.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'duration': duration, 'outputs': record['output_sha256']}, indent=2))


if __name__ == '__main__':
    main()
