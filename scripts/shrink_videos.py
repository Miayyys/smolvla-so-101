"""Copy a LeRobot dataset with its videos at the size the policy sees, for uploading over a slow link.

    .venv/bin/python scripts/shrink_videos.py --src artifacts/datasets/so101_recap_v1 --out artifacts/datasets/so101_recap_v1_512

SmolVLA resizes every camera image to fit 512 x 512 (resize_with_pad: bilinear, aspect kept, padding on the left and
top). Each frame is resized here with that same function (so101vla.train.resize_frame, padding cut off again: top
1280x720 -> 512x288, wrist 640x480 -> 512x384) and re-encoded with SVT-AV1 (CRF, keyframe every GOP frames instead of
LeRobot's 2), keeping every frame's pts and the time base, so the episode metadata (from_timestamp etc.) stays valid.
Resizing an already-resized frame changes nothing (checked), so training with ``cache_resize`` on the copy differs
from training on the original only by this re-encoding. data/ and meta/ are copied; info.json gets the new shapes.

Every output file is checked against the resized source: same number of frames, same pts for every frame, and a mean
absolute error below MAX_MAE over every CHECK_EVERY-th frame; per-file numbers go to <out>/meta/shrink_progress.jsonl
and the totals to <out>/meta/shrink_report.json. Files are re-encoded while they are decoded (a whole file of frames
in memory per worker did not fit in 15 GB). A rerun redoes every file without a progress record.
"""

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAX_MAE = 4.0  # 8-bit levels
CHECK_EVERY = 20  # frames compared with the resized source (all frames: count and pts)


def resized(array, size):
    from so101vla.train import resize_frame
    h, w = array.shape[:2]
    ratio = max(w / size, h / size)
    rh, rw = int(h / ratio), int(w / ratio)
    return np.ascontiguousarray(resize_frame(array, size)[size - rh:, size - rw:])


def shrink(job):
    """Re-encode one file while decoding it (memory: every CHECK_EVERY-th resized frame is kept for the check)."""
    import av
    import torch
    torch.set_num_threads(1)
    src, dst, size, crf, gop = job
    started = time.time()
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_suffix('.partial.mp4')
    pts_list, kept, shape = [], {}, None
    with av.open(str(src)) as container, av.open(str(partial), 'w') as out:
        source = container.streams.video[0]
        source.thread_type = 'AUTO'
        time_base = source.time_base
        stream = None
        for index, frame in enumerate(container.decode(source)):
            image = resized(frame.to_ndarray(format='rgb24'), size)
            if stream is None:
                shape = image.shape
                stream = out.add_stream('libsvtav1', rate=30)
                stream.height, stream.width = shape[:2]
                stream.pix_fmt = 'yuv420p'
                stream.time_base = time_base
                stream.options = {'crf': str(crf), 'preset': '8', 'svtav1-params': 'lp=2'}
                stream.codec_context.gop_size = gop
            if index % CHECK_EVERY == 0:
                kept[index] = image
            pts_list.append(frame.pts)
            encoded = av.VideoFrame.from_ndarray(image, format='rgb24')
            encoded.pts, encoded.time_base = frame.pts, time_base
            for packet in stream.encode(encoded):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    errors, decoded = [], 0
    with av.open(str(partial)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= len(pts_list) or abs(float(frame.pts * frame.time_base) - float(pts_list[index] * time_base)) > 1e-4:
                raise ValueError(f'{dst}: pts differ at frame {index}')
            if index in kept:
                errors.append(np.abs(frame.to_ndarray(format='rgb24').astype(np.int16) - kept[index]).mean())
            decoded += 1
    if decoded != len(pts_list):
        raise ValueError(f'{dst}: {decoded} frames decoded, {len(pts_list)} written')
    if np.mean(errors) > MAX_MAE:
        raise ValueError(f'{dst}: mean absolute error {np.mean(errors):.2f} > {MAX_MAE}')
    partial.replace(dst)
    return {'file': str(dst), 'frames': len(pts_list), 'shape': list(shape), 'mae': float(np.mean(errors)),
            'max_frame_mae': float(np.max(errors)), 'checked_frames': len(errors), 'bytes_in': src.stat().st_size,
            'bytes_out': dst.stat().st_size, 'seconds': time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--crf', type=int, default=24)
    parser.add_argument('--gop', type=int, default=240)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)
    if out.exists() and (out / 'meta/shrink_report.json').exists():
        raise SystemExit(f'{out} is complete already')
    out.mkdir(parents=True, exist_ok=True)
    for part in ('data', 'meta'):
        if not (out / part).exists():
            shutil.copytree(src / part, out / part)
    progress = out / 'meta/shrink_progress.jsonl'
    done = {json.loads(line)['file'] for line in progress.read_text().splitlines()} if progress.exists() else set()
    for path in (out / 'videos').rglob('*.mp4') if (out / 'videos').exists() else []:
        if str(path) not in done:
            path.unlink()  # unfinished or unrecorded: redone
    jobs = [(path, out / path.relative_to(src), args.size, args.crf, args.gop)
            for path in sorted((src / 'videos').rglob('*.mp4')) if str(out / path.relative_to(src)) not in done]
    print(f'{len(jobs)} video files to shrink', flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for k, future in enumerate(as_completed([pool.submit(shrink, job) for job in jobs])):
            result = future.result()
            print(f"{k + 1}/{len(jobs)} {Path(result['file']).relative_to(out)}: {result['frames']} frames "
                  f"{result['bytes_in'] / 1e6:.0f} -> {result['bytes_out'] / 1e6:.1f} MB, MAE {result['mae']:.2f} "
                  f"({result['seconds']:.0f} s)", flush=True)
            with (out / 'meta/shrink_progress.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
    results = [json.loads(line) for line in (out / 'meta/shrink_progress.jsonl').read_text().splitlines()]
    info = json.loads((out / 'meta/info.json').read_text())
    for key, feature in info['features'].items():
        if feature['dtype'] != 'video':
            continue
        shapes = {tuple(r['shape']) for r in results if f'/{key}/' in r['file']}
        if len(shapes) != 1:
            raise SystemExit(f'{key}: shapes {shapes}')
        (height, width, channels), = shapes
        feature['shape'] = [height, width, channels]
        feature.setdefault('info', {}).update({'video.height': height, 'video.width': width, 'video.codec': 'av1',
                                               'video.pix_fmt': 'yuv420p', 'video.g': args.gop, 'video.crf': args.crf})
    (out / 'meta/info.json').write_text(json.dumps(info, indent=4) + '\n')
    report = {'source': str(src), 'size': args.size, 'crf': args.crf, 'gop': args.gop, 'files': len(results),
              'bytes_in': sum(r['bytes_in'] for r in results), 'bytes_out': sum(r['bytes_out'] for r in results),
              'mean_mae': float(np.mean([r['mae'] for r in results])), 'worst_file_mae': max(r['mae'] for r in results),
              'worst_frame_mae': max(r['max_frame_mae'] for r in results)}
    (out / 'meta/shrink_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
