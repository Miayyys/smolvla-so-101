"""Measure the real frame rate of V4L2 cameras, all opened at once (as during recording).

    python scripts/camera_fps.py /dev/v4l/by-id/<top>-video-index0 /dev/v4l/by-id/<wrist>-video-index0
    python scripts/camera_fps.py --width 1280 --height 720 --fourcc MJPG /dev/video4

Cheap UVC cameras often lengthen exposure in dim light and silently drop below the requested fps; the reported
CAP_PROP_FPS stays at the requested value, so only a timed read shows it.
"""

import argparse
import threading
import time

import cv2


def measure(dev, args, out):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        out[dev] = 'open failed'
        return
    for _ in range(args.warmup):
        cap.read()
    start, frames, frame = time.time(), 0, None
    for _ in range(args.frames):
        ok, img = cap.read()
        if ok:
            frames, frame = frames + 1, img
    elapsed = time.time() - start
    size = f'{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}'
    brightness = f'{frame.mean():.0f}' if frame is not None else 'n/a'
    out[dev] = f'{size} {args.fourcc}: {frames / elapsed:.1f} fps (requested {args.fps}), mean brightness {brightness}'
    cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('devices', nargs='+')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--fourcc', default='MJPG')
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--frames', type=int, default=150)
    args = parser.parse_args()
    out = {}
    threads = [threading.Thread(target=measure, args=(d, args, out)) for d in args.devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for dev in args.devices:
        print(dev, '->', out[dev])


if __name__ == '__main__':
    main()
