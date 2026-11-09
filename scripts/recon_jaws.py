"""Extract the jaws as seen by the wrist camera, one transparent image per opening, from the arm capture.

    python scripts/recon_jaws.py --arm runs/scan/table_curtain/arm

The jaws are fixed relative to the wrist camera, so a sharp image of them can simply be laid over a render. For each
opening the capture has several views of different backgrounds: the black parts agree between those views and the table
does not. The white parts pick up reflected light and vary more, but nothing else in the median is that bright. Images stay in the raw (distorted) camera geometry, the geometry of the wrist images the policy sees.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

AGREE_STD = 9.0  # standard deviation between views below which a pixel belongs to something fixed to the camera
WHITE_MIN_LEVEL = 120  # every channel of the median above this: white plastic
LOWER_PART = 150  # jaws only appear below this image row
EDGE_FEATHER_PX = 1.5


def segment(stack):
    """Alpha mask of the jaws from views sharing one opening."""
    white = np.median(stack, axis=0).min(-1) > WHITE_MIN_LEVEL
    agree = ((stack.std(0).mean(-1) < AGREE_STD) | white).astype(np.uint8)
    agree[:LOWER_PART] = 0
    agree = cv2.morphologyEx(agree, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    agree = cv2.morphologyEx(agree, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(agree)
    bottom = agree.shape[0] - 1
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] > bottom and stats[i, cv2.CC_STAT_AREA] > 800]
    mask = np.isin(labels, keep).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arm', required=True)
    args = parser.parse_args()

    arm = Path(args.arm)
    stops = sorted(json.loads((arm / 'stops.json').read_text()).values(), key=lambda v: v['name'])
    openings = np.array([s['opening_pct'] for s in stops])
    images = np.stack([cv2.imread(str(arm / f"wrist_{s['name']}.png")) for s in stops]).astype(np.float32)
    levels = np.unique(openings)
    layers, areas = [], []
    for level in levels:
        stack = images[openings == level]
        mask = segment(stack)
        colour = np.median(stack, axis=0)
        alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), EDGE_FEATHER_PX) * mask
        layers.append(np.dstack([colour, alpha * 255]).round().astype(np.uint8))
        areas.append(mask.mean())
        print(f'opening {level:5.1f}%: {int((openings == level).sum())} views, jaws cover {mask.mean():.1%}', flush=True)
    np.savez_compressed(arm / 'jaw_layers.npz', openings=levels, layers=np.stack(layers))
    print(f'wrote {arm / "jaw_layers.npz"}')


if __name__ == '__main__':
    main()
