#!/usr/bin/env python
"""Map flip/90-degree/scale TTA predictions back to original pixels."""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--xml-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def polygon(box):
    cx, cy, width, height, angle = box
    return cv2.boxPoints(
        ((cx, cy), (width, height), math.degrees(angle))).astype(np.float32)


def invert(points, transform, width, height):
    result = points.copy()
    if transform == 'original':
        return result
    if transform == 'hflip':
        result[:, 0] = width - 1 - result[:, 0]
    elif transform == 'vflip':
        result[:, 1] = height - 1 - result[:, 1]
    elif transform == 'rot90':
        result = np.stack(
            [width - 1 - points[:, 1], points[:, 0]], axis=1)
    elif transform == 'rot180':
        result = np.stack(
            [width - 1 - points[:, 0], height - 1 - points[:, 1]], axis=1)
    elif transform == 'rot270':
        result = np.stack(
            [points[:, 1], height - 1 - points[:, 0]], axis=1)
    else:
        raise ValueError(f'unknown TTA transform: {transform}')
    return result.astype(np.float32)


def main():
    args = parse_args()
    shapes = {}
    output = []
    for item in json.loads(args.input.read_text(encoding='utf-8')):
        image_id = str(item['image_id'])
        if image_id not in shapes:
            root = ET.parse(args.xml_root / f'{image_id}.xml').getroot()
            shapes[image_id] = (
                int(float(root.findtext('size/width'))),
                int(float(root.findtext('size/height'))))
        width, height = shapes[image_id]
        scale = float(item.get('scale', 1.0))
        points = polygon(item['bbox']) / scale
        points = invert(
            points, item.get('tta', 'original'), width, height)
        (cx, cy), (box_width, box_height), angle = cv2.minAreaRect(points)
        item['bbox'] = [cx, cy, box_width, box_height, math.radians(angle)]
        item.pop('tta', None); item.pop('scale', None)
        output.append(item)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {len(output)} canonical predictions')


if __name__ == '__main__':
    main()
