#!/usr/bin/env python
"""Measure HBB anchor coverage for the Oriented R-CNN RPN."""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml-root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--strides', type=int, nargs='+', default=[4, 8, 16, 32, 64])
    parser.add_argument('--scales', type=float, nargs='+', default=[2, 4, 8])
    parser.add_argument(
        '--ratios', type=float, nargs='+', default=[0.25, 0.5, 1, 2, 4])
    return parser.parse_args()


def iou(box, anchor):
    x1 = max(box[0], anchor[0]); y1 = max(box[1], anchor[1])
    x2 = min(box[2], anchor[2]); y2 = min(box[3], anchor[3])
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
    return intersection / max(area1 + area2 - intersection, 1e-9)


def best_anchor_iou(box, strides, scales, ratios):
    cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2
    best = 0.0
    for stride in strides:
        centers_x = {math.floor(cx / stride) * stride + stride / 2,
                     math.ceil(cx / stride) * stride + stride / 2}
        centers_y = {math.floor(cy / stride) * stride + stride / 2,
                     math.ceil(cy / stride) * stride + stride / 2}
        for scale in scales:
            area = (stride * scale)**2
            for ratio in ratios:
                width = math.sqrt(area / ratio); height = width * ratio
                for ax in centers_x:
                    for ay in centers_y:
                        anchor = (ax - width / 2, ay - height / 2,
                                  ax + width / 2, ay + height / 2)
                        best = max(best, iou(box, anchor))
    return best


def main():
    args = parse_args()
    image_ids = [line.strip() for line in args.split.read_text(
        encoding='utf-8-sig').splitlines() if line.strip()]
    boxes = []
    labels = []
    for image_id in image_ids:
        root = ET.parse(args.xml_root / f'{image_id}.xml').getroot()
        for obj in root.findall('objects/object'):
            points = []
            for node in obj.findall('points/point')[:4]:
                x, y = (node.text or '').split(',')[:2]
                points.append((float(x), float(y)))
            if len(points) != 4:
                continue
            xs, ys = zip(*points)
            box = (min(xs), min(ys), max(xs), max(ys))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            name = (obj.findtext('possibleresult/name') or '').strip()
            boxes.append(box); labels.append(name)

    boxes = np.asarray(boxes, dtype=np.float32)
    best = np.zeros(len(boxes), dtype=np.float32)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for stride in args.strides:
        centers_x = [np.floor(cx / stride) * stride + stride / 2,
                     np.ceil(cx / stride) * stride + stride / 2]
        centers_y = [np.floor(cy / stride) * stride + stride / 2,
                     np.ceil(cy / stride) * stride + stride / 2]
        for scale in args.scales:
            area = (stride * scale)**2
            for ratio in args.ratios:
                width = math.sqrt(area / ratio); height = width * ratio
                anchor_area = width * height
                for ax in centers_x:
                    for ay in centers_y:
                        x1 = np.maximum(boxes[:, 0], ax - width / 2)
                        y1 = np.maximum(boxes[:, 1], ay - height / 2)
                        x2 = np.minimum(boxes[:, 2], ax + width / 2)
                        y2 = np.minimum(boxes[:, 3], ay + height / 2)
                        intersection = np.maximum(x2-x1, 0) * np.maximum(y2-y1, 0)
                        overlap = intersection / np.maximum(
                            box_area + anchor_area - intersection, 1e-9)
                        best = np.maximum(best, overlap)

    values = best.tolist()
    per_class = defaultdict(list)
    for name, value in zip(labels, values):
        per_class[name].append(value)

    def summarize(items):
        ordered = sorted(items)
        return dict(
            count=len(items),
            mean=sum(items) / len(items),
            p05=ordered[int(0.05 * (len(ordered) - 1))],
            recall_at_05=sum(value >= 0.5 for value in items) / len(items),
            recall_at_07=sum(value >= 0.7 for value in items) / len(items))

    report = dict(overall=summarize(values), per_class={
        name: summarize(items) for name, items in per_class.items()
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
