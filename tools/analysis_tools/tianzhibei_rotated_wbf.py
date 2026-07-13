#!/usr/bin/env python
"""Class-wise rotated weighted-box fusion for JSON prediction files.

Input files contain a list of records with ``image_id``, ``label``, ``score``
and ``bbox=[cx, cy, width, height, angle_radians]``.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iou-thr', type=float, default=0.55)
    parser.add_argument('--score-thr', type=float, default=0.01)
    parser.add_argument('--weights', type=float, nargs='*')
    parser.add_argument('--class-score-thresholds', type=Path)
    parser.add_argument('--class-iou-thresholds', type=Path)
    return parser.parse_args()


def polygon(box):
    cx, cy, width, height, angle = box
    return cv2.boxPoints(((cx, cy), (width, height), math.degrees(angle)))


def rotated_iou(first, second):
    a = cv2.convexHull(polygon(first)).reshape(-1, 2)
    b = cv2.convexHull(polygon(second)).reshape(-1, 2)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - intersection
    return intersection / max(union, 1e-9)


def fuse(cluster):
    weights = np.asarray([item['score'] * item['model_weight']
                          for item in cluster], dtype=np.float64)
    boxes = np.asarray([item['bbox'] for item in cluster], dtype=np.float64)
    normalizer = weights.sum()
    center_size = (boxes[:, :4] * weights[:, None]).sum(axis=0) / normalizer
    sin2 = (np.sin(2 * boxes[:, 4]) * weights).sum() / normalizer
    cos2 = (np.cos(2 * boxes[:, 4]) * weights).sum() / normalizer
    angle = 0.5 * math.atan2(sin2, cos2)
    score = sum(item['score'] * item['model_weight'] for item in cluster)
    score /= sum(item['model_weight'] for item in cluster)
    return [*center_size.tolist(), angle], score


def main():
    args = parse_args()
    model_weights = args.weights or [1.0] * len(args.inputs)
    class_scores = (json.loads(args.class_score_thresholds.read_text())
                    if args.class_score_thresholds else {})
    class_ious = (json.loads(args.class_iou_thresholds.read_text())
                  if args.class_iou_thresholds else {})
    if len(model_weights) != len(args.inputs):
        raise ValueError('weights and inputs must have the same length')
    groups = defaultdict(list)
    for path, model_weight in zip(args.inputs, model_weights):
        for item in json.loads(path.read_text(encoding='utf-8')):
            threshold = float(class_scores.get(
                str(item['label']), args.score_thr))
            if item['score'] >= threshold:
                item['model_weight'] = model_weight
                groups[(str(item['image_id']), int(item['label']))].append(item)

    output = []
    for (image_id, label), items in groups.items():
        iou_threshold = float(class_ious.get(str(label), args.iou_thr))
        clusters = []
        for item in sorted(items, key=lambda value: value['score'], reverse=True):
            match = None
            for cluster in clusters:
                box, _ = fuse(cluster)
                if rotated_iou(item['bbox'], box) >= iou_threshold:
                    match = cluster
                    break
            if match is None:
                clusters.append([item])
            else:
                match.append(item)
        for cluster in clusters:
            box, score = fuse(cluster)
            output.append(dict(
                image_id=image_id, label=label, score=score, bbox=box))
    args.output.write_text(
        json.dumps(output, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {len(output)} fused boxes to {args.output}')


if __name__ == '__main__':
    main()
