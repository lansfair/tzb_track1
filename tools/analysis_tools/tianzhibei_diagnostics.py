#!/usr/bin/env python
"""Detailed AP50, confusion, scale, density, and hard-image diagnostics."""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASSES = (
    'Small Car', 'Van', 'Dump Truck', 'Cargo Truck', 'other-vehicle',
    'Bus', 'Truck Tractor', 'Excavator', 'Trailer', 'Tractor')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--xml-root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hard-ids-output', type=Path)
    parser.add_argument('--score-thr', type=float, default=0.01)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    return parser.parse_args()


def pred_polygon(box):
    cx, cy, width, height, angle = box
    return cv2.boxPoints(
        ((cx, cy), (width, height), math.degrees(angle))).astype(np.float32)


def polygon_iou(first, second):
    first = cv2.convexHull(first).reshape(-1, 2)
    second = cv2.convexHull(second).reshape(-1, 2)
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = (abs(cv2.contourArea(first)) + abs(cv2.contourArea(second)) -
             intersection)
    return intersection / max(union, 1e-9)


def read_gt(path):
    root = ET.parse(path).getroot()
    width = int(float(root.findtext('size/width')))
    height = int(float(root.findtext('size/height')))
    objects = []
    for obj in root.findall('objects/object'):
        name = (obj.findtext('possibleresult/name') or '').strip()
        if name not in CLASSES:
            continue
        points = []
        for node in obj.findall('points/point')[:4]:
            x, y = (node.text or '').split(',')[:2]
            points.append((float(x), float(y)))
        points = np.asarray(points, dtype=np.float32)
        rectangle = cv2.minAreaRect(points)
        short_side = min(rectangle[1])
        boundary = bool(((points[:, 0] <= 1) | (points[:, 1] <= 1) |
                         (points[:, 0] >= width - 2) |
                         (points[:, 1] >= height - 2)).any())
        objects.append(dict(
            label=CLASSES.index(name), polygon=points,
            short_side=short_side, boundary=boundary))
    return objects


def average_precision(tp, fp, total_gt):
    if total_gt == 0:
        return None
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[indices + 1] - recall[indices]) *
                        precision[indices + 1]))


def main():
    args = parse_args()
    image_ids = [item.strip() for item in args.split.read_text(
        encoding='utf-8-sig').splitlines() if item.strip()]
    ground_truth = {
        image_id: read_gt(args.xml_root / f'{image_id}.xml')
        for image_id in image_ids
    }
    predictions = defaultdict(list)
    for item in json.loads(args.predictions.read_text(encoding='utf-8')):
        if item['score'] >= args.score_thr:
            item['polygon'] = pred_polygon(item['bbox'])
            predictions[str(item['image_id'])].append(item)

    total_by_class = [0] * len(CLASSES)
    for objects in ground_truth.values():
        for obj in objects:
            total_by_class[obj['label']] += 1
    class_detections = [[] for _ in CLASSES]
    for image_id, items in predictions.items():
        for item in items:
            class_detections[int(item['label'])].append((image_id, item))

    per_class_ap = {}
    for label, detections in enumerate(class_detections):
        used = defaultdict(set); tp = []; fp = []
        for image_id, pred in sorted(
                detections, key=lambda pair: pair[1]['score'], reverse=True):
            candidates = ground_truth.get(image_id, [])
            matches = [(polygon_iou(pred['polygon'], gt['polygon']), index)
                       for index, gt in enumerate(candidates)
                       if gt['label'] == label and index not in used[image_id]]
            best = max(matches, default=(0.0, -1))
            correct = best[0] >= args.iou_thr
            tp.append(float(correct)); fp.append(float(not correct))
            if correct:
                used[image_id].add(best[1])
        per_class_ap[CLASSES[label]] = average_precision(
            tp, fp, total_by_class[label])

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    scale = defaultdict(lambda: [0, 0])
    density = {'dense': [0, 0], 'normal': [0, 0]}
    boundary = [0, 0]
    hard_scores = {}
    for image_id in image_ids:
        gt_items = ground_truth[image_id]
        pred_items = sorted(
            predictions.get(image_id, []),
            key=lambda item: item['score'], reverse=True)
        pairs = []
        for pred_index, pred in enumerate(pred_items):
            for gt_index, gt in enumerate(gt_items):
                pairs.append((polygon_iou(pred['polygon'], gt['polygon']),
                              pred_index, gt_index))
        used_pred = set(); used_gt = set(); wrong = 0
        for value, pred_index, gt_index in sorted(pairs, reverse=True):
            if value < args.iou_thr:
                break
            if pred_index in used_pred or gt_index in used_gt:
                continue
            used_pred.add(pred_index); used_gt.add(gt_index)
            gt_label = gt_items[gt_index]['label']
            pred_label = int(pred_items[pred_index]['label'])
            confusion[gt_label, pred_label] += 1
            wrong += int(gt_label != pred_label)
        dense_key = 'dense' if len(gt_items) >= 118 else 'normal'
        density[dense_key][0] += len(used_gt); density[dense_key][1] += len(gt_items)
        for index, gt in enumerate(gt_items):
            key = '<8' if gt['short_side'] < 8 else '8-12' if gt['short_side'] <= 12 else '>12'
            scale[key][0] += int(index in used_gt); scale[key][1] += 1
            if gt['boundary']:
                boundary[0] += int(index in used_gt); boundary[1] += 1
        hard_scores[image_id] = (
            len(gt_items) - len(used_gt) + wrong +
            max(len(pred_items) - len(used_pred), 0))

    report = dict(
        mAP50=float(np.mean([value for value in per_class_ap.values()
                             if value is not None])),
        per_class_AP50=per_class_ap,
        confusion_matrix=dict(classes=CLASSES, values=confusion.tolist()),
        localization_recall_by_short_side={
            key: hits / max(total, 1) for key, (hits, total) in scale.items()},
        localization_recall_by_density={
            key: hits / max(total, 1) for key, (hits, total) in density.items()},
        boundary_localization_recall=boundary[0] / max(boundary[1], 1))
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.hard_ids_output:
        ordered = sorted(hard_scores, key=hard_scores.get, reverse=True)
        args.hard_ids_output.parent.mkdir(parents=True, exist_ok=True)
        args.hard_ids_output.write_text(
            ''.join(f'{image_id}\n' for image_id in ordered[:max(1, len(ordered)//5)]),
            encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
