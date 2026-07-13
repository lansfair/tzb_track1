#!/usr/bin/env python
"""Create SHA-grouped, multilabel-stratified Tianzhibei folds."""
import argparse
import csv
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = (
    'Small Car', 'Van', 'Dump Truck', 'Cargo Truck', 'other-vehicle',
    'Bus', 'Truck Tractor', 'Excavator', 'Trailer', 'Tractor')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-csv', type=Path, required=True)
    parser.add_argument('--xml-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=3407)
    return parser.parse_args()


def image_features(xml_path):
    root = ET.parse(xml_path).getroot()
    counts = Counter(
        (obj.findtext('possibleresult/name') or '').strip()
        for obj in root.findall('objects/object'))
    width = int(float(root.findtext('size/width', 0)))
    height = int(float(root.findtext('size/height', 0)))
    total = sum(counts[name] for name in CLASSES)
    features = []
    features.extend(float(counts[name] > 0) for name in CLASSES)
    features.extend(math.log1p(counts[name]) for name in CLASSES)
    size_bin = 0 if max(width, height) <= 800 else 1 if max(width, height) <= 1024 else 2
    features.extend(float(size_bin == index) for index in range(3))
    density_bin = 0 if total < 32 else 1 if total < 118 else 2
    features.extend(float(density_bin == index) for index in range(3))
    return features, counts, (width, height), total


def main():
    args = parse_args()
    random.seed(args.seed)
    groups = defaultdict(list)
    with args.split_csv.open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            if row['recommended_role'] == 'drop_exact_duplicate':
                continue
            groups[row['sha256']].append(Path(row['competition_image']).stem)

    records = []
    for digest, image_ids in groups.items():
        representative = image_ids[0]
        features, counts, shape, total = image_features(
            args.xml_root / f'{representative}.xml')
        records.append(dict(
            sha256=digest,
            image_ids=sorted(image_ids),
            features=features,
            counts=dict(counts),
            shape=shape,
            total=total))

    dimension = len(records[0]['features'])
    global_sum = [
        sum(record['features'][i] for record in records)
        for i in range(dimension)
    ]
    target = [value / args.folds for value in global_sum]
    rarity = [1.0 / max(value, 1.0) for value in global_sum]
    random.shuffle(records)
    records.sort(
        key=lambda record: sum(
            value * rarity[i] for i, value in enumerate(record['features'])),
        reverse=True)

    fold_records = [[] for _ in range(args.folds)]
    fold_sums = [[0.0] * dimension for _ in range(args.folds)]
    maximum_fold_size = math.ceil(len(records) / args.folds)
    for record in records:
        candidates = []
        for fold in range(args.folds):
            if len(fold_records[fold]) >= maximum_fold_size:
                continue
            feature_delta = sum(
                (((fold_sums[fold][i] + record['features'][i] - target[i]) /
                  max(target[i], 1.0))**2 -
                 ((fold_sums[fold][i] - target[i]) /
                  max(target[i], 1.0))**2)
                for i in range(dimension))
            fold_target = len(records) / args.folds
            size_delta = (
                ((len(fold_records[fold]) + 1 - fold_target) /
                 fold_target)**2 -
                ((len(fold_records[fold]) - fold_target) /
                 fold_target)**2)
            candidates.append((feature_delta + 0.25 * size_delta, fold))
        fold = min(candidates)[1]
        fold_records[fold].append(record)
        fold_sums[fold] = [
            old + value
            for old, value in zip(fold_sums[fold], record['features'])
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_ids = {image_id for record in records for image_id in record['image_ids']}
    manifest = []
    for fold, selected in enumerate(fold_records):
        val_ids = sorted(
            image_id for record in selected for image_id in record['image_ids'])
        train_ids = sorted(all_ids - set(val_ids))
        (args.output_dir / f'fold{fold}_train.txt').write_text(
            ''.join(f'{image_id}\n' for image_id in train_ids), encoding='utf-8')
        (args.output_dir / f'fold{fold}_val.txt').write_text(
            ''.join(f'{image_id}\n' for image_id in val_ids), encoding='utf-8')
        class_counts = Counter()
        for record in selected:
            class_counts.update(record['counts'])
        manifest.append(dict(
            fold=fold,
            train_images=len(train_ids),
            val_images=len(val_ids),
            val_class_instances={name: class_counts[name] for name in CLASSES}))
    (args.output_dir / 'fold_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
