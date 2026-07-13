#!/usr/bin/env python
"""Build direction-aligned tight/context crops for a second-stage classifier."""
import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


CLASSES = (
    'Small Car', 'Van', 'Dump Truck', 'Cargo Truck', 'other-vehicle',
    'Bus', 'Truck Tractor', 'Excavator', 'Trailer', 'Tractor')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--xml-root', type=Path, required=True)
    parser.add_argument('--split-csv', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--tight-scale', type=float, default=1.2)
    parser.add_argument('--context-scale', type=float, default=2.0)
    return parser.parse_args()


def aligned_crop(image, points, scale):
    (cx, cy), (width, height), angle = cv2.minAreaRect(points)
    if width < height:
        width, height = height, width
        angle += 90
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    size = (max(round(width * scale), 4), max(round(height * scale), 4))
    return cv2.getRectSubPix(rotated, size, (cx, cy))


def main():
    args = parse_args()
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise ImportError('This crop builder requires GDAL') from exc
    metadata = {}
    with args.split_csv.open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            metadata[Path(row['competition_image']).stem] = (
                row['sha256'], row['recommended_role'])

    rows = []
    for image_id, (digest, role) in metadata.items():
        if role == 'drop_exact_duplicate':
            continue
        dataset = gdal.Open(str(args.image_root / f'{image_id}.tif'))
        rgb = np.stack([
            dataset.GetRasterBand(index).ReadAsArray()
            for index in (1, 2, 3)
        ], axis=-1)
        image = np.ascontiguousarray(rgb[..., ::-1])
        root = ET.parse(args.xml_root / f'{image_id}.xml').getroot()
        for object_index, obj in enumerate(root.findall('objects/object')):
            name = (obj.findtext('possibleresult/name') or '').strip()
            if name not in CLASSES:
                continue
            points = []
            for node in obj.findall('points/point')[:4]:
                x, y = (node.text or '').split(',')[:2]
                points.append((float(x), float(y)))
            points = np.asarray(points, dtype=np.float32)
            sample_id = f'{image_id}_{object_index:04d}'
            class_dir = args.output_root / name.replace(' ', '_')
            class_dir.mkdir(parents=True, exist_ok=True)
            tight_path = class_dir / f'{sample_id}_tight.png'
            context_path = class_dir / f'{sample_id}_context.png'
            cv2.imwrite(str(tight_path), aligned_crop(
                image, points, args.tight_scale))
            cv2.imwrite(str(context_path), aligned_crop(
                image, points, args.context_scale))
            rows.append(dict(
                sample_id=sample_id, image_id=image_id, sha256=digest,
                split=role, label=CLASSES.index(name), class_name=name,
                tight_path=str(tight_path), context_path=str(context_path)))

    with (args.output_root / 'manifest.csv').open(
            'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f'wrote {len(rows)} paired crops')


if __name__ == '__main__':
    main()
