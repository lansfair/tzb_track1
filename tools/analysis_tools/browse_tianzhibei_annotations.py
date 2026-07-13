#!/usr/bin/env python
"""Render cleaned qboxes and normalized long-axis directions for inspection."""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--xml-root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-images', type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise ImportError('This browser requires GDAL') from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = [item.strip() for item in args.split.read_text(
        encoding='utf-8-sig').splitlines() if item.strip()]
    for image_id in ids[:args.max_images]:
        dataset = gdal.Open(str(args.image_root / f'{image_id}.tif'))
        rgb = np.stack([
            dataset.GetRasterBand(index).ReadAsArray()
            for index in (1, 2, 3)
        ], axis=-1)
        image = np.ascontiguousarray(rgb[..., ::-1])
        root = ET.parse(args.xml_root / f'{image_id}.xml').getroot()
        for obj in root.findall('objects/object'):
            points = []
            for node in obj.findall('points/point')[:4]:
                x, y = (node.text or '').split(',')[:2]
                points.append((float(x), float(y)))
            points = np.asarray(points, dtype=np.float32)
            cv2.polylines(image, [points.astype(int)], True, (0, 255, 0), 1)
            (cx, cy), (width, height), angle = cv2.minAreaRect(points)
            if width < height:
                width, height = height, width; angle += 90
            angle = ((angle + 90) % 180) - 90
            radians = math.radians(angle)
            endpoint = (round(cx + width / 2 * math.cos(radians)),
                        round(cy + width / 2 * math.sin(radians)))
            cv2.arrowedLine(
                image, (round(cx), round(cy)), endpoint, (0, 0, 255), 1,
                tipLength=0.25)
        cv2.imwrite(str(args.output_dir / f'{image_id}.jpg'), image)
    print(f'wrote {min(len(ids), args.max_images)} visualizations')


if __name__ == '__main__':
    main()
