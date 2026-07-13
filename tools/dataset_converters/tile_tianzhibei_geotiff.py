#!/usr/bin/env python
"""Create overlapping GeoTIFF/XML tiles while preserving georeferencing."""
import argparse
import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--xml-root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=256)
    parser.add_argument('--min-visible-ratio', type=float, default=0.5)
    parser.add_argument('--copy-small', action='store_true')
    return parser.parse_args()


def starts(length, size, overlap):
    if length <= size:
        return [0]
    result = list(range(0, length - size + 1, size - overlap))
    if result[-1] != length - size:
        result.append(length - size)
    return result


def object_points(obj):
    result = []
    for node in obj.findall('points/point')[:4]:
        x, y = (node.text or '').split(',')[:2]
        result.append((float(x), float(y)))
    return np.asarray(result, dtype=np.float32)


def clipped_box(points, x0, y0, size, min_visible_ratio):
    shifted = points - np.asarray([x0, y0], dtype=np.float32)
    tile = np.asarray(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32)
    source = cv2.convexHull(shifted).reshape(-1, 2)
    original_area = abs(cv2.contourArea(source))
    area, intersection = cv2.intersectConvexConvex(source, tile)
    if (intersection is None or len(intersection) < 3 or
            area / max(original_area, 1e-6) < min_visible_ratio):
        return None
    return cv2.boxPoints(
        cv2.minAreaRect(intersection.reshape(-1, 2))).astype(np.float32)


def replace_points(obj, points):
    parent = obj.find('points')
    nodes = parent.findall('point')
    values = list(points) + [points[0]]
    for index, point in enumerate(values):
        if index >= len(nodes):
            nodes.append(ET.SubElement(parent, 'point'))
        nodes[index].text = f'{point[0]:.2f},{point[1]:.2f}'
    for node in nodes[len(values):]:
        parent.remove(node)


def main():
    args = parse_args()
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise ImportError('This converter requires GDAL Python bindings') from exc
    image_out = args.output_root / 'input_path'
    xml_out = args.output_root / 'gt'
    split_out = args.output_root / 'splits'
    for path in (image_out, xml_out, split_out):
        path.mkdir(parents=True, exist_ok=True)

    output_ids = []
    image_ids = args.split.read_text(
        encoding='utf-8-sig').splitlines()
    for image_id in filter(None, map(str.strip, image_ids)):
        image_path = args.image_root / f'{image_id}.tif'
        xml_path = args.xml_root / f'{image_id}.xml'
        root = ET.parse(xml_path).getroot()
        width = int(float(root.findtext('size/width')))
        height = int(float(root.findtext('size/height')))
        if max(width, height) <= args.size and args.copy_small:
            shutil.copy2(image_path, image_out / image_path.name)
            shutil.copy2(xml_path, xml_out / xml_path.name)
            output_ids.append(image_id)
            continue
        if max(width, height) <= args.size:
            continue

        for y0 in starts(height, args.size, args.overlap):
            for x0 in starts(width, args.size, args.overlap):
                tile_id = f'{image_id}__x{x0}_y{y0}'
                tile_width = min(args.size, width - x0)
                tile_height = min(args.size, height - y0)
                gdal.Translate(
                    str(image_out / f'{tile_id}.tif'),
                    str(image_path),
                    srcWin=[x0, y0, tile_width, tile_height],
                    creationOptions=['COMPRESS=LZW', 'TILED=YES'])

                tile_root = copy.deepcopy(root)
                tile_root.find('size/width').text = str(tile_width)
                tile_root.find('size/height').text = str(tile_height)
                objects = tile_root.find('objects')
                for obj in list(objects.findall('object')):
                    box = clipped_box(
                        object_points(obj), x0, y0, args.size,
                        args.min_visible_ratio)
                    if box is None:
                        objects.remove(obj)
                    else:
                        replace_points(obj, box)
                ET.ElementTree(tile_root).write(
                    xml_out / f'{tile_id}.xml',
                    encoding='utf-8', xml_declaration=True)
                output_ids.append(tile_id)

    (split_out / 'tiled.txt').write_text(
        ''.join(f'{image_id}\n' for image_id in output_ids), encoding='utf-8')
    print(f'wrote {len(output_ids)} images to {args.output_root}')


if __name__ == '__main__':
    main()
