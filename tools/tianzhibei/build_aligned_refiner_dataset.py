"""Build direction-aligned tight/context vehicle crops from corrected XML."""

import argparse
import csv
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from projects.tianzhibei_car.refiner import (VEHICLE_CLASSES,
                                             extract_dual_aligned_crops,
                                             qbox_to_rbox, read_geotiff_rgb)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--train-split', required=True)
    parser.add_argument('--val-split', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--img-dir', default='input_path')
    parser.add_argument('--ann-dir', default='gt_pixel')
    parser.add_argument('--img-suffix', default='.tif')
    parser.add_argument('--context-scale', type=float, default=1.2)
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


def read_ids(path):
    with open(path, encoding='utf-8-sig') as f:
        return [Path(line.strip()).stem for line in f if line.strip()]


def parse_objects(xml_path):
    class_to_label = {name: i for i, name in enumerate(VEHICLE_CLASSES)}
    root = ET.parse(xml_path).getroot()
    objects = []
    for index, obj in enumerate(root.findall('objects/object')):
        name = (obj.findtext('possibleresult/name', '') or '').strip()
        if name not in class_to_label:
            continue
        points = []
        for node in obj.findall('points/point')[:4]:
            fields = (node.text or '').split(',')
            if len(fields) < 2:
                break
            points.append((float(fields[0]), float(fields[1])))
        if len(points) != 4:
            continue
        points = np.asarray(points, dtype=np.float32)
        if not np.isfinite(points).all() or len(np.unique(points, axis=0)) < 4:
            continue
        objects.append((index, name, class_to_label[name], points))
    return objects


def process_image(job):
    (image_id, split_name, data_root, output_dir, img_dir, ann_dir, img_suffix,
     context_scale) = job
    data_root, output_dir = Path(data_root), Path(output_dir)
    image_path = data_root / img_dir / f'{image_id}{img_suffix}'
    xml_path = data_root / ann_dir / f'{image_id}.xml'
    try:
        image = read_geotiff_rgb(image_path)
        objects = parse_objects(xml_path)
        rows = []
        crop_root = output_dir / 'crops' / split_name
        crop_root.mkdir(parents=True, exist_ok=True)
        for object_index, class_name, label, points in objects:
            rbox = qbox_to_rbox(points)
            if rbox[2] < 1 or rbox[3] < 1:
                continue
            tight, context = extract_dual_aligned_crops(
                image, rbox, context_scale=context_scale)
            stem = f'{image_id}_{object_index:04d}'
            tight_path = crop_root / f'{stem}_tight.png'
            context_path = crop_root / f'{stem}_context.png'
            cv2.imwrite(str(tight_path), tight[..., ::-1])
            cv2.imwrite(str(context_path), context[..., ::-1])
            rows.append({
                'image_id':
                image_id,
                'object_index':
                object_index,
                'class_name':
                class_name,
                'label':
                label,
                'tight_path':
                tight_path.relative_to(output_dir).as_posix(),
                'context_path':
                context_path.relative_to(output_dir).as_posix()
            })
        return rows, None
    except Exception as exc:  # keep a complete build manifest
        return [], f'{image_id}\t{type(exc).__name__}: {exc}'


def build_split(args, split_name, split_path):
    image_ids = read_ids(split_path)
    jobs = [(image_id, split_name, args.data_root, args.output_dir,
             args.img_dir, args.ann_dir, args.img_suffix, args.context_scale)
            for image_id in image_ids]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(process_image, jobs, chunksize=4))
    else:
        results = [process_image(job) for job in jobs]
    rows = [row for image_rows, _ in results for row in image_rows]
    errors = [error for _, error in results if error]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / f'{split_name}.csv'
    fields = ('image_id', 'object_index', 'class_name', 'label', 'tight_path',
              'context_path')
    with manifest.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if errors:
        (output_dir / f'{split_name}_errors.txt').write_text(
            '\n'.join(errors), encoding='utf-8')
    print(f'{split_name}: {len(image_ids)} images, {len(rows)} crops, '
          f'{len(errors)} errors -> {manifest}')


def main():
    args = parse_args()
    build_split(args, 'train', args.train_split)
    build_split(args, 'val', args.val_split)


if __name__ == '__main__':
    main()
