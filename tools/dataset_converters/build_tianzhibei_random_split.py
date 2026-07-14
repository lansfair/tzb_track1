#!/usr/bin/env python
"""Validate Tianzhibei samples and build a grouped stratified holdout.

The script fully decodes every raster, validates its paired XML annotation,
groups byte-identical images to prevent leakage, and creates a deterministic
random multilabel-stratified train/validation split.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CLASSES = (
    'Small Car', 'Van', 'Dump Truck', 'Cargo Truck', 'other-vehicle',
    'Bus', 'Truck Tractor', 'Excavator', 'Trailer', 'Tractor')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate Tianzhibei TIFF/XML pairs and create a seeded '
        'multilabel-stratified train/validation split.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--restarts', type=int, default=64)
    parser.add_argument(
        '--swap-iterations', type=int, default=500000,
        help='Randomized train/validation swaps used to refine class balance.')
    parser.add_argument(
        '--reuse-validation', action='store_true',
        help='Reuse output-dir/sample_validation.csv instead of decoding TIFFs.')
    return parser.parse_args()


def numeric_key(value):
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_annotation(path):
    root = ET.parse(path).getroot()
    width = int(float(root.findtext('size/width', '0')))
    height = int(float(root.findtext('size/height', '0')))
    if width <= 0 or height <= 0:
        raise ValueError(f'invalid XML image size {width}x{height}')

    counts = Counter()
    objects = []
    for obj in root.findall('objects/object'):
        class_name = (obj.findtext('possibleresult/name') or '').strip()
        if class_name not in CLASSES:
            raise ValueError(f'unknown class {class_name!r}')
        counts[class_name] += 1
        points = tuple(
            (point.text or '').strip() for point in obj.findall('points/point'))
        objects.append((class_name, points))

    canonical = json.dumps(
        sorted(objects), ensure_ascii=False, separators=(',', ':'))
    annotation_sha256 = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return width, height, counts, annotation_sha256


def decode_with_rasterio(path):
    import rasterio

    with rasterio.open(path) as dataset:
        width, height, bands = dataset.width, dataset.height, dataset.count
        if width <= 0 or height <= 0 or bands < 3:
            raise ValueError(
                f'invalid raster shape {width}x{height} with {bands} bands')
        # One full read still decodes every TIFF strip/tile, while avoiding
        # thousands of Python/GDAL calls for images stored as two-row strips.
        data = dataset.read()
        if data.shape != (bands, height, width) or data.size == 0:
            raise ValueError(
                f'failed full raster decode: got array shape {data.shape}')
    return width, height, bands


def decode_with_gdal(path):
    from osgeo import gdal

    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError('GDAL failed to open image')
    width, height, bands = (
        dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount)
    if width <= 0 or height <= 0 or bands < 3:
        raise ValueError(
            f'invalid raster shape {width}x{height} with {bands} bands')
    block_height = 256
    for y_offset in range(0, height, block_height):
        rows = min(block_height, height - y_offset)
        data = dataset.ReadAsArray(0, y_offset, width, rows)
        if data is None or data.size == 0:
            raise ValueError(f'failed to decode rows {y_offset}:{y_offset + rows}')
    dataset = None
    return width, height, bands


def select_decoder():
    try:
        import rasterio  # noqa: F401
        return decode_with_rasterio, 'rasterio'
    except ImportError:
        try:
            from osgeo import gdal  # noqa: F401
            return decode_with_gdal, 'gdal'
        except ImportError as exc:
            raise ImportError(
                'The validation scan requires rasterio or GDAL.') from exc


def validate_sample(image_id, image_path, xml_path, decoder):
    result = dict(
        image_id=image_id,
        image_path=str(image_path),
        xml_path=str(xml_path),
        status='invalid',
        error='',
        bytes=image_path.stat().st_size if image_path.is_file() else -1,
        sha256='',
        annotation_sha256='',
        width=0,
        height=0,
        bands=0,
        total_instances=0,
        class_counts={name: 0 for name in CLASSES})
    errors = []
    if not image_path.is_file():
        errors.append('missing_image')
    elif result['bytes'] == 0:
        errors.append('empty_image')
    if not xml_path.is_file():
        errors.append('missing_xml')
    if errors:
        result['error'] = ';'.join(errors)
        return result

    try:
        xml_width, xml_height, counts, annotation_hash = parse_annotation(
            xml_path)
        width, height, bands = decoder(image_path)
        if (width, height) != (xml_width, xml_height):
            raise ValueError(
                f'image/XML size mismatch: image={width}x{height}, '
                f'XML={xml_width}x{xml_height}')
        result.update(
            status='valid',
            sha256=sha256_file(image_path),
            annotation_sha256=annotation_hash,
            width=width,
            height=height,
            bands=bands,
            total_instances=sum(counts.values()),
            class_counts={name: counts[name] for name in CLASSES})
    except Exception as exc:  # Keep scanning after a broken sample.
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def feature_vector(record):
    counts = record['class_counts']
    features = [float(counts[name] > 0) for name in CLASSES]
    features.extend(float(counts[name]) for name in CLASSES)

    longest_side = max(record['width'], record['height'])
    size_bin = 0 if longest_side <= 800 else 1 if longest_side <= 1024 else 2
    features.extend(float(size_bin == index) for index in range(3))

    total = record['total_instances']
    density_bin = 0 if total < 32 else 1 if total < 118 else 2
    features.extend(float(density_bin == index) for index in range(3))
    return features


def holdout_score(val_sum, global_sum, val_ratio):
    """Balance class presence, instances, image sizes, and densities."""
    score = 0.0
    weights = ([2.0] * len(CLASSES) + [1.0] * len(CLASSES) +
               [0.25] * (len(global_sum) - 2 * len(CLASSES)))
    for actual_val, total, weight in zip(val_sum, global_sum, weights):
        target_val = total * val_ratio
        target_train = total - target_val
        actual_train = total - actual_val
        score += weight * (
            ((actual_val - target_val) / max(target_val, 1.0))**2 +
            ((actual_train - target_train) / max(target_train, 1.0))**2)
    return score


def stratified_holdout(records, val_ratio, seed, restarts, swap_iterations):
    if not 0 < val_ratio < 1:
        raise ValueError('--val-ratio must be between 0 and 1')
    if len(records) < 2:
        raise ValueError('At least two valid unique images are required')

    for record in records:
        record['features'] = feature_vector(record)
    dimension = len(records[0]['features'])
    global_sum = [
        sum(record['features'][index] for record in records)
        for index in range(dimension)
    ]
    capacities = [len(records) - round(len(records) * val_ratio),
                  round(len(records) * val_ratio)]
    ratios = [capacities[0] / len(records), capacities[1] / len(records)]
    targets = [[value * ratio for value in global_sum] for ratio in ratios]
    presence_totals = global_sum[:len(CLASSES)]

    best = None
    for restart in range(max(restarts, 1)):
        rng = random.Random(seed + restart * 1000003)
        ordered = list(records)
        rng.shuffle(ordered)
        ordered.sort(
            key=lambda record: sum(
                record['features'][index] / max(presence_totals[index], 1.0)
                for index in range(len(CLASSES))),
            reverse=True)

        assignments = [[], []]  # train, validation
        sums = [[0.0] * dimension for _ in range(2)]
        for record in ordered:
            candidates = []
            for fold in range(2):
                if len(assignments[fold]) >= capacities[fold]:
                    continue
                delta = 0.0
                for index, value in enumerate(record['features']):
                    target = targets[fold][index]
                    scale = max(target, 1.0)
                    before = (sums[fold][index] - target) / scale
                    after = (sums[fold][index] + value - target) / scale
                    delta += after * after - before * before
                size_target = capacities[fold]
                size_before = (
                    (len(assignments[fold]) - size_target) / size_target)
                size_after = (
                    (len(assignments[fold]) + 1 - size_target) / size_target)
                delta += 0.25 * (size_after * size_after - size_before * size_before)
                candidates.append((delta, rng.random(), fold))
            fold = min(candidates)[2]
            assignments[fold].append(record)
            sums[fold] = [
                old + value
                for old, value in zip(sums[fold], record['features'])
            ]

        missing = 0
        for class_index, total in enumerate(presence_totals):
            if total >= 2:
                missing += int(sums[0][class_index] == 0)
                missing += int(sums[1][class_index] == 0)
        score = holdout_score(sums[1], global_sum, val_ratio) + missing * 1e9
        if best is None or score < best[0]:
            best = (score, assignments, sums)

    score, assignments, sums = best
    train, validation = assignments
    val_sum = sums[1]
    rng = random.Random(seed + 982451653)
    initial_temperature = max(score * 0.002, 1e-8)
    for iteration in range(max(swap_iterations, 0)):
        val_index = rng.randrange(len(validation))
        train_index = rng.randrange(len(train))
        val_record = validation[val_index]
        train_record = train[train_index]
        candidate_sum = [
            current - old + new
            for current, old, new in zip(
                val_sum, val_record['features'], train_record['features'])
        ]
        candidate_score = holdout_score(
            candidate_sum, global_sum, val_ratio)
        progress = iteration / max(swap_iterations - 1, 1)
        temperature = initial_temperature * (1.0 - progress)**3
        accept = candidate_score < score
        if not accept and temperature > 0:
            accept = rng.random() < math.exp(
                max((score - candidate_score) / temperature, -60.0))
        if accept:
            validation[val_index], train[train_index] = (
                train_record, val_record)
            val_sum = candidate_sum
            score = candidate_score

    sums = [
        [total - value for total, value in zip(global_sum, val_sum)],
        val_sum
    ]
    for class_index, class_name in enumerate(CLASSES):
        if presence_totals[class_index] >= 2:
            if sums[0][class_index] == 0 or sums[1][class_index] == 0:
                raise RuntimeError(
                    f'Failed to place class {class_name!r} in both splits')
    return train, validation


def read_validation_csv(path):
    results = []
    with path.open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            results.append(dict(
                image_id=row['image_id'],
                status=row['status'],
                error=row['error'],
                bytes=int(row['bytes']),
                sha256=row['sha256'],
                annotation_sha256=row['annotation_sha256'],
                width=int(row['width']),
                height=int(row['height']),
                bands=int(row['bands']),
                total_instances=int(row['total_instances']),
                class_counts={
                    name: int(row[f'{name}_instances']) for name in CLASSES
                }))
    return results


def aggregate(records):
    image_counts = Counter()
    instance_counts = Counter()
    for record in records:
        for class_name in CLASSES:
            count = record['class_counts'][class_name]
            image_counts[class_name] += int(count > 0)
            instance_counts[class_name] += count
    return image_counts, instance_counts


def write_ids(path, records):
    ids = sorted((record['image_id'] for record in records), key=numeric_key)
    path.write_text(''.join(f'{image_id}\n' for image_id in ids), encoding='utf-8')
    return ids


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    image_root = data_root / 'input_path'
    xml_root = data_root / 'gt'
    output_dir = (args.output_dir or data_root / 'splits').resolve()
    if not image_root.is_dir() or not xml_root.is_dir():
        raise FileNotFoundError(
            f'Expected {image_root} and {xml_root}')

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_csv = output_dir / 'sample_validation.csv'
    if args.reuse_validation:
        if not validation_csv.is_file():
            raise FileNotFoundError(validation_csv)
        results = read_validation_csv(validation_csv)
        all_ids = [record['image_id'] for record in results]
        decoder_name = 'reused_validation_csv'
        print(f'Reusing {validation_csv}', flush=True)
    else:
        decoder, decoder_name = select_decoder()
        image_ids = {path.stem for path in image_root.glob('*.tif')}
        xml_ids = {path.stem for path in xml_root.glob('*.xml')}
        all_ids = sorted(image_ids | xml_ids, key=numeric_key)
        if not all_ids:
            raise RuntimeError(f'No TIFF/XML samples found below {data_root}')

        print(
            f'Validating {len(all_ids)} samples with {decoder_name} and '
            f'{args.workers} workers...', flush=True)
        results = []
        with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
            futures = {
                executor.submit(
                    validate_sample, image_id,
                    image_root / f'{image_id}.tif',
                    xml_root / f'{image_id}.xml', decoder): image_id
                for image_id in all_ids
            }
            for index, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if index % 250 == 0 or index == len(futures):
                    print(f'  validated {index}/{len(futures)}', flush=True)
        results.sort(key=lambda record: numeric_key(record['image_id']))

        fieldnames = [
            'image_id', 'status', 'error', 'bytes', 'sha256',
            'annotation_sha256', 'width', 'height', 'bands', 'total_instances'
        ] + [f'{name}_instances' for name in CLASSES]
        with validation_csv.open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for record in results:
                row = {name: record[name] for name in fieldnames[:10]}
                row.update({
                    f'{name}_instances': record['class_counts'][name]
                    for name in CLASSES
                })
                writer.writerow(row)

    invalid = [record for record in results if record['status'] != 'valid']
    valid = [record for record in results if record['status'] == 'valid']
    groups = defaultdict(list)
    for record in valid:
        groups[record['sha256']].append(record)

    unique = []
    duplicates = []
    duplicate_rows = []
    for digest, group in groups.items():
        group.sort(
            key=lambda record: (-record['total_instances'],
                                numeric_key(record['image_id'])))
        representative = group[0]
        unique.append(representative)
        for duplicate in group[1:]:
            duplicates.append(duplicate)
            duplicate_rows.append(dict(
                duplicate_image_id=duplicate['image_id'],
                representative_image_id=representative['image_id'],
                sha256=digest,
                duplicate_total_instances=duplicate['total_instances'],
                representative_total_instances=(
                    representative['total_instances']),
                class_counts_equal=(
                    duplicate['class_counts'] ==
                    representative['class_counts']),
                annotation_equal=(
                    duplicate['annotation_sha256'] ==
                    representative['annotation_sha256'])))

    train, validation = stratified_holdout(
        unique, args.val_ratio, args.seed, args.restarts,
        args.swap_iterations)
    train_ids = write_ids(output_dir / 'train.txt', train)
    val_ids = write_ids(output_dir / 'val.txt', validation)
    write_ids(output_dir / 'all_unique.txt', train + validation)
    write_ids(output_dir / 'drop_exact_duplicate.txt', duplicates)
    write_ids(output_dir / 'drop_invalid.txt', invalid)

    duplicate_csv = output_dir / 'duplicate_groups.csv'
    with duplicate_csv.open('w', newline='', encoding='utf-8-sig') as file:
        fields = [
            'duplicate_image_id', 'representative_image_id', 'sha256',
            'duplicate_total_instances', 'representative_total_instances',
            'class_counts_equal', 'annotation_equal'
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(duplicate_rows)

    train_images, train_instances = aggregate(train)
    val_images, val_instances = aggregate(validation)
    all_images, all_instances = aggregate(unique)
    class_summary = {}
    for class_name in CLASSES:
        class_summary[class_name] = dict(
            all_images=all_images[class_name],
            train_images=train_images[class_name],
            val_images=val_images[class_name],
            val_image_ratio=(
                val_images[class_name] / all_images[class_name]
                if all_images[class_name] else 0.0),
            all_instances=all_instances[class_name],
            train_instances=train_instances[class_name],
            val_instances=val_instances[class_name],
            val_instance_ratio=(
                val_instances[class_name] / all_instances[class_name]
                if all_instances[class_name] else 0.0))

    manifest = dict(
        data_root=str(data_root),
        decoder=decoder_name,
        seed=args.seed,
        requested_val_ratio=args.val_ratio,
        total_pairs=len(all_ids),
        valid_images=len(valid),
        invalid_images=len(invalid),
        exact_duplicates_dropped=len(duplicates),
        duplicate_class_count_mismatches=sum(
            not row['class_counts_equal'] for row in duplicate_rows),
        duplicate_annotation_mismatches=sum(
            not row['annotation_equal'] for row in duplicate_rows),
        unique_images=len(unique),
        train_images=len(train_ids),
        val_images=len(val_ids),
        actual_val_ratio=len(val_ids) / len(unique),
        invalid_samples=[dict(
            image_id=record['image_id'], error=record['error'])
                         for record in invalid],
        class_summary=class_summary)
    (output_dir / 'split_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
