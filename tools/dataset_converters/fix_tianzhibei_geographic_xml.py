#!/usr/bin/env python
"""Convert geographic Tianzhibei XML polygons to pixel coordinates offline."""

import argparse
import csv
import json
import math
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Copy Tianzhibei XML files to a clean directory and '
        'convert WGS84 polygons to pixel coordinates with each TIFF inverse '
        'GeoTransform.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--xml-dir', default='gt')
    parser.add_argument('--image-dir', default='input_path')
    parser.add_argument('--output-dir', default='gt_pixel')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--max-geographic-span', type=float, default=0.01)
    parser.add_argument('--pixel-tolerance', type=float, default=2.0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def parse_points(obj):
    point_nodes = obj.findall('points/point')
    points = []
    for node in point_nodes:
        text = (node.text or '').strip()
        try:
            x, y = map(float, text.split(',')[:2])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'invalid point {text!r}') from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f'non-finite point {text!r}')
        points.append((x, y))
    return point_nodes, points


def is_geographic_candidate(points, max_span):
    if len(points) < 4:
        return False
    points = points[:4]
    if not all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in points):
        return False
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 1e-10 or height <= 1e-10:
        return False
    if width >= max_span or height >= max_span:
        return False
    return any(
        abs(value - round(value)) > 1e-5
        for point in points for value in point)


def polygon_geometry(points):
    points = points[:4]
    area = 0.0
    sides = []
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - y1 * x2
        sides.append(math.hypot(x2 - x1, y2 - y1))
    return abs(area) * 0.5, min(sides)


def gdal_georeference(path):
    from osgeo import gdal, osr

    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f'GDAL failed to open {path}')
    transform = dataset.GetGeoTransform(can_return_null=True)
    projection = dataset.GetProjectionRef()
    if transform is None or not projection:
        raise ValueError(f'{path} has no GeoTransform/CRS')
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(projection)
    if not spatial_ref.IsGeographic():
        raise ValueError(f'{path} CRS is not geographic')
    inverse = gdal.InvGeoTransform(transform)
    if inverse is None:
        raise ValueError(f'{path} GeoTransform is not invertible')
    if (len(inverse) == 2 and isinstance(inverse[0], (bool, int))):
        if not inverse[0]:
            raise ValueError(f'{path} GeoTransform is not invertible')
        inverse = inverse[1]
    width, height = dataset.RasterXSize, dataset.RasterYSize

    def to_pixel(x, y):
        return gdal.ApplyGeoTransform(inverse, x, y)

    dataset = None
    return to_pixel, width, height, 'gdal'


def rasterio_georeference(path):
    import rasterio

    with rasterio.open(path) as dataset:
        if dataset.crs is None or not dataset.crs.is_geographic:
            raise ValueError(f'{path} CRS is not geographic')
        inverse = ~dataset.transform
        width, height = dataset.width, dataset.height

    def to_pixel(x, y):
        return inverse * (x, y)

    return to_pixel, width, height, 'rasterio'


def select_georeference_reader():
    try:
        from osgeo import gdal  # noqa: F401
        return gdal_georeference, 'gdal'
    except ImportError:
        try:
            import rasterio  # noqa: F401
            return rasterio_georeference, 'rasterio'
        except ImportError as exc:
            raise ImportError(
                'The conversion requires GDAL or rasterio.') from exc


def convert_one(xml_path, image_root, output_root, georef_reader, args):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    candidates = []
    for object_index, obj in enumerate(root.findall('objects/object')):
        point_nodes, points = parse_points(obj)
        if is_geographic_candidate(points, args.max_geographic_span):
            candidates.append((object_index, obj, point_nodes, points))

    output_path = output_root / xml_path.name
    if not candidates:
        shutil.copy2(xml_path, output_path)
        return dict(image_id=xml_path.stem, converted=[], backend='')

    image_path = image_root / f'{xml_path.stem}.tif'
    to_pixel, width, height, backend = georef_reader(image_path)
    converted = []
    for object_index, obj, point_nodes, geographic_points in candidates:
        pixel_points = [to_pixel(x, y) for x, y in geographic_points]
        tolerance = args.pixel_tolerance
        if not all(
                -tolerance <= x <= width + tolerance and
                -tolerance <= y <= height + tolerance
                for x, y in pixel_points):
            continue
        area, short_side = polygon_geometry(pixel_points)
        if area <= 1.0 or short_side < 1.0:
            raise ValueError(
                f'{xml_path.name} object {object_index} converted to invalid '
                f'geometry: area={area}, short_side={short_side}')
        for node, (x, y) in zip(point_nodes, pixel_points):
            node.text = f'{x:.6f},{y:.6f}'
        coordinate_node = obj.find('coordinate')
        if coordinate_node is not None:
            coordinate_node.text = 'pixel'
        class_name = (obj.findtext('possibleresult/name') or '').strip()
        converted.append(dict(
            image_id=xml_path.stem,
            object_index=object_index,
            class_name=class_name,
            area=area,
            short_side=short_side,
            min_x=min(point[0] for point in pixel_points),
            max_x=max(point[0] for point in pixel_points),
            min_y=min(point[1] for point in pixel_points),
            max_y=max(point[1] for point in pixel_points)))

    if candidates and not converted:
        shutil.copy2(xml_path, output_path)
        return dict(image_id=xml_path.stem, converted=[], backend=backend)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    return dict(image_id=xml_path.stem, converted=converted, backend=backend)


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    xml_root = (data_root / args.xml_dir).resolve()
    image_root = (data_root / args.image_dir).resolve()
    output_root = (data_root / args.output_dir).resolve()
    if not xml_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError(
            f'Expected XML and image directories: {xml_root}, {image_root}')
    if output_root == xml_root:
        raise ValueError('Output directory must differ from the source XML directory')
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f'{output_root} already exists; pass --overwrite to regenerate')
    output_root.mkdir(parents=True, exist_ok=True)

    georef_reader, preferred_backend = select_georeference_reader()
    xml_paths = sorted(
        xml_root.glob('*.xml'),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)
    print(
        f'Processing {len(xml_paths)} XML files with {preferred_backend} and '
        f'{args.workers} workers...', flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        iterator = executor.map(
            lambda path: convert_one(
                path, image_root, output_root, georef_reader, args),
            xml_paths,
            chunksize=16)
        for index, result in enumerate(iterator, 1):
            results.append(result)
            if index % 500 == 0 or index == len(xml_paths):
                print(f'  processed {index}/{len(xml_paths)}', flush=True)

    converted_objects = [
        item for result in results for item in result['converted']
    ]
    converted_images = sorted({item['image_id'] for item in converted_objects})
    audit_path = output_root / 'geographic_conversion.csv'
    fields = [
        'image_id', 'object_index', 'class_name', 'area', 'short_side',
        'min_x', 'max_x', 'min_y', 'max_y'
    ]
    with audit_path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(converted_objects)

    manifest = dict(
        source_xml_dir=str(xml_root),
        output_xml_dir=str(output_root),
        xml_files=len(xml_paths),
        converted_images=len(converted_images),
        converted_image_ids=converted_images,
        converted_objects=len(converted_objects),
        preferred_backend=preferred_backend,
        max_geographic_span=args.max_geographic_span,
        pixel_tolerance=args.pixel_tolerance)
    (output_root / 'geographic_conversion_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
