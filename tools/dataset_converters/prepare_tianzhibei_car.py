#!/usr/bin/env python
import argparse
import csv
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract Tianzhibei data and build hash-safe split files.')
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--split-csv', type=Path, required=True)
    parser.add_argument('--tar', default='tar')
    parser.add_argument('--unrar', default='unrar')
    parser.add_argument('--skip-extract', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_extract:
        if args.archive.suffix.lower() == '.rar':
            subprocess.run(
                [
                    args.unrar, 'x', '-o+', '-idq', str(args.archive),
                    f'{args.output_root}/'
                ],
                check=True)
        else:
            subprocess.run(
                [
                    args.tar, '-xf', str(args.archive), '-C',
                    str(args.output_root)
                ],
                check=True)

    data_root = args.output_root / 'car_det_train'
    image_root = data_root / 'input_path'
    xml_root = data_root / 'gt'
    if not image_root.is_dir() or not xml_root.is_dir():
        raise FileNotFoundError(
            f'Expected {image_root} and {xml_root} after extraction.')

    roles = {'train': [], 'validation': [], 'drop_exact_duplicate': []}
    with args.split_csv.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            role = row['recommended_role']
            if role not in roles:
                continue
            image_id = Path(row['competition_image']).stem
            if not (image_root / f'{image_id}.tif').is_file():
                raise FileNotFoundError(image_root / f'{image_id}.tif')
            if not (xml_root / f'{image_id}.xml').is_file():
                raise FileNotFoundError(xml_root / f'{image_id}.xml')
            roles[role].append(image_id)

    split_root = data_root / 'splits'
    split_root.mkdir(exist_ok=True)
    for role, image_ids in roles.items():
        name = 'val' if role == 'validation' else role
        (split_root / f'{name}.txt').write_text(
            ''.join(f'{image_id}\n' for image_id in image_ids),
            encoding='utf-8')
    all_ids = roles['train'] + roles['validation']
    (split_root / 'all_unique.txt').write_text(
        ''.join(f'{image_id}\n' for image_id in all_ids), encoding='utf-8')

    print(f'data_root={data_root}')
    for role, image_ids in roles.items():
        print(f'{role}={len(image_ids)}')
    print(f'all_unique={len(all_ids)}')


if __name__ == '__main__':
    main()
