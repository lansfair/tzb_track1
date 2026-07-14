#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract, validate, deduplicate, and stratify Tianzhibei.')
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--tar', default='tar')
    parser.add_argument('--unrar', default='unrar')
    parser.add_argument('--skip-extract', action='store_true')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--restarts', type=int, default=64)
    parser.add_argument('--swap-iterations', type=int, default=500000)
    parser.add_argument('--reuse-validation', action='store_true')
    parser.add_argument('--overwrite-fixed-xml', action='store_true')
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
    if not (data_root / 'input_path').is_dir() or not (data_root / 'gt').is_dir():
        raise FileNotFoundError(
            f'Expected input_path and gt below {data_root} after extraction.')

    fixed_xml_root = data_root / 'gt_pixel'
    fixed_manifest = fixed_xml_root / 'geographic_conversion_manifest.json'
    if args.overwrite_fixed_xml or not fixed_manifest.is_file():
        xml_fixer = Path(__file__).with_name(
            'fix_tianzhibei_geographic_xml.py')
        fix_command = [
            sys.executable, str(xml_fixer), '--data-root', str(data_root),
            '--output-dir', 'gt_pixel', '--workers', str(args.workers)
        ]
        if fixed_xml_root.exists():
            fix_command.append('--overwrite')
        subprocess.run(fix_command, check=True)
    else:
        print(f'Reusing corrected XML directory: {fixed_xml_root}')

    split_builder = Path(__file__).with_name(
        'build_tianzhibei_random_split.py')
    command = [
        sys.executable, str(split_builder), '--data-root', str(data_root),
        '--output-dir', str(data_root / 'splits'), '--val-ratio',
        str(args.val_ratio), '--seed', str(args.seed), '--workers',
        str(args.workers), '--restarts', str(args.restarts),
        '--swap-iterations', str(args.swap_iterations)
    ]
    if args.reuse_validation:
        command.append('--reuse-validation')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
