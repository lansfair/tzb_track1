#!/usr/bin/env python
"""Convert MMRotate ``tools/test.py --out`` results to portable JSON."""
import argparse
import json
from pathlib import Path

from mmengine.fileio import load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tta', default='original')
    parser.add_argument('--scale', type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    for sample in load(args.input):
        instances = sample.pred_instances
        boxes = instances.bboxes.tensor if hasattr(
            instances.bboxes, 'tensor') else instances.bboxes
        image_id = str(sample.metainfo.get(
            'img_id', Path(sample.metainfo['img_path']).stem))
        for box, label, score in zip(
                boxes.detach().cpu().tolist(),
                instances.labels.detach().cpu().tolist(),
                instances.scores.detach().cpu().tolist()):
            records.append(dict(
                image_id=image_id, label=int(label), score=float(score),
                bbox=box, tta=args.tta, scale=args.scale))
    args.output.write_text(
        json.dumps(records, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {len(records)} predictions')


if __name__ == '__main__':
    main()
