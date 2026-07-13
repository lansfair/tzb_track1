"""Prepare a FAIR1M MTP detector checkpoint for the 10-class task.

The FAIR1M checkpoint has a 38-output classifier (37 foreground classes plus
background).  Removing only that classifier lets MMEngine load the backbone,
FPN, RPN, class-agnostic box regressor and RoI layers into the 10-class model.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F


INCOMPATIBLE_PREFIXES = (
    'roi_head.bbox_head.fc_cls.',
    'module.roi_head.bbox_head.fc_cls.',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Original 37-class MTP checkpoint')
    parser.add_argument('output', help='Output initialization checkpoint')
    parser.add_argument(
        '--target-grid',
        type=int,
        default=64,
        help='Target square token grid; 64 corresponds to 1024/16')
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)

    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise TypeError('Expected a checkpoint containing a state_dict')

    state_dict = checkpoint['state_dict']
    removed = [
        key for key in state_dict
        if key.startswith(INCOMPATIBLE_PREFIXES)
    ]
    if not removed:
        raise KeyError('No FAIR1M fc_cls tensors were found in the checkpoint')

    converted_state_dict = {
        key: value for key, value in state_dict.items() if key not in removed
    }
    pos_key = 'backbone.pos_embed'
    if pos_key not in converted_state_dict:
        raise KeyError(f'Missing required tensor: {pos_key}')
    pos_embed = converted_state_dict[pos_key]
    source_grid = int(math.sqrt(pos_embed.shape[1]))
    if source_grid * source_grid != pos_embed.shape[1]:
        raise ValueError(
            f'Position embedding has a non-square token count: '
            f'{pos_embed.shape[1]}')
    if source_grid != args.target_grid:
        pos_tokens = pos_embed.reshape(
            1, source_grid, source_grid, pos_embed.shape[-1])
        pos_tokens = pos_tokens.permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(
            pos_tokens,
            size=(args.target_grid, args.target_grid),
            mode='bicubic',
            align_corners=False)
        converted_state_dict[pos_key] = pos_tokens.permute(
            0, 2, 3, 1).flatten(1, 2).contiguous()

    output = {
        'state_dict': converted_state_dict,
        'meta': {
            'source_checkpoint': str(source.resolve()),
            'purpose': '10-class Tianzhibei initialization',
            'removed_keys': removed,
            'position_grid': {
                'source': source_grid,
                'target': args.target_grid,
            },
        }
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)

    print(f'Wrote: {destination}')
    print(f'Kept tensors: {len(converted_state_dict)}')
    print(f'Removed tensors: {removed}')
    print(f'Position grid: {source_grid}x{source_grid} -> '
          f'{args.target_grid}x{args.target_grid}')


if __name__ == '__main__':
    main()
