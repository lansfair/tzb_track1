import argparse
import math

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert the FAIR1M MTP detector for 10-class training.')
    parser.add_argument('src', help='FAIR1M MTP detector checkpoint')
    parser.add_argument('dst', help='Output checkpoint')
    parser.add_argument('--target-grid', type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        checkpoint = torch.load(
            args.src, map_location='cpu', mmap=True, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.src, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    drop_prefixes = (
        'roi_head.bbox_head.fc_cls.',
        'rpn_head.rpn_cls.',
        'rpn_head.rpn_reg.',
    )
    converted = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(drop_prefixes)
    }

    pos_key = 'backbone.pos_embed'
    if pos_key in converted:
        pos_embed = converted[pos_key]
        old_grid = int(math.sqrt(pos_embed.shape[1]))
        if old_grid * old_grid != pos_embed.shape[1]:
            raise ValueError(
                f'Unsupported position embedding shape: {pos_embed.shape}')
        if old_grid != args.target_grid:
            pos_embed = pos_embed.reshape(
                1, old_grid, old_grid, pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed,
                size=(args.target_grid, args.target_grid),
                mode='bicubic',
                align_corners=False)
            converted[pos_key] = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

    torch.save(
        {
            'state_dict': converted,
            'meta': {
                'source': args.src,
                'target_num_classes': 10,
                'target_grid': args.target_grid,
                'removed_prefixes': drop_prefixes,
            }
        }, args.dst)
    print(f'kept={len(converted)} removed={len(state_dict) - len(converted)}')
    if pos_key in converted:
        print(f'{pos_key}={tuple(converted[pos_key].shape)}')


if __name__ == '__main__':
    main()
