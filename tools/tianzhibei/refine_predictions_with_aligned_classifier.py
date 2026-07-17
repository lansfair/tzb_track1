"""Apply the aligned classifier to ambiguous groups in predictions.pkl."""

import argparse
import pickle
from pathlib import Path

import torch
from mmcv.ops import nms_rotated
from torchvision.transforms import functional as TF

from projects.tianzhibei_car.refiner import (AMBIGUOUS_CLASS_GROUPS,
                                             DualCropConvNeXtTiny,
                                             extract_dual_aligned_crops,
                                             read_geotiff_rgb,
                                             restricted_group_prediction)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('checkpoint')
    parser.add_argument('output')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--det-score-min', type=float, default=0.1)
    parser.add_argument('--classifier-confidence', type=float, default=0.6)
    parser.add_argument('--classifier-margin', type=float, default=0.15)
    parser.add_argument('--detector-weight', type=float, default=0.6)
    parser.add_argument('--nms-iou', type=float, default=0.1)
    parser.add_argument('--max-per-img', type=int, default=1000)
    parser.add_argument(
        '--image-root',
        help='Optional TIFF directory overriding img_path stored in the pkl')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def normalize_crop(crop):
    tensor = TF.to_tensor(crop)
    return TF.normalize(tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def ambiguous_mask(labels):
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for group in AMBIGUOUS_CLASS_GROUPS:
        for label in group:
            mask |= labels == label
    return mask


@torch.no_grad()
def classify(model, tight_crops, context_crops, batch_size, device):
    probabilities = []
    for begin in range(0, len(tight_crops), batch_size):
        tight = torch.stack(tight_crops[begin:begin + batch_size]).to(device)
        context = torch.stack(context_crops[begin:begin +
                                            batch_size]).to(device)
        probabilities.append(model(tight, context).softmax(-1).cpu())
    return torch.cat(probabilities) if probabilities else torch.empty(0, 10)


@torch.no_grad()
def classwise_rotated_nms(instances, iou_threshold, max_per_img, device):
    """Re-run NMS because relabelling can create new same-class overlaps."""
    bboxes = instances['bboxes']
    scores = instances['scores']
    labels = instances['labels']
    kept = []
    for label in labels.unique(sorted=True):
        class_indices = (labels == label).nonzero(as_tuple=False).squeeze(1)
        _, local_keep = nms_rotated(bboxes[class_indices].to(device),
                                    scores[class_indices].to(device),
                                    iou_threshold)
        kept.append(class_indices[local_keep.cpu()])
    if not kept:
        return instances
    kept = torch.cat(kept)
    kept = kept[scores[kept].argsort(descending=True)[:max_per_img]]
    for key, value in list(instances.items()):
        if isinstance(value, torch.Tensor) and value.shape[:1] == labels.shape:
            instances[key] = value[kept.to(value.device)]
    return instances


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model = DualCropConvNeXtTiny(10, pretrained=False)
    model.load_state_dict(
        checkpoint.get('state_dict', checkpoint), strict=True)
    model.to(device).eval()
    with open(args.input, 'rb') as f:
        samples = pickle.load(f)

    changed_total, considered_total = 0, 0
    for sample in samples:
        instances = sample['pred_instances']
        labels = instances['labels'].cpu()
        scores = instances['scores'].cpu()
        selected = ambiguous_mask(labels) & (scores >= args.det_score_min)
        indices = selected.nonzero(as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            continue
        image_path = sample['img_path']
        if args.image_root:
            image_path = Path(args.image_root) / Path(image_path).name
        image = read_geotiff_rgb(image_path)
        bboxes = instances['bboxes'].cpu()
        tight_crops, context_crops = [], []
        for index in indices.tolist():
            tight, context = extract_dual_aligned_crops(
                image, bboxes[index].tolist())
            tight_crops.append(normalize_crop(tight))
            context_crops.append(normalize_crop(context))
        probabilities = classify(model, tight_crops, context_crops,
                                 args.batch_size, device)
        new_labels, accepted = restricted_group_prediction(
            probabilities,
            labels[indices],
            min_confidence=args.classifier_confidence,
            min_margin=args.classifier_margin)
        changed = accepted & (new_labels != labels[indices])
        destination = indices[accepted]
        labels[destination] = new_labels[accepted]
        classifier_score = probabilities[accepted, new_labels[accepted]]
        detector_score = scores[destination]
        scores[destination] = (
            detector_score.clamp_min(1e-8)**args.detector_weight *
            classifier_score.clamp_min(1e-8)**(1 - args.detector_weight))
        instances['labels'] = labels.to(instances['labels'].device)
        instances['scores'] = scores.to(instances['scores'].device)
        sample['pred_instances'] = classwise_rotated_nms(
            instances, args.nms_iou, args.max_per_img, device)
        considered_total += indices.numel()
        changed_total += changed.sum().item()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'considered={considered_total}, relabelled={changed_total}, '
          f'output={output}')


if __name__ == '__main__':
    main()
