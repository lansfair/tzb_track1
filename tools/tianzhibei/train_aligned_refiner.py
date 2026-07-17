"""Train the direction-aligned dual-crop vehicle classifier."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from torch.utils.data import DataLoader, WeightedRandomSampler

from projects.tianzhibei_car.refiner import (VEHICLE_CLASSES,
                                             AlignedVehicleCropDataset,
                                             DualCropConvNeXtTiny)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--work-dir')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_balanced_sampler(labels, power):
    counts = Counter(labels)
    weights = [counts[label]**(-power) for label in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros(10, 10, dtype=torch.long)
    total_loss = 0.0
    total = 0
    for tight, context, labels, _ in loader:
        tight, context = tight.to(device), context.to(device)
        labels = labels.to(device)
        logits = model(tight, context)
        total_loss += F.cross_entropy(logits, labels, reduction='sum').item()
        predictions = logits.argmax(1)
        indices = (labels.cpu() * 10 + predictions.cpu()).long()
        confusion += torch.bincount(indices, minlength=100).reshape(10, 10)
        total += labels.numel()
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(0).clamp_min(1)
    recall = true_positive / confusion.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        'loss': total_loss / max(total, 1),
        'accuracy': true_positive.sum().item() / max(total, 1),
        'macro_f1': f1.mean().item(),
        'class_f1': dict(zip(VEHICLE_CLASSES, f1.tolist())),
        'confusion_matrix': confusion.tolist()
    }


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    work_dir = Path(args.work_dir or cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = AlignedVehicleCropDataset(
        cfg.train_manifest, training=True)
    val_dataset = AlignedVehicleCropDataset(cfg.val_manifest, training=False)
    sampler = class_balanced_sampler(train_dataset.labels, cfg.sampler_power)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0)

    model = DualCropConvNeXtTiny(10, pretrained=cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.amp and device.type == 'cuda')
    best_f1 = -1.0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for tight, context, labels, _ in train_loader:
            tight, context = tight.to(device), context.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                    enabled=scaler.is_enabled(), dtype=torch.float16):
                logits = model(tight, context)
                loss = F.cross_entropy(
                    logits, labels, label_smoothing=cfg.label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.numel()
            sample_count += labels.numel()
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        metrics.update(
            epoch=epoch,
            train_loss=running_loss / max(sample_count, 1),
            lr=optimizer.param_groups[0]['lr'])
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        checkpoint = {
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'classes': VEHICLE_CLASSES,
            'config': cfg.pretty_text
        }
        torch.save(checkpoint, work_dir / 'last.pth')
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            torch.save(checkpoint, work_dir / 'best_macro_f1.pth')
    (work_dir / 'history.json').write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
