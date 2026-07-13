#!/usr/bin/env python
"""Train the dual-crop ConvNeXt-T vehicle refinement classifier."""
import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projects.tianzhibei_car.refiner import DualCropConvNeXtTiny


class CropDataset(Dataset):
    def __init__(self, manifest, role, training):
        with open(manifest, encoding='utf-8', newline='') as file:
            self.rows = [row for row in csv.DictReader(file)
                         if row['split'] == role]
        operations = [transforms.Resize((96, 192))]
        if training:
            operations += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.2)]
        operations += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))]
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tight = Image.open(row['tight_path']).convert('RGB')
        context = Image.open(row['context_path']).convert('RGB')
        return (self.transform(tight), self.transform(context),
                int(row['label']))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    correct = total = 0; loss_sum = 0.0
    for tight, context, label in loader:
        tight, context, label = tight.to(device), context.to(device), label.to(device)
        with torch.set_grad_enabled(training):
            logits = model(tight, context)
            loss = criterion(logits, label)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_sum += loss.item() * label.numel()
        correct += (logits.argmax(1) == label).sum().item()
        total += label.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = CropDataset(args.manifest, 'train', True)
    val = CropDataset(args.manifest, 'validation', False)
    train_loader = DataLoader(
        train, args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True)
    val_loader = DataLoader(
        val, args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True)
    model = DualCropConvNeXtTiny().to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1e-6)
    best = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device)
        scheduler.step()
        print(epoch + 1, train_loss, train_acc, val_loss, val_acc)
        if val_acc > best:
            best = val_acc
            torch.save(dict(
                state_dict=model.state_dict(), epoch=epoch + 1,
                val_accuracy=val_acc), args.output)


if __name__ == '__main__':
    main()
