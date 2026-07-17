# Tianzhibei fine-grained controlled experiments

These experiments start from the existing **MTP ViT-L+RVSA SmoothL1 epoch
35 checkpoint**.  They are deliberately independent; do not stack E1 and E2
until each has been compared with the unchanged checkpoint under the same
validation split and IoU=0.3 macro-F1 evaluator.

## E1: decoupled classification residual + ProtoCL

Config: `mtp-vit-l-rvsa_smoothl1_proto_12e.py`

- Keeps the baseline Shared2FC feature and bbox regression path.
- Adds a lightweight Conv-GAP classification residual.
- Zero-initializes the residual projection, so the first forward is exactly
  the loaded baseline rather than a randomly perturbed classifier.
- Applies a class-balanced learnable-prototype contrastive loss with weight
  0.05 during training only.
- Loads baseline layer names unchanged; missing keys should only belong to
  `cls_residual`, `proto_projector`, and `loss_proto.prototypes`.

## E2: adjacent-level BCFN-style fusion + mild ARL

Config: `mtp-vit-l-rvsa_smoothl1_bcfn_arl_12e.py`

- P2 RoIs retain P2 detail and receive gated P3 semantic context.
- The adjacent residual is zero-initialized, preserving baseline RoI features
  at the start of fine-tuning.
- Positive classification weights use detached confidence and rotated IoU.
- The quality weight is floored at 0.25, normalized to mean one, and capped at
  2.0 so tiny vehicles are not discarded because of unstable IoU.

Both configs are flattened and do not use `_base_`.  Edit `data_root` if the
A100 dataset path differs and place/copy the starting checkpoint at:

```text
/autodl-fs/data/tianzhibei/weights/mtp_smoothl1_epoch35.pth
```

Run on 4xA100 80 GB (per-GPU batch 4, global batch 16):

```bash
torchrun --nproc_per_node=4 tools/train.py \
  configs/tianzhibei_car/mtp-vit-l-rvsa_smoothl1_proto_12e.py \
  --launcher pytorch

torchrun --nproc_per_node=4 tools/train.py \
  configs/tianzhibei_car/mtp-vit-l-rvsa_smoothl1_bcfn_arl_12e.py \
  --launcher pytorch
```

Use `load_from`, not `--resume`: these are 12 new fine-tuning epochs rather
than epochs 36--47 of the old staged run.

## E3: direction-aligned RGB secondary classifier

Generate crops from the corrected XML and the same image-level split:

```bash
python tools/tianzhibei/build_aligned_refiner_dataset.py \
  --data-root /autodl-fs/data/tianzhibei/data/car_det_train \
  --train-split /autodl-fs/data/tianzhibei/data/car_det_train/splits/train.txt \
  --val-split /autodl-fs/data/tianzhibei/data/car_det_train/splits/val.txt \
  --ann-dir gt_pixel \
  --output-dir /autodl-fs/data/tianzhibei/refiner_crops \
  --workers 16
```

The crop builder reads TIFFs through GDAL, rotates the vehicle long axis to
horizontal, writes a 128x64 tight crop and a 192x96 crop with 20% context, and
keeps source `image_id` in both manifests.

Train the classifier:

```bash
python tools/tianzhibei/train_aligned_refiner.py \
  configs/tianzhibei_car/aligned_refiner_convnext_tiny.py
```

Apply it to an MMRotate `predictions.pkl` for offline F1 comparison:

```bash
python tools/tianzhibei/refine_predictions_with_aligned_classifier.py \
  predictions.pkl \
  work_dirs/aligned_refiner_convnext_tiny/best_macro_f1.pth \
  predictions_aligned_refined.pkl \
  --image-root /autodl-fs/data/tianzhibei/data/car_det_train/input_path
```

Only four predefined pairs can be relabelled: Small Car/Van, Dump/Cargo,
Truck Tractor/Trailer, and Excavator/Tractor.  The default classifier gate is
confidence >=0.60 and pairwise margin >=0.15.  Search these two values and the
detector/classifier fusion weight on validation data, then report NPU latency
before enabling this branch in the submission image.  Rotated NMS is re-run
after relabelling because two boxes that previously had different labels can
become same-class duplicates.

## Acceptance criteria

For every experiment record:

1. IoU=0.3 macro F1 and per-class F1;
2. Small Car/Van and the three tail-pair confusion counts;
3. localization-only recall to ensure classification changes did not hide a
   proposal regression problem;
4. A100 validation time and Ascend 910B end-to-end latency;
5. class-wise thresholds re-optimized from the experiment's own predictions.
