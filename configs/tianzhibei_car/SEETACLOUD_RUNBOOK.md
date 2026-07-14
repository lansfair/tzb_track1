# 4×A100 80 GB 运行手册

## 激活环境

```bash
source /mnt/ht2-nas2/EO_test/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/ht2-nas2/EO_test/miniconda3/envs/tzb
cd /mnt/ht2-nas2/EO_test/tianzhibei/src/mmrotate
```

配置不依赖项目环境变量。

## 0. 校验数据并生成固定划分

首次训练前执行一次：

```bash
python tools/dataset_converters/build_tianzhibei_random_split.py \
  --data-root /mnt/ht2-nas2/EO_test/tianzhibei/data/car_det_train \
  --val-ratio 0.2 \
  --seed 3407 \
  --workers 8
```

确认 `split_manifest.json` 中 `invalid_images=3`、`train_images=7447`、
`val_images=1862`，并确认十类在训练集和验证集中都存在，再启动训练。

## 1. LSKNet-S baseline

```bash
bash tools/dist_train.sh \
  configs/tianzhibei_car/lsknet-s_fpn_smoothl1_baseline_40e.py \
  4 \
  --work-dir /mnt/ht2-nas2/EO_test/tianzhibei/work_dirs/lsknet_s_baseline
```

## 2. LSKNet-S A/B/C

```bash
bash tools/dist_train.sh \
  configs/tianzhibei_car/lsknet-s_fpn_smoothl1_staged_40e.py \
  4 \
  --work-dir /mnt/ht2-nas2/EO_test/tianzhibei/work_dirs/lsknet_s_abc
```

## 3. MTP ViT-L+RVSA A/B/C

```bash
bash tools/dist_train.sh \
  configs/tianzhibei_car/mtp-vit-l-rvsa_smoothl1_staged_40e.py \
  4 \
  --work-dir /mnt/ht2-nas2/EO_test/tianzhibei/work_dirs/mtp_vit_l_rvsa_abc
```

三个实验按顺序运行，不要在同一组 GPU 上并发启动。不要添加
`--auto-scale-lr`，配置内学习率已经对应全局 batch 16。

## 恢复训练

使用相同命令并追加：

```bash
--resume
```

每个 work directory 最多保留三个常规 checkpoint，并额外保存验证集最佳模型。
