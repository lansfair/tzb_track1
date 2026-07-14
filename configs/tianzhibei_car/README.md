# 天智杯车辆检测：正式训练配置

本目录只保留三个需要运行的配置。三个文件均为完全展开的独立配置，不使用
`_base_` 继承，也不从环境变量读取路径或训练参数。

## 固定目录

```text
/mnt/ht2-nas2/EO_test/tianzhibei/
├── data/car_det_train/
├── weights/
└── src/
    ├── mmrotate/
    └── tianzhibei-inference/
```

如果 A100 服务器上的实际目录不同，直接修改三个配置中的 `data_root`、LSK
`init_cfg.checkpoint` 和 MTP `load_from`。

## 数据校验与划分

训练前先完整解码每张 TIFF、解析 XML、剔除坏图，并按图像 SHA-256 去重。
随后使用固定随机种子进行多标签分层划分，同时约束每类出现图像数和实例数：

```bash
python tools/dataset_converters/build_tianzhibei_random_split.py \
  --data-root /mnt/ht2-nas2/EO_test/tianzhibei/data/car_det_train \
  --val-ratio 0.2 \
  --seed 3407 \
  --workers 8
```

当前压缩包审计结果：

```text
原始 TIFF/XML 对: 9445
坏图: 3（85.tif、3057.tif、3250.tif）
丢弃字节完全相同的重复影像: 133
有效唯一图像: 9309
训练集: 7447
验证集: 1862
```

`3057.tif` 是 0 字节文件；`85.tif` 和 `3250.tif` 能读取 TIFF 头但完整
像素解码失败。脚本不会删除原始数据，而是将它们写入
`splits/drop_invalid.txt`。三个配置读取 `splits/train.txt` 和
`splits/val.txt`，因此坏图和重复图不会进入训练或验证。

重复影像的 XML 并不总是完全一致。每个 SHA-256 图像组只保留标注实例最多
的样本；差异记录在 `duplicate_groups.csv`，保证同一影像不会同时出现在训练集
和验证集中。

主要输出包括：

```text
splits/train.txt
splits/val.txt
splits/all_unique.txt
splits/drop_invalid.txt
splits/drop_exact_duplicate.txt
splits/sample_validation.csv
splits/duplicate_groups.csv
splits/split_manifest.json
```

## 三个正式配置

### 1. LSKNet-S baseline

```text
lsknet-s_fpn_smoothl1_baseline_40e.py
```

作为无阶段训练对照组。使用固定训练增强、普通采样、Smooth L1 和 EMA。

### 2. LSKNet-S A/B/C

```text
lsknet-s_fpn_smoothl1_staged_40e.py
```

- Stage A，epoch 0–27：主训练。
- Stage B，epoch 28–35：长尾采样和困难增强。
- Stage C，epoch 36–39：恢复原始分布并低学习率校准。

### 3. MTP ViT-L+RVSA A/B/C

```text
mtp-vit-l-rvsa_smoothl1_staged_40e.py
```

使用转换后的 FAIR1M 检测权重，并沿用相同 A/B/C 数据策略。MTP backbone
从相邻的 `tianzhibei-inference` 仓库自动加载。

## 固定四卡参数

面向 4×A100 80 GB：

```text
每卡训练 batch: 4
全局 batch: 16
梯度累积: 1
验证/测试每卡 batch: 4
AMP loss scale 初值: 512
LSK 主学习率: 2e-4
MTP 主学习率: 1e-4
```

LSK 使用 SyncBN；MTP 关闭 activation checkpointing。两者均使用 fused
AdamW、EMA、固定 1024 输入、DDP gradient bucket view 和 50 MB bucket。

验证安排：前 5 个 epoch 每轮验证；Stage A 中段每 2 轮验证；从 epoch 28
开始恢复每轮验证。常规 checkpoint 每 2 轮保存，最佳验证 checkpoint 仍即时保存。

## 推荐运行顺序

```text
LSK baseline → LSK A/B/C → MTP A/B/C
```

运行命令见 `SEETACLOUD_RUNBOOK.md`。
