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

## 固定数据划分

`splits/` 保存由比赛集与 FAIR1M 来源匹配及 SHA-256 去重结果生成的固定划分：

```text
splits/train.txt                  6294 张
splits/val.txt                    3018 张
splits/drop_exact_duplicate.txt    133 张
splits/all_unique.txt             9312 张
```

训练集与验证集不存在 SHA-256 完全重复图像。三个配置从数据目录中的
`car_det_train/splits/train.txt` 和 `car_det_train/splits/val.txt` 读取划分。
首次部署到服务器时执行：

```bash
mkdir -p /mnt/ht2-nas2/EO_test/tianzhibei/data/car_det_train/splits
cp configs/tianzhibei_car/splits/*.txt \
  /mnt/ht2-nas2/EO_test/tianzhibei/data/car_det_train/splits/
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
