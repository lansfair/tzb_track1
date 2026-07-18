# 天智杯细粒度车辆实验：A100 服务器运行手册

本文说明代码同步到 A100 服务器后需要执行的动作、执行顺序及原因。目标是比较三个相互独立的改进：

1. E1：MTP + 独立分类残差分支 + ProtoCL；
2. E2：MTP + 相邻层 BCFN + 温和 ARL；
3. E3：方向对齐的 ConvNeXt-T 二次分类器。

不要一开始叠加 E1、E2 和 E3。每次只改变一个因素，才能判断 F1 变化来自哪项改动。

## 1. 固定目录和硬件假设

本文按以下服务器目录编写：

```text
代码：/mnt/ht2-nas2/EO_test/wyf/tzb/src/mmrotate
环境：/mnt/ht2-nas2/EO_test/miniconda3/envs/tzb
数据：/mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train
权重：/mnt/ht2-nas2/EO_test/wyf/tzb/weights/mtp_smoothl1_epoch35.pth
输出：/mnt/ht2-nas2/EO_test/wyf/tzb/work_dirs
GPU：4 x A100 80 GB
```

E1、E2 和 E3 的配置已集中使用这些路径，不依赖环境变量。如果服务器实际目录不同，只修改三个配置文件顶部的 `data_root`、`checkpoint`、`work_root` 或 manifest 路径，不要在配置中到处替换字符串。

相关配置：

```text
configs/tianzhibei_car/mtp-vit-l-rvsa_smoothl1_proto_12e.py
configs/tianzhibei_car/mtp-vit-l-rvsa_smoothl1_bcfn_arl_12e.py
configs/tianzhibei_car/aligned_refiner_convnext_tiny.py
```

## 2. 同步代码，但不要破坏服务器现场

```bash
source /mnt/ht2-nas2/EO_test/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/ht2-nas2/EO_test/miniconda3/envs/tzb
cd /mnt/ht2-nas2/EO_test/wyf/tzb/src/mmrotate

git status --short
git fetch origin
git pull --ff-only origin main
git log -1 --oneline
```

理由：`--ff-only` 能避免服务器上的临时修改被 Git 自动合并。如果 `git status --short` 非空，先保存服务器上的日志或补丁，再处理代码同步，不要直接删除现场。

应能看到本仓库包含以下模块：

```bash
test -f projects/tianzhibei_car/fine_grained_heads.py
test -f projects/tianzhibei_car/bcfn.py
test -f tools/tianzhibei/build_aligned_refiner_dataset.py
test -f tools/tianzhibei/train_aligned_refiner.py
test -f tools/tianzhibei/refine_predictions_with_aligned_classifier.py
```

## 3. 只检查现有环境，不重新安装 OpenMMLab

```bash
python - <<'PY'
import torch
import mmcv
import mmengine
import mmdet
import mmrotate

print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('mmcv:', mmcv.__version__)
print('mmengine:', mmengine.__version__)
print('mmdet:', mmdet.__version__)
print('mmrotate:', mmrotate.__version__)
print('gpu_count:', torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

pip check
nvidia-smi
```

理由：这套环境已经能运行原 MTP/MMRotate 模型。重新执行 `pip install -U` 很容易改变 MMCV、MMDetection 和 PyTorch 的版本组合，导致旋转算子或 registry 失效。

E2 训练和 E3 后处理还依赖旋转 IoU/NMS，因此再做一次 CUDA 算子冒烟测试：

```bash
python - <<'PY'
import torch
from mmcv.ops import box_iou_rotated, nms_rotated

boxes = torch.tensor([
    [50., 50., 20., 10., 0.0],
    [52., 50., 20., 10., 0.0],
], device='cuda')
scores = torch.tensor([0.9, 0.8], device='cuda')
print('IoU:', box_iou_rotated(boxes, boxes, aligned=True))
print('NMS keep:', nms_rotated(boxes, scores, 0.1)[1])
PY
```

理由：仅成功导入 Python 包并不能证明 MMCV CUDA 扩展与当前 PyTorch/CUDA ABI 一致。这个测试能在正式训练前发现算子链接或架构错误。

## 4. 核对数据、划分和起始权重

```bash
test -d /mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train/input_path
test -d /mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train/gt_pixel
test -f /mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train/splits/train.txt
test -f /mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train/splits/val.txt
test -f /mnt/ht2-nas2/EO_test/wyf/tzb/weights/mtp_smoothl1_epoch35.pth

wc -l /mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train/splits/train.txt
wc -l /mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train/splits/val.txt
ls -lh /mnt/ht2-nas2/EO_test/wyf/tzb/weights/mtp_smoothl1_epoch35.pth
```

预期划分为 train 7447 张、val 1862 张。E1、E2 和 E3 都必须使用相同的 split 和 `gt_pixel/`，原因是：

- 只有固定验证集，实验分数才可直接比较；
- `gt_pixel/` 已离线处理异常地理坐标 XML；
- 二次分类器必须继承检测器的图像级划分，不能把同一原图的车辆裁剪分到训练集和验证集两侧。

进一步检查 split 是否重叠、引用文件是否齐全：

```bash
python - <<'PY'
from pathlib import Path

root = Path('/mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train')
train = {Path(x.strip()).stem for x in (root / 'splits/train.txt').read_text().splitlines() if x.strip()}
val = {Path(x.strip()).stem for x in (root / 'splits/val.txt').read_text().splitlines() if x.strip()}
missing = []
for image_id in sorted(train | val):
    if not (root / 'input_path' / f'{image_id}.tif').is_file():
        missing.append(f'image:{image_id}')
    if not (root / 'gt_pixel' / f'{image_id}.xml').is_file():
        missing.append(f'xml:{image_id}')
print('train:', len(train))
print('val:', len(val))
print('overlap:', len(train & val))
print('missing:', len(missing))
print('missing_examples:', missing[:20])
assert len(train) == 7447
assert len(val) == 1862
assert not train & val
assert not missing
PY
```

如果这些断言失败，不要启动训练。数据差异会让后续的模型消融失去意义。

## 5. 先复现 epoch 35 起点

用 E1 配置加载起始 checkpoint，先运行一次完整验证并保存预测：

```bash
mkdir -p /mnt/ht2-nas2/EO_test/wyf/tzb/work_dirs/preflight_epoch35

bash tools/dist_test.sh \
  configs/tianzhibei_car/mtp-vit-l-rvsa_smoothl1_proto_12e.py \
  /mnt/ht2-nas2/EO_test/wyf/tzb/weights/mtp_smoothl1_epoch35.pth \
  4 \
  --out /mnt/ht2-nas2/EO_test/wyf/tzb/work_dirs/preflight_epoch35/predictions.pkl
```

理由：E1 分类残差分支为零初始化，第一次前向应与原 checkpoint 的检测输出一致。这个步骤同时验证：

- checkpoint 与当前 MTP 结构匹配；
- 数据根目录和验证 split 正确；
- 四卡推理、RotatedRoIAlign 和 rotated NMS 正常；
- 当前环境仍能复现原约 0.655 mAP 的基线。

日志中允许出现的缺失参数应只属于新增模块，例如：

```text
cls_residual
proto_projector
loss_proto.prototypes
```

如果 backbone、FPN、RPN、`shared_fcs`、`fc_cls` 或 `fc_reg` 大量缺失，说明 checkpoint 或配置不匹配，不能继续训练。

还要用此前统一的 IoU=0.3 官方 F1 评估流程计算并保存：总 F1、macro-F1、每类 F1、Small Car/Van 混淆数。MMRotate 的 DOTA mAP 只能作为辅助指标，不能代替比赛 F1。

## 6. 运行 E1：独立分类分支 + ProtoCL

```bash
bash tools/dist_train.sh \
  configs/tianzhibei_car/mtp-vit-l-rvsa_smoothl1_proto_12e.py \
  4
```

配置已经固定：每卡 batch 4、全局 batch 16、12 个微调 epoch、每 2 个 epoch 验证一次、起始学习率 `2e-5`。

为什么先跑 E1：当前主要问题是正确定位后的细粒度分类错误，E1 直接解耦分类特征和回归特征，并通过类别原型改善 Small Car/Van 及尾类间隔；同时不会改变 proposal 和 bbox 回归路径，因果关系最清楚。

初次启动不要使用 `--resume`。这是从 epoch 35 权重开始的新 12 epoch 消融，不是恢复旧训练状态。只有 E1 自身中断后，才使用：

```bash
bash tools/dist_train.sh \
  configs/tianzhibei_car/mtp-vit-l-rvsa_smoothl1_proto_12e.py \
  4 \
  --resume
```

前 500 iteration 重点检查：

- `loss_cls`、`loss_bbox`、`loss_proto` 都是有限值；
- `loss_proto` 非零并逐渐稳定；
- 四卡显存相近，没有某一卡独占数据；
- GPU 利用率没有长期降到很低；
- 加载日志中没有非预期的旧模型参数缺失。

## 7. 运行 E2：BCFN + 温和 ARL

E1 完成后再启动 E2：

```bash
bash tools/dist_train.sh \
  configs/tianzhibei_car/mtp-vit-l-rvsa_smoothl1_bcfn_arl_12e.py \
  4
```

为什么独立运行：E2 同时解决小车辆的 P2 细节与相邻层语义不足，并根据分类置信度和旋转 IoU 温和调整正样本权重。它与 E1 的作用机制不同，先独立比较才能知道收益来自特征融合还是原型分类。

日志中额外检查：

```text
arl_mean_weight
arl_mean_iou
```

`arl_mean_weight` 应接近 1，因为代码做了批内均值归一化；若长期接近上限 2 或出现 NaN，应立即停止并检查 bbox decode/IoU。初次启动同样不要使用 `--resume`。

## 8. 对 E1、E2 使用同一验证流程

分别找到各 work directory 的最佳 checkpoint 和最后一个 checkpoint：

```bash
find /mnt/ht2-nas2/EO_test/wyf/tzb/work_dirs/mtp-vit-l-rvsa_smoothl1_proto_12e \
  -maxdepth 1 -name '*.pth' -printf '%f\n' | sort

find /mnt/ht2-nas2/EO_test/wyf/tzb/work_dirs/mtp-vit-l-rvsa_smoothl1_bcfn_arl_12e \
  -maxdepth 1 -name '*.pth' -printf '%f\n' | sort
```

对每个候选权重执行 `tools/dist_test.sh` 并添加 `--out predictions.pkl`。每个模型的预测必须写入不同目录，不能互相覆盖。

比较时固定以下内容：

| 项目 | 必须固定或记录的内容 |
|---|---|
| 数据 | 同一 train/val split、同一 `gt_pixel/` |
| 起点 | 同一个 MTP SmoothL1 epoch 35 checkpoint |
| 定位 | mAP、IoU=0.3 localization recall |
| 分类 | 总 F1、macro-F1、每类 F1、易混类别混淆矩阵 |
| 后处理 | 每个实验基于自己的预测重新搜索分类别阈值 |
| 性能 | 验证总时长、峰值显存、单图推理时间 |

理由：新分类分支会改变置信度分布。沿用旧模型的分类别阈值会低估新模型，直接比较未校准结果也不公平。

## 9. 运行 E3：方向对齐二次分类器

### 9.1 预缓存 ConvNeXt-T ImageNet 权重

```bash
python - <<'PY'
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
print(ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url)
convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
print('ConvNeXt-T pretrained weight is ready')
PY
```

理由：正式训练时才发现服务器无法访问 `download.pytorch.org` 会浪费排队时间。该权重通常缓存为 `~/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth`；离线服务器可提前把同名文件放入该目录。

### 9.2 生成方向对齐裁剪，只需执行一次

```bash
python tools/tianzhibei/build_aligned_refiner_dataset.py \
  --data-root /mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train \
  --train-split /mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train/splits/train.txt \
  --val-split /mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train/splits/val.txt \
  --ann-dir gt_pixel \
  --output-dir /mnt/ht2-nas2/EO_test/wyf/tzb/refiner_crops \
  --workers 16
```

完成后检查：

```bash
ls -lh /mnt/ht2-nas2/EO_test/wyf/tzb/refiner_crops/train.csv
ls -lh /mnt/ht2-nas2/EO_test/wyf/tzb/refiner_crops/val.csv
du -sh /mnt/ht2-nas2/EO_test/wyf/tzb/refiner_crops
find /mnt/ht2-nas2/EO_test/wyf/tzb/refiner_crops -name '*_errors.txt' -size +0c -print
```

理由：裁剪过程使用 GDAL 读取前三个 TIFF 波段，并将车辆长轴旋转到水平方向。它输出紧框和 20% 上下文两种视图，让分类器同时学习车体纹理和道路/停车环境。错误文件非空时，应先检查坏图或 XML，不能静默丢弃验证样本。

### 9.3 单卡训练分类器

在两个四卡检测实验都未运行时执行：

```bash
python tools/tianzhibei/train_aligned_refiner.py \
  configs/tianzhibei_car/aligned_refiner_convnext_tiny.py
```

理由：ConvNeXt-T 二次分类器远小于 MTP 检测器，单张 A100 足够。避免与四卡检测训练并发，可以减少共享存储读取和 GPU 资源竞争，使耗时记录更可信。

训练结果位于：

```text
/mnt/ht2-nas2/EO_test/wyf/tzb/work_dirs/aligned_refiner_convnext_tiny/
```

重点查看 `best_macro_f1.pth`、`last.pth` 和 `history.json`。选择 macro-F1 最优权重，而不是只看总体 accuracy，因为类别分布非常不均衡。

### 9.4 只对易混类别进行受控重分类

先对原 epoch 35 的验证预测运行：

```bash
python tools/tianzhibei/refine_predictions_with_aligned_classifier.py \
  /mnt/ht2-nas2/EO_test/wyf/tzb/work_dirs/preflight_epoch35/predictions.pkl \
  /mnt/ht2-nas2/EO_test/wyf/tzb/work_dirs/aligned_refiner_convnext_tiny/best_macro_f1.pth \
  /mnt/ht2-nas2/EO_test/wyf/tzb/work_dirs/aligned_refiner_convnext_tiny/predictions_refined.pkl \
  --image-root /mnt/ht2-nas2/EO_test/wyf/tzb/data/car_det_train/input_path \
  --classifier-confidence 0.60 \
  --classifier-margin 0.15 \
  --detector-weight 0.60
```

默认只允许四组内部换类：Small Car/Van、Dump Truck/Cargo Truck、Truck Tractor/Trailer、Excavator/Tractor。不会把任意类别改成另一个类别，且换类后重新执行 rotated NMS。

在验证集上至少比较以下门控组合：

```text
confidence: 0.55, 0.60, 0.65, 0.70
margin:     0.10, 0.15, 0.20
detector_weight: 0.50, 0.60, 0.70
```

理由：二次分类器最危险的问题不是漏改，而是错误地覆盖本来正确的检测类别。置信度和 margin 门控应根据验证 F1 选择，不能直接在测试集上调参。

确认 E3 对原 baseline 有收益后，再分别应用到 E1 和 E2 的最佳 `predictions.pkl`。这样可以判断 E3 与哪个检测模型互补，而不是默认三者都要叠加。

## 10. 推荐执行顺序和决策规则

严格按以下顺序：

```text
环境/数据/权重自检
  -> epoch35 基线复现
  -> E1 训练、预测、阈值重标定
  -> E2 训练、预测、阈值重标定
  -> E3 裁剪与分类器训练
  -> E3 分别后处理 baseline/E1/E2
  -> 选择最终方案
```

建议采用以下决策规则：

- E1 或 E2 只有在 macro-F1 和比赛总 F1 都不下降时才保留；
- 如果 mAP/定位召回下降而分类 F1 上升，先检查分类改动是否间接影响 proposal，不急于叠加其他模块；
- E3 必须报告额外推理耗时，F1 收益不足以抵消 Ascend 端延迟时不进入最终镜像；
- E1 与 E2 都有效时，最后再新建一次 `E1+E2` 组合实验，不能用两个独立实验的收益简单相加；
- 所有最终候选都重新优化分类别 score threshold，并使用同一官方 F1 脚本复核。

## 11. 需要保存的实验产物

每个实验目录至少保留：

```text
最终 config 副本
训练日志 JSON/文本
最佳 checkpoint
最后 checkpoint
validation predictions.pkl
统一评估结果 JSON
每类阈值 JSON
混淆矩阵
验证耗时和峰值显存记录
```

理由：只有 checkpoint 没有对应配置、阈值和预测记录时，无法复现比赛 F1，也无法判断精度变化来自模型还是后处理。

## 12. 常见错误与处理

### checkpoint 大量 missing/unexpected keys

停止训练，检查是否误用了 KFIoU、LSKNet 或类别数不是 10 的权重。E1 只应缺少新增分类/原型模块，E2 只应缺少 BCFN 新参数。

### loss 出现 NaN

E1 先检查 `loss_proto`；E2 检查 `arl_mean_iou` 和 rotated IoU CUDA 算子。不要通过忽略 NaN 继续训练。

### 四卡利用率低

先运行仓库已有的：

```bash
bash tools/diagnose_tianzhibei_gpu.sh
```

重点区分数据读取瓶颈、验证频率、CPU/GDAL 解码和 GPU 计算瓶颈。不要第一步就扩大 batch 或修改模型，这会破坏消融条件。

### 训练中断

只有相同实验、相同 work directory、相同配置的中断恢复才使用 `--resume`。从 epoch 35 开始新的 E1/E2 实验使用配置里的 `load_from`，不使用 `--resume`。
