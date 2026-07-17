train_manifest = '/autodl-fs/data/tianzhibei/refiner_crops/train.csv'
val_manifest = '/autodl-fs/data/tianzhibei/refiner_crops/val.csv'
work_dir = '/autodl-fs/data/tianzhibei/work_dirs/aligned_refiner_convnext_tiny'

seed = 3407
pretrained = True
epochs = 30
batch_size = 64
val_batch_size = 128
num_workers = 8
sampler_power = 0.5
label_smoothing = 0.05
lr = 2e-4
min_lr = 1e-6
weight_decay = 0.05
grad_clip = 5.0
amp = True
