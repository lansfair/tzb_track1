_base_ = './lsknet-s_fpn_smoothl1_staged_40e.py'

# KFIoU is applied to the second-stage rotated box regression only. The RPN
# keeps its original SmoothL1 loss, matching the repository's KFIoU recipes.
model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_bbox_type='kfiou',
            loss_bbox=dict(
                _delete_=True, type='KFLoss', fun='ln', loss_weight=5.0))))
