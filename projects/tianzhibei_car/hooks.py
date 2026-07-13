from mmengine.hooks import Hook

from mmrotate.registry import HOOKS


@HOOKS.register_module()
class TianzhibeiStageHook(Hook):
    """Apply non-pipeline changes at continuous-training stage boundaries."""

    priority = 'NORMAL'

    def __init__(self, stages):
        self.stages = {stage['begin_epoch']: stage for stage in stages}

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if epoch not in self.stages:
            return
        stage = self.stages[epoch]
        runner.logger.info(
            f'Entering Tianzhibei stage {stage["name"]} at epoch {epoch}')
        if 'lr' in stage:
            for group in runner.optim_wrapper.optimizer.param_groups:
                group['lr'] = stage['lr']
                group['initial_lr'] = stage['lr']
        if 'frozen_stages' in stage:
            model = runner.model.module if hasattr(
                runner.model, 'module') else runner.model
            backbone = model.backbone
            backbone.frozen_stages = stage['frozen_stages']
            if hasattr(backbone, '_freeze_stages'):
                backbone._freeze_stages()
