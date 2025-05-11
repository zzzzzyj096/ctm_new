import torch
import math

from torch.optim.lr_scheduler import LambdaLR, SequentialLR, MultiStepLR

class warmup():
    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps

    def step(self, current_step):
        if current_step < self.warmup_steps:  # current_step / warmup_steps * base_lr
            return float(current_step / self.warmup_steps)
        else:                                 # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
            return 1.0
        
class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.00001,
        eta_min: float = 0.00001,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer):
                最適化手法インスタンス
            warmup_epochs (int):
                linear warmupを行うepoch数
            max_epochs (int):
                cosine曲線の終了に用いる 学習のepoch数
            warmup_start_lr (float):
                linear warmup 0 epoch目の学習率
            eta_min (float):
                cosine曲線の下限
            last_epoch (int):
                cosine曲線の位相オフセット
        学習率をmax_epochsに至るまでコサイン曲線に沿ってスケジュールする
        epoch 0からwarmup_epochsまでの学習曲線は線形warmupがかかる
        https://pytorch-lightning-bolts.readthedocs.io/en/stable/schedulers/warmup_cosine_annealing.html
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        return None

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
    
class WarmupMultiStepLR(object):
    def __init__(self, optimizer, warmup_steps, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.gamma = gamma

        # Define the warmup scheduler
        lambda_func = lambda step: step / warmup_steps if step < warmup_steps else 1.0
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda_func, last_epoch=last_epoch)

        # Define the multi-step scheduler
        multistep_scheduler = MultiStepLR(optimizer, milestones=[m - warmup_steps for m in milestones], gamma=gamma, last_epoch=last_epoch)

        # Chain the schedulers
        self.scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, multistep_scheduler], milestones=[warmup_steps])

    def step(self, epoch=None):
        self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)