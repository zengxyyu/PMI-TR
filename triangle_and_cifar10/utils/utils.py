from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, multiplier: float, steps: int):
        self.multiplier = multiplier
        self.steps = steps
        super(WarmupScheduler, self).__init__(optimizer=optimizer)

    def get_lr(self):
        if self.last_epoch < self.steps:
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return self.base_lrs

    def decay_lr(self, decay_factor: float):
        self.base_lrs = [decay_factor * base_lr for base_lr in self.base_lrs]