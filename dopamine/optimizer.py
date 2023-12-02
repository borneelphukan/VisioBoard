import torch

from torch import nn
import numpy as np

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class Dopamine:
    def __init__(self, lr=1e-3, s_init=0.9, beta_s=0.5, beta_lr=0.999):
        self.lr = lr
        self.beta_s = beta_s
        self.beta_lr = beta_lr
        self.s = s_init

    def compute_s(self, param, reward):
        param.s = self.beta_s * param.s - (1 - self.beta_s) * reward

        return param.s

    def compute_lr(self, param, reward):
        param.s = self.compute_s(param, reward)
        param.lr = (1-self.beta_lr) * param.lr + self.beta_lr * param.s
        return param.lr
