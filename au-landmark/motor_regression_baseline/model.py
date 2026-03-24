#!/usr/bin/env python3
from __future__ import annotations

import torch.nn as nn


# 回归主模型：输入 latent24，输出 30 维归一化电机控制值
class MotorRegressorMLP(nn.Module):
    """Baseline MLP: 24 -> 64 -> 64 -> 30"""

    def __init__(self, input_dim: int = 24, hidden1: int = 64, hidden2: int = 64, output_dim: int = 30):
        super().__init__()
        # 两层隐藏层的 MLP 主干
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, output_dim),
        )

    # 前向计算：返回 30 维预测
    def forward(self, x):
        return self.net(x)
