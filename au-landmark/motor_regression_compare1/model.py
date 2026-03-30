#!/usr/bin/env python3
from __future__ import annotations

import torch.nn as nn


class MotorRegressorMLP(nn.Module):
    """Baseline MLP: input_dim -> hidden1 -> hidden2 -> output_dim"""

    def __init__(self, input_dim: int = 382, hidden1: int = 512, hidden2: int = 256, output_dim: int = 30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        return self.net(x)
