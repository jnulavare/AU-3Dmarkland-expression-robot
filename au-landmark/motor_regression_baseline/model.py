#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn as nn


# 30 维电机按区域分组（默认分组，可由配置覆盖）
DEFAULT_MOTOR_REGION_INDICES: Dict[str, List[int]] = {
    "brow": [0, 1, 2, 3],
    "eye": [4, 5, 6, 7, 8, 9],
    "mouth": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29],
    "jaw": [10, 11, 12, 13, 14, 27, 28],
}


def _sanitize_key(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


def _to_int_list(v: Sequence[int]) -> List[int]:
    out: List[int] = []
    for x in v:
        if not isinstance(x, int):
            raise RuntimeError(f"motor region index must be int, got: {x!r}")
        out.append(int(x))
    return out


def _normalize_region_map(
    region_cfg: Mapping[str, Sequence[int]] | None,
    output_dim: int,
) -> tuple[List[str], Dict[str, List[int]]]:
    region_map = (
        {str(k): _to_int_list(v) for k, v in region_cfg.items()}
        if isinstance(region_cfg, Mapping)
        else {k: list(v) for k, v in DEFAULT_MOTOR_REGION_INDICES.items()}
    )
    if len(region_map) == 0:
        raise RuntimeError("motor_region_indices is empty")

    for region, idxs in region_map.items():
        if len(idxs) == 0:
            raise RuntimeError(f"motor region '{region}' has empty indices")
        for i in idxs:
            if i < 0 or i >= output_dim:
                raise RuntimeError(f"motor region '{region}' index out of range: {i}, output_dim={output_dim}")

    owner: List[str | None] = [None] * output_dim
    for region, idxs in region_map.items():
        for i in idxs:
            if owner[i] is not None:
                raise RuntimeError(f"motor index overlap: index={i} used by '{owner[i]}' and '{region}'")
            owner[i] = region
    missing = [i for i, v in enumerate(owner) if v is None]
    if missing:
        raise RuntimeError(f"motor indices not fully covered, missing={missing}")

    return list(region_map.keys()), region_map


class MotorRegressorMLP(nn.Module):
    """
    latent24 -> motor30 回归模型（shared trunk + grouped heads）。

    结构：
    1) 共享 trunk 提取全局公共表征 h_shared；
    2) 每个 motor 区域一个 head 预测该区域电机子向量；
    3) 按原始 30 维电机索引回填拼接，得到完整输出。
    """

    def __init__(
        self,
        input_dim: int = 24,
        hidden1: int = 64,
        hidden2: int = 64,
        output_dim: int = 30,
        motor_region_indices: Mapping[str, Sequence[int]] | None = None,
    ):
        super().__init__()
        self.output_dim = int(output_dim)

        # 共享主干：保留跨区域协同、全局表情上下文和通用表征能力
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
        )

        region_order, region_map = _normalize_region_map(motor_region_indices, output_dim=self.output_dim)
        self.region_order = region_order
        self.region_map = region_map

        # 分组输出头：每个区域单独预测，再回填到最终 30 维输出
        self.heads = nn.ModuleDict()
        self._region_buffer_names: Dict[str, str] = {}
        for region in self.region_order:
            idxs = self.region_map[region]
            self.heads[region] = nn.Linear(hidden2, len(idxs))
            buffer_name = f"_region_idx_{_sanitize_key(region)}"
            self.register_buffer(buffer_name, torch.tensor(idxs, dtype=torch.long), persistent=False)
            self._region_buffer_names[region] = buffer_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_shared = self.trunk(x)
        out = x.new_zeros((x.shape[0], self.output_dim))
        for region in self.region_order:
            idx = getattr(self, self._region_buffer_names[region])
            region_pred = self.heads[region](h_shared)
            out.index_copy_(1, idx, region_pred)
        return out
