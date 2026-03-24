#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# 将图片文件名解析为整型索引（如 01234.jpg -> 1234）
def _parse_idx_from_name(name: str) -> int:
    return int(Path(name).stem)


# 读取 latent24 文件并构建 idx -> latent 向量映射
def load_latent24_map(latent_file: Path) -> Dict[int, np.ndarray]:
    latent_cols = [f"latent_{i:02d}" for i in range(24)]
    out: Dict[int, np.ndarray] = {}
    with gzip.open(latent_file, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        need = ["image_name"] + latent_cols
        missing = [c for c in need if c not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"latent file missing columns: {missing}")
        for row in reader:
            idx = _parse_idx_from_name(row["image_name"])
            out[idx] = np.asarray([float(row[c]) for c in latent_cols], dtype=np.float32)
    return out


# 读取归一化标签并构建 idx -> 30维电机值映射
def load_target30_map(metadata_normalize_file: Path) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    with metadata_normalize_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = _parse_idx_from_name(obj["file_name"])
            ctrl = np.asarray(obj["ctrl_value"], dtype=np.float32)
            if ctrl.shape[0] != 30:
                raise RuntimeError(f"ctrl_value dim != 30: {obj['file_name']}")
            out[idx] = ctrl
    return out


# 从 split pkl 中读取样本索引列表
def load_split_indices(split_pkl: Path) -> List[int]:
    obj = pickle.load(open(split_pkl, "rb"))
    if "img_path" not in obj:
        raise RuntimeError(f"split file format error: {split_pkl}")
    idxs = [_parse_idx_from_name(str(p)) for p in obj["img_path"]]
    return idxs


# 根据 split 组装训练/验证/测试的 X(latent24) 与 Y(ctrl30)
def build_xy_from_split(
    split_pkl: Path,
    latent24_map: Dict[int, np.ndarray],
    target30_map: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    idxs = load_split_indices(split_pkl)
    x_list = []
    y_list = []
    miss_lat = 0
    miss_tgt = 0
    for idx in idxs:
        x = latent24_map.get(idx)
        y = target30_map.get(idx)
        if x is None:
            miss_lat += 1
            continue
        if y is None:
            miss_tgt += 1
            continue
        x_list.append(x)
        y_list.append(y)
    if miss_lat > 0 or miss_tgt > 0:
        raise RuntimeError(f"split {split_pkl.name} missing latent={miss_lat}, missing target={miss_tgt}")
    x_arr = np.vstack(x_list).astype(np.float32)
    y_arr = np.vstack(y_list).astype(np.float32)
    return x_arr, y_arr


# 训练数据集封装：返回 (x, y)
class XYDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
