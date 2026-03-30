#!/usr/bin/env python3
from __future__ import annotations

"""Data loading helpers for compare1 (feature382 -> motor30)."""

import csv
import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_idx_from_name(name: str) -> int:
    return int(Path(name).stem)


def _open_text(path: Path, mode: str):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def _extract_numeric_suffix(col: str, prefix: str) -> int:
    if not col.startswith(prefix):
        return -1
    tail = col[len(prefix) :]
    return int(tail) if tail.isdigit() else -1


def _infer_feature_columns(fieldnames: List[str]) -> Tuple[List[str], str]:
    # New compare1 format: feat_000 ... feat_381
    feat_cols = [c for c in fieldnames if _extract_numeric_suffix(c, "feat_") >= 0]
    if feat_cols:
        feat_cols.sort(key=lambda c: _extract_numeric_suffix(c, "feat_"))
        return feat_cols, "feat_"

    # Legacy format fallback: latent_00 ... latent_23
    latent_cols = [c for c in fieldnames if _extract_numeric_suffix(c, "latent_") >= 0]
    if latent_cols:
        latent_cols.sort(key=lambda c: _extract_numeric_suffix(c, "latent_"))
        return latent_cols, "latent_"

    raise RuntimeError("feature file must contain either feat_### columns or latent_## columns")


def load_feature_map(feature_file: Path, expected_dim: int | None = None) -> Dict[int, np.ndarray]:
    """Load `image_name -> feature vector` mapping from feature CSV."""
    out: Dict[int, np.ndarray] = {}
    with _open_text(feature_file, "rt") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "image_name" not in fieldnames:
            raise RuntimeError("feature file missing required column: image_name")

        feature_cols, feature_prefix = _infer_feature_columns(fieldnames)
        if expected_dim is not None and len(feature_cols) != int(expected_dim):
            raise RuntimeError(
                f"feature dim mismatch in {feature_file}: expected={expected_dim}, actual={len(feature_cols)} "
                f"(prefix={feature_prefix})"
            )

        for row in reader:
            idx = _parse_idx_from_name(row["image_name"])
            out[idx] = np.asarray([float(row[c]) for c in feature_cols], dtype=np.float32)
    return out


def load_latent24_map(latent_file: Path) -> Dict[int, np.ndarray]:
    """Legacy compatibility helper for old latent24 experiments."""
    return load_feature_map(latent_file, expected_dim=24)


def load_target30_map(metadata_normalize_file: Path) -> Dict[int, np.ndarray]:
    """Load normalized 30-dim motor target vectors from metadata jsonl."""
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


def load_split_indices(split_pkl: Path) -> List[int]:
    """Load split sample ids from pickle (`img_path`)."""
    obj = pickle.load(open(split_pkl, "rb"))
    if "img_path" not in obj:
        raise RuntimeError(f"split file format error: {split_pkl}")
    return [_parse_idx_from_name(str(p)) for p in obj["img_path"]]


def build_xy_from_split(
    split_pkl: Path,
    feature_map: Dict[int, np.ndarray],
    target30_map: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble aligned X(feature) and Y(target) arrays for a given split."""
    idxs = load_split_indices(split_pkl)
    x_list = []
    y_list = []
    miss_x = 0
    miss_y = 0
    for idx in idxs:
        x = feature_map.get(idx)
        y = target30_map.get(idx)
        if x is None:
            miss_x += 1
            continue
        if y is None:
            miss_y += 1
            continue
        x_list.append(x)
        y_list.append(y)
    if miss_x > 0 or miss_y > 0:
        raise RuntimeError(f"split {split_pkl.name} missing feature={miss_x}, missing target={miss_y}")
    return np.vstack(x_list).astype(np.float32), np.vstack(y_list).astype(np.float32)


class XYDataset(Dataset):
    """Torch dataset wrapper for prebuilt numpy arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
