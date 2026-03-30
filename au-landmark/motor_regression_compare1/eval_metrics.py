#!/usr/bin/env python3
"""评估指标与误差剖析工具模块。"""

# flake8: noqa: E501
# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,invalid-name,line-too-long
from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


# 默认的 30 维电机区域映射（可被 config.metrics.motor_region_indices 覆盖）
DEFAULT_MOTOR_REGION_INDICES: Dict[str, List[int]] = {
    "brow": [0, 1, 2, 3],
    "eye": [4, 5, 6, 7, 8, 9],
    "jaw": [10, 11, 12, 13, 14, 27, 28],
    "mouth": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29],
}


# 默认电机名称（与 ctrl_value[0..29] 索引对齐）
DEFAULT_MOTOR_NAMES: List[str] = [
    "Brow Inner Left",
    "Brow Inner Right",
    "Brow Outer Left",
    "Brow Outer Right",
    "Eyelid Lower Left",
    "Eyelid Lower Right",
    "Eyelid Upper Left",
    "Eyelid Upper Right",
    "Gaze Target Phi",
    "Gaze Target Theta",
    "Head Pitch",
    "Head Roll",
    "Head Yaw",
    "Jaw Pitch",
    "Jaw Yaw",
    "Lip Bottom Curl",
    "Lip Bottom Depress Left",
    "Lip Bottom Depress Middle",
    "Lip Bottom Depress Right",
    "Lip Corner Raise Left",
    "Lip Corner Raise Right",
    "Lip Corner Stretch Left",
    "Lip Corner Stretch Right",
    "Lip Top Curl",
    "Lip Top Raise Left",
    "Lip Top Raise Middle",
    "Lip Top Raise Right",
    "Neck Pitch",
    "Neck Roll",
    "Nose Wrinkle",
]


def _open_text(path: Path, mode: str):
    """按文件后缀自动以普通文本或 gzip 文本方式打开。"""
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def _parse_idx_from_name(name: str) -> int:
    """将图片文件名解析为整数索引（如 000123.jpg -> 123）。"""
    return int(Path(name).stem)


def _to_index_list(v: object) -> List[int]:
    """将任意对象校验并转换为 list[int]。"""
    if not isinstance(v, list):
        raise RuntimeError(f"region indices must be list[int], got: {type(v)}")
    out: List[int] = []
    for x in v:
        if not isinstance(x, int):
            raise RuntimeError(f"region index must be int, got: {x!r}")
        out.append(x)
    return out


def load_motor_region_indices(metrics_cfg: Mapping[str, object] | None, dim: int) -> Dict[str, List[int]]:
    """从配置读取电机区域索引映射，并做范围合法性校验。"""
    region_cfg = None
    if isinstance(metrics_cfg, Mapping):
        region_cfg = metrics_cfg.get("motor_region_indices")

    if isinstance(region_cfg, Mapping):
        region_map: Dict[str, List[int]] = {str(k): _to_index_list(v) for k, v in region_cfg.items()}
    else:
        region_map = {k: list(v) for k, v in DEFAULT_MOTOR_REGION_INDICES.items()}

    for region, idxs in region_map.items():
        if len(idxs) == 0:
            raise RuntimeError(f"region '{region}' has empty indices")
        for i in idxs:
            if i < 0 or i >= dim:
                raise RuntimeError(f"region '{region}' index out of range: {i}, dim={dim}")
    return region_map


def load_motor_names(metrics_cfg: Mapping[str, object] | None, dim: int) -> List[str]:
    """从配置读取电机名称，长度不足时自动补齐占位名称。"""
    names_cfg = None
    if isinstance(metrics_cfg, Mapping):
        names_cfg = metrics_cfg.get("motor_names")

    if isinstance(names_cfg, list):
        names = [str(x) for x in names_cfg]
    else:
        names = list(DEFAULT_MOTOR_NAMES)

    if len(names) < dim:
        names.extend([f"motor_{i:02d}" for i in range(len(names), dim)])
    return names[:dim]


def clip_predictions_to_range(y_pred: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """将预测值裁剪到合法区间 [lo, hi]。"""
    return np.clip(y_pred, lo, hi).astype(np.float64)


def compute_boundary_violation_metrics(
    y_pred: np.ndarray,
    lo: float,
    hi: float,
) -> Dict[str, object]:
    """计算边界相关统计：越界比例、boundary loss、各维度越界比例。"""
    if y_pred.ndim != 2:
        raise RuntimeError(f"expect 2D y_pred, got ndim={y_pred.ndim}")
    n, d = y_pred.shape
    total = int(n * d)

    low_violation = np.maximum(lo - y_pred, 0.0)
    high_violation = np.maximum(y_pred - hi, 0.0)
    violation = low_violation + high_violation
    oor_mask = violation > 0.0

    count_per_dim = np.sum(oor_mask, axis=0).astype(np.int64)
    ratio_per_dim = np.mean(oor_mask.astype(np.float64), axis=0)

    return {
        "lo": float(lo),
        "hi": float(hi),
        "count": int(np.sum(oor_mask)),
        "total": total,
        "ratio": float(np.sum(oor_mask) / max(total, 1)),
        "ratio_per_dim": [float(v) for v in ratio_per_dim.tolist()],
        "count_per_dim": [int(v) for v in count_per_dim.tolist()],
        "boundary_loss_mean": float(np.mean(violation)),
        "boundary_loss_per_dim": [float(v) for v in np.mean(violation, axis=0).tolist()],
        "boundary_loss_p95": float(np.percentile(violation, 95.0)),
        "low_violation_mean": float(np.mean(low_violation)),
        "high_violation_mean": float(np.mean(high_violation)),
    }


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """运行模型并收集整套预测与真值数组。"""
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x)
            ys.append(y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
    if len(ys) == 0:
        raise RuntimeError("empty loader: no samples to evaluate")
    return np.vstack(ys).astype(np.float64), np.vstack(preds).astype(np.float64)


def _r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, int]:
    """逐维计算 R2，并返回有效维度数量（方差非零）。"""
    err = y_true - y_pred
    ss_res = np.sum(err * err, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    valid = ss_tot > 1e-12
    r2 = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    r2[valid] = 1.0 - (ss_res[valid] / ss_tot[valid])
    return r2, int(np.sum(valid))


def _explained_variance_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, int]:
    """逐维计算 Explained Variance，并返回有效维度数量。"""
    err = y_true - y_pred
    var_true = np.var(y_true, axis=0)
    var_err = np.var(err, axis=0)
    valid = var_true > 1e-12
    ev = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    ev[valid] = 1.0 - (var_err[valid] / var_true[valid])
    return ev, int(np.sum(valid))


def _jsonable_float_list(arr: np.ndarray) -> List[float | None]:
    """将浮点数组转换为可写入 JSON 的列表（NaN/Inf -> None）。"""
    out: List[float | None] = []
    for v in arr.tolist():
        fv = float(v)
        out.append(fv if np.isfinite(fv) else None)
    return out


def _region_ranking(region_metrics: Mapping[str, Mapping[str, object]], key: str) -> List[Dict[str, object]]:
    """按指定指标字段对区域误差进行降序排序。"""
    rows: List[Dict[str, object]] = []
    for region, obj in region_metrics.items():
        rows.append(
            {
                "region": region,
                "mae": float(obj["mae"]),
                "rmse": float(obj["rmse"]),
                "p95_abs_err": float(obj["p95_abs_err"]),
                "max_abs_err": float(obj["max_abs_err"]),
            }
        )
    return sorted(rows, key=lambda x: x[key], reverse=True)


def _motor_ranking(values: np.ndarray, metric_name: str, motor_names: List[str]) -> List[Dict[str, object]]:
    """按某个电机指标值做降序排名。"""
    order = np.argsort(-values)
    out: List[Dict[str, object]] = []
    for idx in order.tolist():
        out.append({"motor_idx": int(idx), "motor_name": motor_names[idx], metric_name: float(values[idx])})
    return out


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """计算皮尔逊相关系数；样本不足或方差过小则返回 None。"""
    if x.size < 3 or y.size < 3:
        return None
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _quantile_relation(x: np.ndarray, y: np.ndarray, bins: int) -> List[Dict[str, object]]:
    """按分位数分箱统计特征与样本误差关系（用于趋势观察）。"""
    if x.size < 10:
        return []
    bins = int(max(2, bins))
    q = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    q = np.unique(q)
    if q.size < 3:
        return []

    out: List[Dict[str, object]] = []
    for i in range(q.size - 1):
        lo = float(q[i])
        hi = float(q[i + 1])
        if i < q.size - 2:
            mask = (x >= lo) & (x < hi)
        else:
            mask = (x >= lo) & (x <= hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        out.append(
            {
                "bin": int(i),
                "lo": lo,
                "hi": hi,
                "count": cnt,
                "feature_mean": float(np.mean(x[mask])),
                "sample_mae_mean": float(np.mean(y[mask])),
            }
        )
    return out


def _analyze_scalar_relation(feature: np.ndarray, sample_mae: np.ndarray, bins: int) -> Dict[str, object]:
    """分析单个标量特征与样本误差的相关性及分箱趋势。"""
    mask = np.isfinite(feature) & np.isfinite(sample_mae)
    valid_n = int(np.sum(mask))
    out: Dict[str, object] = {
        "available": valid_n >= 10,
        "valid_samples": valid_n,
        "coverage_ratio": float(valid_n / max(sample_mae.size, 1)),
    }
    if valid_n < 10:
        return out

    x = feature[mask]
    y = sample_mae[mask]
    out.update(
        {
            "pearson_corr": _pearson_corr(x, y),
            "pearson_corr_abs_feature": _pearson_corr(np.abs(x), y),
            "quantile_bins": _quantile_relation(x, y, bins=bins),
            "quantile_bins_abs_feature": _quantile_relation(np.abs(x), y, bins=bins),
        }
    )
    return out


def load_context_feature_arrays(
    split_indices: List[int],
    rel_file: Path | None,
    abs_file: Path | None,
) -> Dict[str, object]:
    """按样本索引加载 ENERGY_rel 与 yaw/pitch/roll 侧信息数组。"""
    n = len(split_indices)
    energy = np.full(n, np.nan, dtype=np.float64)
    yaw = np.full(n, np.nan, dtype=np.float64)
    pitch = np.full(n, np.nan, dtype=np.float64)
    roll = np.full(n, np.nan, dtype=np.float64)

    out: Dict[str, object] = {
        "energy_rel": energy,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "sources": {
            "rel_file": str(rel_file) if rel_file is not None else None,
            "abs_file": str(abs_file) if abs_file is not None else None,
        },
        "status": {},
    }

    if rel_file is not None and rel_file.exists():
        rel_map: Dict[int, float] = {}
        with _open_text(rel_file, "rt") as f:
            reader = csv.DictReader(f)
            if "image_name" in (reader.fieldnames or []) and "ENERGY_rel" in (reader.fieldnames or []):
                for row in reader:
                    idx = _parse_idx_from_name(row["image_name"])
                    rel_map[idx] = float(row["ENERGY_rel"])
                for i, idx in enumerate(split_indices):
                    if idx in rel_map:
                        energy[i] = rel_map[idx]
                out["status"]["energy_rel"] = "ok"
            else:
                out["status"]["energy_rel"] = "missing_columns"
    else:
        out["status"]["energy_rel"] = "file_not_found"

    if abs_file is not None and abs_file.exists():
        pose_map: Dict[int, Tuple[float, float, float]] = {}
        with _open_text(abs_file, "rt") as f:
            reader = csv.DictReader(f)
            need = {"image_name", "yaw", "pitch", "roll"}
            if need.issubset(set(reader.fieldnames or [])):
                for row in reader:
                    idx = _parse_idx_from_name(row["image_name"])
                    pose_map[idx] = (float(row["yaw"]), float(row["pitch"]), float(row["roll"]))
                for i, idx in enumerate(split_indices):
                    if idx in pose_map:
                        y0, p0, r0 = pose_map[idx]
                        yaw[i] = y0
                        pitch[i] = p0
                        roll[i] = r0
                out["status"]["pose"] = "ok"
            else:
                out["status"]["pose"] = "missing_columns"
    else:
        out["status"]["pose"] = "file_not_found"

    return out


def analyze_error_vs_context(
    sample_mae: np.ndarray,
    context: Mapping[str, object],
    bins: int = 10,
) -> Dict[str, object]:
    """计算样本误差与 ENERGY_rel / yaw / pitch / roll 的关系统计。"""
    energy = np.asarray(context.get("energy_rel", np.array([])), dtype=np.float64)
    yaw = np.asarray(context.get("yaw", np.array([])), dtype=np.float64)
    pitch = np.asarray(context.get("pitch", np.array([])), dtype=np.float64)
    roll = np.asarray(context.get("roll", np.array([])), dtype=np.float64)

    return {
        "sample_error_definition": "sample_mae = mean(abs(pred - target), axis=30_motors)",
        "sources": context.get("sources", {}),
        "status": context.get("status", {}),
        "energy_rel": _analyze_scalar_relation(energy, sample_mae, bins=bins),
        "pose": {
            "yaw": _analyze_scalar_relation(yaw, sample_mae, bins=bins),
            "pitch": _analyze_scalar_relation(pitch, sample_mae, bins=bins),
            "roll": _analyze_scalar_relation(roll, sample_mae, bins=bins),
        },
    }


def compute_pose_slice_mae_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
    roll: np.ndarray,
    region_indices: Mapping[str, Iterable[int]],
    motor_names: List[str],
    frontal_max_deg: float = 10.0,
    moderate_max_deg: float = 25.0,
) -> Dict[str, object]:
    """按姿态切片统计 MAE（overall / per-region / per-motor）。"""
    if y_true.shape != y_pred.shape:
        raise RuntimeError(f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.ndim != 2:
        raise RuntimeError(f"expect 2D arrays, got y_true.ndim={y_true.ndim}")
    if moderate_max_deg <= frontal_max_deg:
        raise RuntimeError(
            f"invalid pose slice thresholds: frontal_max_deg={frontal_max_deg}, moderate_max_deg={moderate_max_deg}"
        )

    n, d = y_true.shape
    if len(motor_names) < d:
        motor_names = motor_names + [f"motor_{i:02d}" for i in range(len(motor_names), d)]
    motor_names = motor_names[:d]

    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    roll = np.asarray(roll, dtype=np.float64)
    if yaw.shape[0] != n or pitch.shape[0] != n or roll.shape[0] != n:
        raise RuntimeError(
            f"pose array size mismatch: y={n}, yaw={yaw.shape[0]}, pitch={pitch.shape[0]}, roll={roll.shape[0]}"
        )

    valid_pose = np.isfinite(yaw) & np.isfinite(pitch) & np.isfinite(roll)
    pose_strength = np.maximum(np.abs(yaw), np.maximum(np.abs(pitch), np.abs(roll)))
    abs_err = np.abs(y_pred - y_true)

    slice_masks = {
        "frontal": valid_pose & (pose_strength <= frontal_max_deg),
        "moderate_pose": valid_pose & (pose_strength > frontal_max_deg) & (pose_strength <= moderate_max_deg),
        "extreme_pose": valid_pose & (pose_strength > moderate_max_deg),
    }

    out_slices: Dict[str, object] = {}
    for slice_name, mask in slice_masks.items():
        count = int(np.sum(mask))
        slice_obj: Dict[str, object] = {
            "samples": count,
            "sample_ratio": float(count / max(n, 1)),
            "overall_mae": None,
            "per_region_mae": {},
            "per_motor_mae": [],
        }
        if count == 0:
            out_slices[slice_name] = slice_obj
            continue

        s_abs_err = abs_err[mask]
        mae_per_motor = np.mean(s_abs_err, axis=0)
        slice_obj["overall_mae"] = float(np.mean(mae_per_motor))
        slice_obj["per_motor_mae"] = [
            {"motor_idx": int(i), "motor_name": motor_names[i], "mae": float(mae_per_motor[i])}
            for i in range(d)
        ]

        region_mae: Dict[str, float] = {}
        for region_name, idxs in region_indices.items():
            idx_arr = np.asarray(list(idxs), dtype=np.int64)
            if idx_arr.size == 0:
                continue
            region_mae[str(region_name)] = float(np.mean(s_abs_err[:, idx_arr]))
        slice_obj["per_region_mae"] = region_mae
        out_slices[slice_name] = slice_obj

    return {
        "enabled": True,
        "thresholds_deg": {
            "frontal_max_deg": float(frontal_max_deg),
            "moderate_max_deg": float(moderate_max_deg),
        },
        "pose_strength_definition": "max(abs(yaw), abs(pitch), abs(roll))",
        "valid_pose_samples": int(np.sum(valid_pose)),
        "total_samples": int(n),
        "valid_pose_ratio": float(np.sum(valid_pose) / max(n, 1)),
        "slices": out_slices,
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_indices: Mapping[str, Iterable[int]],
    abs_error_percentile: float = 95.0,
    out_range_lo: float = 0.0,
    out_range_hi: float = 1.0,
    motor_names: List[str] | None = None,
    out_of_range_top_k: int = 10,
) -> Dict[str, object]:
    """计算回归指标、区域/电机排行、越界统计与样本误差摘要。"""
    if y_true.shape != y_pred.shape:
        raise RuntimeError(f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.ndim != 2:
        raise RuntimeError(f"expect 2D arrays, got y_true.ndim={y_true.ndim}")

    n, d = y_true.shape
    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err * err

    if motor_names is None:
        motor_names = load_motor_names(metrics_cfg=None, dim=d)
    if len(motor_names) < d:
        motor_names = motor_names + [f"motor_{i:02d}" for i in range(len(motor_names), d)]
    motor_names = motor_names[:d]

    mae_per_dim = np.mean(abs_err, axis=0)
    rmse_per_dim = np.sqrt(np.mean(sq_err, axis=0))
    mae = float(np.mean(mae_per_dim))
    rmse = float(np.mean(rmse_per_dim))

    r2_per_dim, r2_valid_dims = _r2_per_dim(y_true, y_pred)
    ev_per_dim, ev_valid_dims = _explained_variance_per_dim(y_true, y_pred)
    r2 = float(np.nanmean(r2_per_dim)) if r2_valid_dims > 0 else None
    explained_variance = float(np.nanmean(ev_per_dim)) if ev_valid_dims > 0 else None

    pctl = float(abs_error_percentile)
    p95_abs_err_per_dim = np.percentile(abs_err, pctl, axis=0)
    max_abs_err_per_dim = np.max(abs_err, axis=0)

    region_metrics: Dict[str, object] = {}
    for region_name, idxs in region_indices.items():
        idx_arr = np.asarray(list(idxs), dtype=np.int64)
        if idx_arr.size == 0:
            continue
        r_abs = abs_err[:, idx_arr]
        r_sq = sq_err[:, idx_arr]
        region_metrics[str(region_name)] = {
            "indices": idx_arr.tolist(),
            "mae": float(np.mean(r_abs)),
            "rmse": float(np.sqrt(np.mean(r_sq))),
            "p95_abs_err": float(np.percentile(r_abs, pctl)),
            "max_abs_err": float(np.max(r_abs)),
        }

    boundary = compute_boundary_violation_metrics(y_pred=y_pred, lo=out_range_lo, hi=out_range_hi)

    sample_mae = np.mean(abs_err, axis=1)

    region_error_ranking = {
        "by_mae": _region_ranking(region_metrics, key="mae"),
        "by_rmse": _region_ranking(region_metrics, key="rmse"),
    }
    motor_error_ranking = {
        "by_mae": _motor_ranking(mae_per_dim, "mae", motor_names),
        "by_rmse": _motor_ranking(rmse_per_dim, "rmse", motor_names),
        "by_p95_abs_err": _motor_ranking(p95_abs_err_per_dim, "p95_abs_err", motor_names),
    }
    oor_ratio_per_dim_arr = np.asarray(boundary["ratio_per_dim"], dtype=np.float64)
    oor_items = _motor_ranking(oor_ratio_per_dim_arr, "out_of_range_ratio", motor_names)
    out_of_range_motor_ranking = {
        "top_k": int(max(1, out_of_range_top_k)),
        "items": [
            {
                **row,
                "out_of_range_count": int(boundary["count_per_dim"][row["motor_idx"]]),
            }
            for row in oor_items[: int(max(1, out_of_range_top_k))]
        ],
    }

    return {
        "samples": int(n),
        "dim": int(d),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "explained_variance": explained_variance,
        "r2_valid_dims": r2_valid_dims,
        "explained_variance_valid_dims": ev_valid_dims,
        "mae_per_dim": [float(v) for v in mae_per_dim.tolist()],
        "rmse_per_dim": [float(v) for v in rmse_per_dim.tolist()],
        "r2_per_dim": _jsonable_float_list(r2_per_dim),
        "explained_variance_per_dim": _jsonable_float_list(ev_per_dim),
        "p95_abs_err_percentile": pctl,
        "p95_abs_err_per_dim": [float(v) for v in p95_abs_err_per_dim.tolist()],
        "max_abs_err_per_dim": [float(v) for v in max_abs_err_per_dim.tolist()],
        "p95_abs_err_mean": float(np.mean(p95_abs_err_per_dim)),
        "max_abs_err_mean": float(np.mean(max_abs_err_per_dim)),
        "region_metrics": region_metrics,
        "region_error_ranking": region_error_ranking,
        "motor_error_ranking": motor_error_ranking,
        "out_of_range_motor_ranking": out_of_range_motor_ranking,
        "out_of_range": {
            **boundary,
            "within_range_ratio": float(1.0 - float(boundary["ratio"])),
        },
        "boundary_loss": {
            "mean": float(boundary["boundary_loss_mean"]),
            "p95": float(boundary["boundary_loss_p95"]),
            "per_dim": [float(v) for v in boundary["boundary_loss_per_dim"]],
        },
        "sample_error_summary": {
            "sample_mae_mean": float(np.mean(sample_mae)),
            "sample_mae_p95": float(np.percentile(sample_mae, 95.0)),
            "sample_mae_max": float(np.max(sample_mae)),
        },
    }
