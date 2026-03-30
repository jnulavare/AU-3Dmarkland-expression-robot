#!/usr/bin/env python3
from __future__ import annotations

"""Test entry for compare1 direct regression (feature382 -> motor30)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data_utils import XYDataset, build_xy_from_split, load_feature_map, load_split_indices, load_target30_map
from eval_metrics import (
    analyze_error_vs_context,
    clip_predictions_to_range,
    collect_predictions,
    compute_boundary_violation_metrics,
    compute_pose_slice_mae_analysis,
    compute_regression_metrics,
    load_context_feature_arrays,
    load_motor_names,
    load_motor_region_indices,
)
from model import MotorRegressorMLP
from run_utils import resolve_eval_ckpt_path


def _as_bool(v: object, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def resolve_boundary_eval_cfg(cfg: dict) -> dict:
    # Keep eval boundary behavior aligned with training/validation scripts.
    """解析验证/测试阶段的边界后处理配置。"""
    metrics_cfg = cfg.get("metrics", {})
    boundary_cfg = cfg.get("boundary", {})
    lo = float(boundary_cfg.get("lo", metrics_cfg.get("out_range_lo", 0.0)))
    hi = float(boundary_cfg.get("hi", metrics_cfg.get("out_range_hi", 1.0)))
    if hi <= lo:
        raise RuntimeError(f"invalid boundary range: lo={lo}, hi={hi}")
    return {
        "lo": lo,
        "hi": hi,
        "clip_predictions_in_eval": _as_bool(boundary_cfg.get("clip_predictions_in_eval", True), default=True),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test baseline feature->motor30 regressor on test split.")
    p.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    p.add_argument("--ckpt", type=Path, default=None, help="Override config.eval and use explicit checkpoint path")
    return p.parse_args()


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = resolve_device(str(cfg["train"]["device"]))
    metrics_cfg = cfg.get("metrics", {})
    boundary_eval_cfg = resolve_boundary_eval_cfg(cfg)

    ckpt_path, output_dir, run_name = resolve_eval_ckpt_path(cfg, args.ckpt)

    # 1) Resolve inputs and load split arrays.
    data_cfg = cfg["data"]
    feature_file_cfg = data_cfg.get("feature_file")
    if feature_file_cfg is None:
        feature_file_cfg = data_cfg["latent_file"]
    feature_file = Path(str(feature_file_cfg))
    feature_map = load_feature_map(feature_file, expected_dim=int(cfg["model"]["input_dim"]))
    target_map = load_target30_map(Path(cfg["data"]["target_file"]))
    split_path = Path(cfg["data"]["test_split"])
    x_test, y_test = build_xy_from_split(split_path, feature_map, target_map)

    ds = XYDataset(x_test, y_test)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    # 2) Restore checkpoint and run forward pass.
    model = MotorRegressorMLP(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden1=int(cfg["model"]["hidden_dim1"]),
        hidden2=int(cfg["model"]["hidden_dim2"]),
        output_dim=int(cfg["model"]["output_dim"]),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_pred_raw = collect_predictions(model, loader, device)
    if boundary_eval_cfg["clip_predictions_in_eval"]:
        y_pred = clip_predictions_to_range(y_pred_raw, lo=boundary_eval_cfg["lo"], hi=boundary_eval_cfg["hi"])
    else:
        y_pred = y_pred_raw
    region_indices = load_motor_region_indices(metrics_cfg=metrics_cfg, dim=y_true.shape[1])
    motor_names = load_motor_names(metrics_cfg=metrics_cfg, dim=y_true.shape[1])
    metric_dict = compute_regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
        region_indices=region_indices,
        abs_error_percentile=float(metrics_cfg.get("abs_error_percentile", 95.0)),
        out_range_lo=float(boundary_eval_cfg["lo"]),
        out_range_hi=float(boundary_eval_cfg["hi"]),
        motor_names=motor_names,
        out_of_range_top_k=int(metrics_cfg.get("out_of_range_top_k", 10)),
    )
    metric_dict["boundary_constraint"] = {
        "clip_predictions_in_eval": bool(boundary_eval_cfg["clip_predictions_in_eval"]),
        "raw_prediction_boundary": compute_boundary_violation_metrics(
            y_pred=y_pred_raw,
            lo=float(boundary_eval_cfg["lo"]),
            hi=float(boundary_eval_cfg["hi"]),
        ),
        "final_prediction_boundary": compute_boundary_violation_metrics(
            y_pred=y_pred,
            lo=float(boundary_eval_cfg["lo"]),
            hi=float(boundary_eval_cfg["hi"]),
        ),
    }

    # 3) Optional context analyses: error-context + pose slices.
    context_cfg = metrics_cfg.get("error_context", {})
    pose_slice_cfg = metrics_cfg.get("pose_slice", {})
    need_pose_context = bool(context_cfg.get("enabled", True)) or bool(pose_slice_cfg.get("enabled", True))
    context = None
    if need_pose_context:
        split_indices = load_split_indices(split_path)
        if len(split_indices) != int(y_true.shape[0]):
            raise RuntimeError(
                f"split size mismatch for context analysis: split={len(split_indices)} eval={y_true.shape[0]}"
            )

        latent_parent = feature_file.parent
        rel_file = Path(context_cfg.get("rel_file", str(latent_parent / "REL_input_vec_X2C_gpu.csv.gz")))
        abs_file = Path(context_cfg.get("abs_file", str(latent_parent / "ABS_input_vec_X2C_gpu.csv.gz")))
        context = load_context_feature_arrays(split_indices=split_indices, rel_file=rel_file, abs_file=abs_file)

    if bool(context_cfg.get("enabled", True)):
        if context is None:
            raise RuntimeError("context should not be None when error_context is enabled")

        sample_mae = np.mean(np.abs(y_pred - y_true), axis=1)
        metric_dict["error_context_analysis"] = analyze_error_vs_context(
            sample_mae=sample_mae,
            context=context,
            bins=int(context_cfg.get("bins", 10)),
        )
    else:
        metric_dict["error_context_analysis"] = {"enabled": False}

    if bool(pose_slice_cfg.get("enabled", True)):
        if context is None:
            raise RuntimeError("context should not be None when pose_slice is enabled")
        metric_dict["pose_slice_mae_analysis"] = compute_pose_slice_mae_analysis(
            y_true=y_true,
            y_pred=y_pred,
            yaw=np.asarray(context.get("yaw", np.array([])), dtype=np.float64),
            pitch=np.asarray(context.get("pitch", np.array([])), dtype=np.float64),
            roll=np.asarray(context.get("roll", np.array([])), dtype=np.float64),
            region_indices=region_indices,
            motor_names=motor_names,
            frontal_max_deg=float(pose_slice_cfg.get("frontal_max_deg", 10.0)),
            moderate_max_deg=float(pose_slice_cfg.get("moderate_max_deg", 25.0)),
        )
    else:
        metric_dict["pose_slice_mae_analysis"] = {"enabled": False}

    metrics = {
        "split": "test",
        **metric_dict,
        "ckpt": str(ckpt_path),
        "run_name": run_name,
        "device": str(device),
        "region_indices": region_indices,
        "motor_names": motor_names,
    }
    out_json = output_dir / "test_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[DONE] test "
        f"run={run_name if run_name is not None else 'none'} "
        f"mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} "
        f"r2={metrics['r2'] if metrics['r2'] is not None else 'NA'} "
        f"ev={metrics['explained_variance'] if metrics['explained_variance'] is not None else 'NA'} "
        f"oor={metrics['out_of_range']['ratio']:.6f} "
        f"raw_oor={metrics['boundary_constraint']['raw_prediction_boundary']['ratio']:.6f} "
        f"samples={metrics['samples']}"
    )
    if metrics["pose_slice_mae_analysis"].get("enabled", False):
        slices = metrics["pose_slice_mae_analysis"]["slices"]
        print(
            "[POSE] "
            f"frontal_mae={slices['frontal']['overall_mae']} "
            f"moderate_mae={slices['moderate_pose']['overall_mae']} "
            f"extreme_mae={slices['extreme_pose']['overall_mae']}"
        )
    print(f"[DONE] ckpt={ckpt_path}")
    print(f"[DONE] metrics={out_json}")


if __name__ == "__main__":
    main()
