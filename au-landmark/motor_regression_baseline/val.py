#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data_utils import XYDataset, build_xy_from_split, load_latent24_map, load_target30_map
from eval_metrics import collect_predictions, compute_regression_metrics, load_motor_region_indices
from model import MotorRegressorMLP


# 读取验证脚本参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate baseline regressor on val split.")
    p.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    p.add_argument("--ckpt", type=Path, default=None, help="Default: <output_dir>/best.pt")
    return p.parse_args()


# 根据配置自动选择 GPU/CPU
def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    # 读取配置和 checkpoint 路径
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = resolve_device(str(cfg["train"]["device"]))
    output_dir = Path(cfg["train"]["output_dir"])
    ckpt_path = args.ckpt or (output_dir / "best.pt")
    metrics_cfg = cfg.get("metrics", {})

    # 加载验证集
    latent_map = load_latent24_map(Path(cfg["data"]["latent_file"]))
    target_map = load_target30_map(Path(cfg["data"]["target_file"]))
    x_val, y_val = build_xy_from_split(Path(cfg["data"]["val_split"]), latent_map, target_map)

    ds = XYDataset(x_val, y_val)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    # 构建模型并加载权重
    model = MotorRegressorMLP(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden1=int(cfg["model"]["hidden_dim1"]),
        hidden2=int(cfg["model"]["hidden_dim2"]),
        output_dim=int(cfg["model"]["output_dim"]),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 评估：RMSE/R2/EV/P95+Max误差/分区域/越界率
    y_true, y_pred = collect_predictions(model, loader, device)
    region_indices = load_motor_region_indices(metrics_cfg=metrics_cfg, dim=y_true.shape[1])
    metric_dict = compute_regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
        region_indices=region_indices,
        abs_error_percentile=float(metrics_cfg.get("abs_error_percentile", 95.0)),
        out_range_lo=float(metrics_cfg.get("out_range_lo", 0.0)),
        out_range_hi=float(metrics_cfg.get("out_range_hi", 1.0)),
    )

    # 保存验证结果
    metrics = {
        "split": "val",
        **metric_dict,
        "ckpt": str(ckpt_path),
        "device": str(device),
        "region_indices": region_indices,
    }
    out_json = output_dir / "val_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[DONE] val "
        f"mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} "
        f"r2={metrics['r2'] if metrics['r2'] is not None else 'NA'} "
        f"ev={metrics['explained_variance'] if metrics['explained_variance'] is not None else 'NA'} "
        f"oor={metrics['out_of_range']['ratio']:.6f} samples={metrics['samples']}"
    )
    print(f"[DONE] metrics={out_json}")


if __name__ == "__main__":
    main()
