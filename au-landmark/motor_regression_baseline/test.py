#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data_utils import XYDataset, build_xy_from_split, load_latent24_map, load_target30_map
from model import MotorRegressorMLP


# 读取测试脚本参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test baseline regressor on test split.")
    p.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    p.add_argument("--ckpt", type=Path, default=None, help="Default: <output_dir>/best.pt")
    return p.parse_args()


# 根据配置自动选择 GPU/CPU
def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    # 读取配置与 checkpoint 路径
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = resolve_device(str(cfg["train"]["device"]))
    output_dir = Path(cfg["train"]["output_dir"])
    ckpt_path = args.ckpt or (output_dir / "best.pt")

    # 加载测试集数据
    latent_map = load_latent24_map(Path(cfg["data"]["latent_file"]))
    target_map = load_target30_map(Path(cfg["data"]["target_file"]))
    x_test, y_test = build_xy_from_split(Path(cfg["data"]["test_split"]), latent_map, target_map)

    ds = XYDataset(x_test, y_test)
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

    # 测试：统计每个维度 MAE 及总体 MAE
    abs_err_sum = np.zeros(30, dtype=np.float64)
    n = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            err = torch.abs(pred - y).detach().cpu().numpy()
            abs_err_sum += err.sum(axis=0)
            n += err.shape[0]

    # 保存测试结果
    mae_per_dim = abs_err_sum / max(n, 1)
    mae = float(np.mean(mae_per_dim))
    metrics = {
        "split": "test",
        "samples": int(n),
        "mae": mae,
        "mae_per_dim": mae_per_dim.tolist(),
        "ckpt": str(ckpt_path),
        "device": str(device),
    }
    out_json = output_dir / "test_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] test_mae={mae:.6f} samples={n}")
    print(f"[DONE] metrics={out_json}")


if __name__ == "__main__":
    main()
