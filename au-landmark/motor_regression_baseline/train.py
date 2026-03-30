#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data_utils import XYDataset, build_xy_from_split, load_latent24_map, load_target30_map
from model import MotorRegressorMLP
from run_utils import resolve_train_output_dir


def _as_bool(v: object, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def resolve_boundary_train_cfg(cfg: dict) -> dict:
    """解析边界约束训练配置。"""
    metrics_cfg = cfg.get("metrics", {})
    boundary_cfg = cfg.get("boundary", {})
    train_boundary_cfg = boundary_cfg.get("train", {}) if isinstance(boundary_cfg, dict) else {}
    lo = float(boundary_cfg.get("lo", metrics_cfg.get("out_range_lo", 0.0)))
    hi = float(boundary_cfg.get("hi", metrics_cfg.get("out_range_hi", 1.0)))
    if hi <= lo:
        raise RuntimeError(f"invalid boundary range: lo={lo}, hi={hi}")
    return {
        "lo": lo,
        "hi": hi,
        "clip_predictions_in_eval": _as_bool(boundary_cfg.get("clip_predictions_in_eval", True), default=True),
        "clamp_for_task_loss": _as_bool(train_boundary_cfg.get("clamp_for_task_loss", True), default=True),
        "enable_boundary_loss": _as_bool(train_boundary_cfg.get("enable_boundary_loss", True), default=True),
        "boundary_loss_weight": float(train_boundary_cfg.get("boundary_loss_weight", 0.1)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline latent24->motor30 regressor.")
    p.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_boundary_loss_torch(pred: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """边界惩罚：超出上下界的幅度均值。"""
    return (torch.relu(lo - pred) + torch.relu(pred - hi)).mean()


def evaluate_mae(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    clip_predictions: bool,
    lo: float,
    hi: float,
) -> float:
    model.eval()
    mae_sum = 0.0
    n = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            if clip_predictions:
                pred = torch.clamp(pred, min=lo, max=hi)
            mae = torch.mean(torch.abs(pred - y), dim=1)
            mae_sum += float(mae.sum().item())
            n += int(x.shape[0])
    return mae_sum / max(n, 1)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    boundary_cfg = resolve_boundary_train_cfg(cfg)

    device = resolve_device(str(cfg["train"]["device"]))
    output_dir, run_name = resolve_train_output_dir(cfg["train"])

    latent_file = Path(cfg["data"]["latent_file"])
    target_file = Path(cfg["data"]["target_file"])
    train_split = Path(cfg["data"]["train_split"])
    val_split = Path(cfg["data"]["val_split"])

    print(f"[INFO] device={device}")
    if run_name is not None:
        print(f"[INFO] run_name={run_name}")
    print(f"[INFO] output_dir={output_dir}")
    print(
        "[INFO] boundary "
        f"lo={boundary_cfg['lo']} hi={boundary_cfg['hi']} "
        f"eval_clip={boundary_cfg['clip_predictions_in_eval']} "
        f"clamp_for_task_loss={boundary_cfg['clamp_for_task_loss']} "
        f"boundary_loss={boundary_cfg['enable_boundary_loss']} "
        f"weight={boundary_cfg['boundary_loss_weight']}"
    )
    print("[INFO] loading latent/target maps...")
    latent_map = load_latent24_map(latent_file)
    target_map = load_target30_map(target_file)

    print("[INFO] building train/val arrays...")
    x_train, y_train = build_xy_from_split(train_split, latent_map, target_map)
    x_val, y_val = build_xy_from_split(val_split, latent_map, target_map)
    print(f"[INFO] train={x_train.shape} val={x_val.shape}")

    train_ds = XYDataset(x_train, y_train)
    val_ds = XYDataset(x_val, y_val)

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = MotorRegressorMLP(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden1=int(cfg["model"]["hidden_dim1"]),
        hidden2=int(cfg["model"]["hidden_dim2"]),
        output_dim=int(cfg["model"]["output_dim"]),
    ).to(device)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["early_stopping"]["patience"])
    min_delta = float(cfg["train"]["early_stopping"]["min_delta"])

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history_path = output_dir / "train_history.csv"
    best_ckpt = output_dir / "best.pt"
    last_ckpt = output_dir / "last.pt"

    with history_path.open("w", encoding="utf-8", newline="") as f_hist:
        writer = csv.writer(f_hist)
        writer.writerow(["epoch", "train_loss_l1", "train_boundary_loss", "train_total_loss", "val_mae"])

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            task_loss_sum = 0.0
            boundary_loss_sum = 0.0
            total_loss_sum = 0.0
            n = 0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                raw_pred = model(x)
                pred_for_task = (
                    torch.clamp(raw_pred, min=boundary_cfg["lo"], max=boundary_cfg["hi"])
                    if boundary_cfg["clamp_for_task_loss"]
                    else raw_pred
                )
                task_loss = criterion(pred_for_task, y)
                boundary_loss = compute_boundary_loss_torch(raw_pred, lo=boundary_cfg["lo"], hi=boundary_cfg["hi"])
                if boundary_cfg["enable_boundary_loss"]:
                    loss = task_loss + float(boundary_cfg["boundary_loss_weight"]) * boundary_loss
                else:
                    loss = task_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                bs = int(x.shape[0])
                task_loss_sum += float(task_loss.item()) * bs
                boundary_loss_sum += float(boundary_loss.item()) * bs
                total_loss_sum += float(loss.item()) * bs
                n += bs

            train_task_loss = task_loss_sum / max(n, 1)
            train_boundary_loss = boundary_loss_sum / max(n, 1)
            train_total_loss = total_loss_sum / max(n, 1)
            val_mae = evaluate_mae(
                model,
                val_loader,
                device,
                clip_predictions=bool(boundary_cfg["clip_predictions_in_eval"]),
                lo=float(boundary_cfg["lo"]),
                hi=float(boundary_cfg["hi"]),
            )
            writer.writerow([epoch, train_task_loss, train_boundary_loss, train_total_loss, val_mae])
            f_hist.flush()

            improved = val_mae < (best_val - min_delta)
            if improved:
                best_val = val_mae
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "best_val_mae": best_val,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                    },
                    best_ckpt,
                )
            else:
                bad_epochs += 1

            print(
                f"[EPOCH {epoch:03d}] "
                f"train_l1={train_task_loss:.6f} "
                f"train_boundary={train_boundary_loss:.6f} "
                f"train_total={train_total_loss:.6f} "
                f"val_mae={val_mae:.6f} best={best_val:.6f}"
            )

            if bad_epochs >= patience:
                print(f"[INFO] early stopping at epoch={epoch}, best_epoch={best_epoch}")
                break

        elapsed = time.time() - t0

    torch.save(
        {
            "epoch": epoch,
            "best_val_mae": best_val,
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        last_ckpt,
    )

    summary = {
        "best_val_mae": best_val,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
        "device": str(device),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "boundary_train_cfg": boundary_cfg,
        "paths": {
            "best_ckpt": str(best_ckpt),
            "last_ckpt": str(last_ckpt),
            "history_csv": str(history_path),
        },
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] best_val_mae={best_val:.6f} at epoch={best_epoch}")
    print(f"[DONE] output_dir={output_dir}")


if __name__ == "__main__":
    main()
