#!/usr/bin/env python3
"""latent->motor 可解释性与结构一致性分析脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch
import yaml

from data_utils import build_xy_from_split, load_latent24_map, load_target30_map
from eval_metrics import load_motor_names, load_motor_region_indices
from model import MotorRegressorMLP
from run_utils import resolve_eval_ckpt_path


# latent24 的默认区域划分（基于 build_latent24_from_abs_rel.py 的拼接顺序）
DEFAULT_LATENT_REGION_INDICES: Dict[str, List[int]] = {
    "brow": [0, 1, 2, 3],
    "eye": [4, 5, 6, 7],
    "mouth": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    "jaw": [18, 19, 20, 21],
    "global": [22, 23],
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="Explainability & structure consistency analysis for latent24->motor30.")
    p.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    p.add_argument("--ckpt", type=Path, default=None, help="Override config.eval and use explicit checkpoint path")
    p.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Default from config.explainability.split (fallback=test)",
    )
    return p.parse_args()


def _to_index_list(v: object) -> List[int]:
    """将对象校验并转换为 list[int]。"""
    if not isinstance(v, list):
        raise RuntimeError(f"expect list[int], got: {type(v)}")
    out: List[int] = []
    for x in v:
        if not isinstance(x, int):
            raise RuntimeError(f"index must be int, got: {x!r}")
        out.append(x)
    return out


def load_latent_region_indices(explain_cfg: Mapping[str, object] | None, dim: int) -> Dict[str, List[int]]:
    """读取 latent 区域索引配置并做合法性校验。"""
    region_cfg = None
    if isinstance(explain_cfg, Mapping):
        region_cfg = explain_cfg.get("latent_region_indices")

    if isinstance(region_cfg, Mapping):
        region_map = {str(k): _to_index_list(v) for k, v in region_cfg.items()}
    else:
        region_map = {k: list(v) for k, v in DEFAULT_LATENT_REGION_INDICES.items()}

    for region, idxs in region_map.items():
        if len(idxs) == 0:
            raise RuntimeError(f"latent region '{region}' has empty indices")
        for i in idxs:
            if i < 0 or i >= dim:
                raise RuntimeError(f"latent region '{region}' index out of range: {i}, dim={dim}")
    return region_map


def _corr_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """计算 x(latent) 与 y(motor) 的皮尔逊相关矩阵 [x_dim, y_dim]。"""
    if x.ndim != 2 or y.ndim != 2:
        raise RuntimeError(f"corr input must be 2D, got x={x.ndim}, y={y.ndim}")
    if x.shape[0] != y.shape[0]:
        raise RuntimeError(f"corr input sample mismatch: x={x.shape}, y={y.shape}")

    xc = x - np.mean(x, axis=0, keepdims=True)
    yc = y - np.mean(y, axis=0, keepdims=True)
    x_std = np.std(xc, axis=0, keepdims=True)
    y_std = np.std(yc, axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-12, np.nan, x_std)
    y_std = np.where(y_std < 1e-12, np.nan, y_std)

    xz = xc / x_std
    yz = yc / y_std
    corr = (xz.T @ yz) / max(x.shape[0] - 1, 1)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    return corr


def _region_block_stats(corr: np.ndarray) -> Dict[str, float]:
    """对某个相关矩阵子块计算统计摘要。"""
    abs_corr = np.abs(corr)
    return {
        "mean_corr": float(np.mean(corr)),
        "mean_abs_corr": float(np.mean(abs_corr)),
        "median_abs_corr": float(np.median(abs_corr)),
        "p95_abs_corr": float(np.percentile(abs_corr, 95.0)),
        "max_abs_corr": float(np.max(abs_corr)),
    }


def build_region_corr_stats(
    corr_lm: np.ndarray,
    latent_regions: Mapping[str, Iterable[int]],
    motor_regions: Mapping[str, Iterable[int]],
) -> Dict[str, object]:
    """统计区域相关性（同名区域统计 + 全区域组合矩阵统计）。"""
    pair_stats: Dict[str, object] = {}
    matrix_stats: Dict[str, object] = {}

    for l_name, l_idxs in latent_regions.items():
        li = np.asarray(list(l_idxs), dtype=np.int64)
        matrix_stats[l_name] = {}
        for m_name, m_idxs in motor_regions.items():
            mi = np.asarray(list(m_idxs), dtype=np.int64)
            block = corr_lm[np.ix_(li, mi)]
            matrix_stats[l_name][m_name] = _region_block_stats(block)

    common = sorted(set(latent_regions.keys()) & set(motor_regions.keys()))
    for name in common:
        li = np.asarray(list(latent_regions[name]), dtype=np.int64)
        mi = np.asarray(list(motor_regions[name]), dtype=np.int64)
        block = corr_lm[np.ix_(li, mi)]
        pair_stats[name] = {
            "latent_indices": li.tolist(),
            "motor_indices": mi.tolist(),
            **_region_block_stats(block),
        }

    return {
        "matched_region_stats": pair_stats,
        "all_region_pair_matrix_stats": matrix_stats,
    }


def _predict_in_batches(model: torch.nn.Module, x_np: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """按 batch 做前向推理并返回拼接后的预测数组。"""
    preds: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for s in range(0, x_np.shape[0], batch_size):
            e = min(s + batch_size, x_np.shape[0])
            xb = torch.from_numpy(x_np[s:e]).to(device=device, dtype=torch.float32)
            yb = model(xb).detach().cpu().numpy()
            preds.append(yb)
    return np.vstack(preds).astype(np.float64)


def _rank_motor_delta(delta_per_motor: np.ndarray, motor_names: List[str], top_k: int) -> List[Dict[str, object]]:
    """对电机变化幅度做降序排名并截取 top-k。"""
    order = np.argsort(-delta_per_motor)
    out: List[Dict[str, object]] = []
    for idx in order[: max(1, top_k)].tolist():
        out.append(
            {
                "motor_idx": int(idx),
                "motor_name": motor_names[idx],
                "mean_abs_delta": float(delta_per_motor[idx]),
            }
        )
    return out


def perturbation_sensitivity_analysis(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    latent_regions: Mapping[str, Iterable[int]],
    motor_names: List[str],
    batch_size: int,
    noise_std_scale: float,
    random_seed: int,
    top_k: int,
) -> Dict[str, object]:
    """做区域置零与扰动实验，计算电机输出变化敏感性。"""
    base_pred = _predict_in_batches(model, x, device=device, batch_size=batch_size)

    rng = np.random.default_rng(random_seed)
    x_std = np.std(x, axis=0).astype(np.float64)

    out: Dict[str, object] = {
        "noise_std_scale": float(noise_std_scale),
        "random_seed": int(random_seed),
        "regions": {},
    }

    for region, idxs in latent_regions.items():
        idx_arr = np.asarray(list(idxs), dtype=np.int64)

        x_zero = x.copy()
        x_zero[:, idx_arr] = 0.0
        pred_zero = _predict_in_batches(model, x_zero, device=device, batch_size=batch_size)
        delta_zero = np.mean(np.abs(pred_zero - base_pred), axis=0)

        x_noise = x.copy()
        std_vec = x_std[idx_arr] * float(noise_std_scale)
        noise = rng.normal(loc=0.0, scale=std_vec, size=(x.shape[0], idx_arr.shape[0]))
        x_noise[:, idx_arr] = x_noise[:, idx_arr] + noise
        pred_noise = _predict_in_batches(model, x_noise, device=device, batch_size=batch_size)
        delta_noise = np.mean(np.abs(pred_noise - base_pred), axis=0)

        out["regions"][region] = {
            "latent_indices": idx_arr.tolist(),
            "zeroing": {
                "mean_abs_delta_per_motor": [float(v) for v in delta_zero.tolist()],
                "top_motors": _rank_motor_delta(delta_zero, motor_names, top_k=top_k),
            },
            "noise": {
                "mean_abs_delta_per_motor": [float(v) for v in delta_noise.tolist()],
                "top_motors": _rank_motor_delta(delta_noise, motor_names, top_k=top_k),
            },
        }

    return out


def save_corr_heatmap_png(corr: np.ndarray, out_png: Path) -> bool:
    """尝试保存相关矩阵热图；缺少 matplotlib 时返回 False。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("Latent24 vs Motor30 Correlation")
    ax.set_xlabel("motor index")
    ax.set_ylabel("latent index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return True


def resolve_split_path(data_cfg: Mapping[str, object], split: str) -> Path:
    """根据 split 名称解析配置中的 split 文件路径。"""
    key = f"{split}_split"
    if key not in data_cfg:
        raise RuntimeError(f"config.data missing split path: {key}")
    return Path(str(data_cfg[key]))


def resolve_device(device_cfg: str) -> torch.device:
    """根据配置字符串与 CUDA 可用性选择运行设备。"""
    if device_cfg.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    """主流程：加载数据与模型，输出相关性和扰动敏感性分析文件。"""
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    explain_cfg = cfg.get("explainability", {})
    split = args.split or str(explain_cfg.get("split", "test"))
    if split not in {"train", "val", "test"}:
        raise RuntimeError(f"invalid split: {split}")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    metrics_cfg = cfg.get("metrics", {})

    device = resolve_device(str(train_cfg["device"]))
    ckpt_path, eval_output_dir, run_name = resolve_eval_ckpt_path(cfg, args.ckpt)

    split_path = resolve_split_path(data_cfg, split)
    latent_map = load_latent24_map(Path(str(data_cfg["latent_file"])))
    target_map = load_target30_map(Path(str(data_cfg["target_file"])))
    x, y = build_xy_from_split(split_path, latent_map, target_map)

    corr_lm = _corr_matrix(x.astype(np.float64), y.astype(np.float64))

    motor_regions = load_motor_region_indices(metrics_cfg=metrics_cfg, dim=y.shape[1])
    latent_regions = load_latent_region_indices(explain_cfg=explain_cfg, dim=x.shape[1])
    motor_names = load_motor_names(metrics_cfg=metrics_cfg, dim=y.shape[1])

    region_corr = build_region_corr_stats(
        corr_lm=corr_lm,
        latent_regions=latent_regions,
        motor_regions=motor_regions,
    )

    # grouped-head 输出分组（不填则使用模型内默认的 brow/eye/mouth/jaw 分组）
    grouped_head_cfg = model_cfg.get("grouped_head", {})
    motor_region_indices = grouped_head_cfg.get("motor_region_indices") if isinstance(grouped_head_cfg, dict) else None

    model = MotorRegressorMLP(
        input_dim=int(model_cfg["input_dim"]),
        hidden1=int(model_cfg["hidden_dim1"]),
        hidden2=int(model_cfg["hidden_dim2"]),
        output_dim=int(model_cfg["output_dim"]),
        motor_region_indices=motor_region_indices,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    perturb_cfg = explain_cfg.get("perturbation", {}) if isinstance(explain_cfg, Mapping) else {}
    sens = perturbation_sensitivity_analysis(
        model=model,
        x=x.astype(np.float64),
        device=device,
        latent_regions=latent_regions,
        motor_names=motor_names,
        batch_size=int(train_cfg.get("batch_size", 256)),
        noise_std_scale=float(perturb_cfg.get("noise_std_scale", 1.0)),
        random_seed=int(perturb_cfg.get("random_seed", 42)),
        top_k=int(perturb_cfg.get("top_k", 10)),
    )

    run_dir = eval_output_dir / "explainability" / f"{split}_{Path(ckpt_path).stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    corr_csv = run_dir / "latent_motor_corr.csv"
    corr_npy = run_dir / "latent_motor_corr.npy"
    region_json = run_dir / "region_corr_stats.json"
    sens_json = run_dir / "perturbation_sensitivity.json"
    summary_json = run_dir / "explainability_summary.json"

    np.save(corr_npy, corr_lm)
    np.savetxt(corr_csv, corr_lm, delimiter=",", fmt="%.8f")
    region_json.write_text(json.dumps(region_corr, ensure_ascii=False, indent=2), encoding="utf-8")
    sens_json.write_text(json.dumps(sens, ensure_ascii=False, indent=2), encoding="utf-8")

    heatmap_png = run_dir / "latent_motor_corr_heatmap.png"
    heatmap_ok = save_corr_heatmap_png(corr_lm, heatmap_png)

    summary = {
        "split": split,
        "samples": int(x.shape[0]),
        "latent_dim": int(x.shape[1]),
        "motor_dim": int(y.shape[1]),
        "device": str(device),
        "ckpt": str(ckpt_path),
        "run_name": run_name,
        "split_path": str(split_path),
        "latent_regions": latent_regions,
        "motor_regions": motor_regions,
        "paths": {
            "corr_csv": str(corr_csv),
            "corr_npy": str(corr_npy),
            "region_corr_stats": str(region_json),
            "perturbation_sensitivity": str(sens_json),
            "corr_heatmap_png": str(heatmap_png) if heatmap_ok else None,
        },
        "heatmap_saved": bool(heatmap_ok),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] explainability split={split} samples={x.shape[0]} ckpt={ckpt_path}")
    print(f"[DONE] out_dir={run_dir}")
    print(f"[DONE] summary={summary_json}")


if __name__ == "__main__":
    main()
