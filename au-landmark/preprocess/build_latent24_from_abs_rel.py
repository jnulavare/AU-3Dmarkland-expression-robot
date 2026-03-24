#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# 区域 AU 分组定义
BROW_AU = ["AU1", "AU2", "AU4"]
EYE_AU = ["AU5", "AU6", "AU7"]
MOUTH_AU = ["AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25"]
JAW_AU = ["AU26"]

BROW_DIST = [
    "brow_left_eye_dist",
    "brow_right_eye_dist",
    "brow_inner_dist",
    "brow_outer_height_diff",
]
EYE_DIST = [
    "eye_left_open",
    "eye_right_open",
    "eye_left_width",
    "eye_right_width",
    "eye_left_ratio",
    "eye_right_ratio",
]
MOUTH_DIST = [
    "mouth_width",
    "mouth_open",
    "mouth_left_corner_to_nose",
    "mouth_right_corner_to_nose",
    "mouth_left_corner_raise",
    "mouth_right_corner_raise",
    "upper_lip_to_lower_lip",
    "upper_lip_to_nose",
    "lower_lip_to_chin",
    "mouth_center_to_nose",
    "mouth_center_to_chin",
]
JAW_DIST = ["jaw_open", "chin_to_nose", "chin_to_upper_lip"]


# 读取 latent 构建参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 24D latent from ABS(192) + REL(190)")
    p.add_argument("--abs-file", type=Path, default=Path(r"D:\code\test2\x2c_data_bundle\ABS_input_vec_X2C_gpu.csv.gz"))
    p.add_argument("--rel-file", type=Path, default=Path(r"D:\code\test2\x2c_data_bundle\REL_input_vec_X2C_gpu.csv.gz"))
    p.add_argument("--output", type=Path, default=Path(r"D:\code\test2\x2c_data_bundle\LATENT24_X2C_gpu.csv.gz"))
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--log-every", type=int, default=20000)
    p.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-model", action="store_true")
    return p.parse_args()


# 安全浮点转换（处理空值和 NaN）
def safe_float(v: object) -> float:
    if v is None:
        return 0.0
    if isinstance(v, float):
        if math.isnan(v):
            return 0.0
        return v
    s = str(v).strip()
    if s == "":
        return 0.0
    x = float(s)
    if math.isnan(x):
        return 0.0
    return x


# 按输入后缀自动选择普通文本或 gzip 文本读写
def open_text(path: Path, mode: str):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def lmk_cols_abs(start_idx: int, end_idx: int) -> List[str]:
    cols: List[str] = []
    for i in range(start_idx, end_idx + 1):
        cols.extend([f"lmk_abs_norm_{i:02d}_x", f"lmk_abs_norm_{i:02d}_y", f"lmk_abs_norm_{i:02d}_z"])
    return cols


def lmk_cols_rel(start_idx: int, end_idx: int) -> List[str]:
    cols: List[str] = []
    for i in range(start_idx, end_idx + 1):
        cols.extend([f"lmk_rel_{i:02d}_x", f"lmk_rel_{i:02d}_y", f"lmk_rel_{i:02d}_z"])
    return cols


def au_abs_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_abs_intensity" for n in names]


def au_rel_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_rel" for n in names]


def dist_rel_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_rel" for n in names]


# 依据 ABS/REL 字段拼接每个区域输入维度
def build_region_columns() -> Dict[str, Dict[str, List[str]]]:
    # Selected landmark order in ABS/REL files:
    # 0-9 brows, 10-21 eyes, 22-41 mouth, 42-49 jaw
    out = {
        "brow_abs": au_abs_cols(BROW_AU) + lmk_cols_abs(0, 9) + BROW_DIST,
        "eye_abs": au_abs_cols(EYE_AU) + lmk_cols_abs(10, 21) + EYE_DIST,
        "mouth_abs": au_abs_cols(MOUTH_AU) + lmk_cols_abs(22, 41) + MOUTH_DIST,
        "jaw_abs": au_abs_cols(JAW_AU) + lmk_cols_abs(42, 49) + JAW_DIST,
        "global_abs": ["yaw", "pitch", "roll"],
        "brow_rel": au_rel_cols(BROW_AU) + lmk_cols_rel(0, 9) + dist_rel_cols(BROW_DIST),
        "eye_rel": au_rel_cols(EYE_AU) + lmk_cols_rel(10, 21) + dist_rel_cols(EYE_DIST),
        "mouth_rel": au_rel_cols(MOUTH_AU) + lmk_cols_rel(22, 41) + dist_rel_cols(MOUTH_DIST),
        "jaw_rel": au_rel_cols(JAW_AU) + lmk_cols_rel(42, 49) + dist_rel_cols(JAW_DIST),
        "global_rel": ["ENERGY_rel"],
    }
    return {"regions": out}


# 区域编码器：两层 MLP（Linear-ReLU-Linear-ReLU）
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ABS/REL 门控融合模块
class GateFusion(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.gate = nn.Linear(feat_dim * 2, feat_dim)

    def forward(self, h_abs: torch.Tensor, h_rel: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([h_abs, h_rel], dim=1)))
        return g * h_abs + (1.0 - g) * h_rel


# 总模型：区域编码 -> 门控融合 -> 分区投影 -> 24维 latent
class RegionLatent24(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoders
        self.brow_abs_enc = MLPEncoder(37, 32, 16)
        self.brow_rel_enc = MLPEncoder(37, 32, 16)
        self.eye_abs_enc = MLPEncoder(45, 32, 16)
        self.eye_rel_enc = MLPEncoder(45, 32, 16)
        self.mouth_abs_enc = MLPEncoder(79, 64, 32)
        self.mouth_rel_enc = MLPEncoder(79, 64, 32)
        self.jaw_abs_enc = MLPEncoder(28, 32, 16)
        self.jaw_rel_enc = MLPEncoder(28, 32, 16)
        self.global_abs_enc = MLPEncoder(3, 8, 8)
        self.global_rel_enc = MLPEncoder(1, 8, 8)

        # Gate fusion
        self.brow_fuse = GateFusion(16)
        self.eye_fuse = GateFusion(16)
        self.mouth_fuse = GateFusion(32)
        self.jaw_fuse = GateFusion(16)
        self.global_fuse = GateFusion(8)

        # Projection heads
        self.z_brow = nn.Linear(16, 4)
        self.z_eye = nn.Linear(16, 4)
        self.z_mouth = nn.Linear(32, 10)
        self.z_jaw = nn.Linear(16, 4)
        self.z_global = nn.Linear(8, 2)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h_brow_abs = self.brow_abs_enc(x["brow_abs"])
        h_brow_rel = self.brow_rel_enc(x["brow_rel"])
        h_eye_abs = self.eye_abs_enc(x["eye_abs"])
        h_eye_rel = self.eye_rel_enc(x["eye_rel"])
        h_mouth_abs = self.mouth_abs_enc(x["mouth_abs"])
        h_mouth_rel = self.mouth_rel_enc(x["mouth_rel"])
        h_jaw_abs = self.jaw_abs_enc(x["jaw_abs"])
        h_jaw_rel = self.jaw_rel_enc(x["jaw_rel"])
        h_global_abs = self.global_abs_enc(x["global_abs"])
        h_global_rel = self.global_rel_enc(x["global_rel"])

        h_brow = self.brow_fuse(h_brow_abs, h_brow_rel)
        h_eye = self.eye_fuse(h_eye_abs, h_eye_rel)
        h_mouth = self.mouth_fuse(h_mouth_abs, h_mouth_rel)
        h_jaw = self.jaw_fuse(h_jaw_abs, h_jaw_rel)
        h_global = self.global_fuse(h_global_abs, h_global_rel)

        z_brow = self.z_brow(h_brow)
        z_eye = self.z_eye(h_eye)
        z_mouth = self.z_mouth(h_mouth)
        z_jaw = self.z_jaw(h_jaw)
        z_global = self.z_global(h_global)

        latent = torch.cat([z_brow, z_eye, z_mouth, z_jaw, z_global], dim=1)
        return {
            "z_brow": z_brow,
            "z_eye": z_eye,
            "z_mouth": z_mouth,
            "z_jaw": z_jaw,
            "z_global": z_global,
            "latent24": latent,
        }


# 线性层权重初始化
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# 输出 CSV 列结构
def build_output_columns() -> List[str]:
    cols = ["image_path", "image_name", "face_found", "face_detect_conf", "landmark_conf", "error"]
    cols.extend([f"z_brow_{i:02d}" for i in range(4)])
    cols.extend([f"z_eye_{i:02d}" for i in range(4)])
    cols.extend([f"z_mouth_{i:02d}" for i in range(10)])
    cols.extend([f"z_jaw_{i:02d}" for i in range(4)])
    cols.extend([f"z_global_{i:02d}" for i in range(2)])
    cols.extend([f"latent_{i:02d}" for i in range(24)])
    return cols


# 将批量字典行转换为模型输入张量
def build_batch_inputs(
    batch_abs: List[Dict[str, str]],
    batch_rel: List[Dict[str, str]],
    region_cols: Dict[str, List[str]],
    device: str,
) -> Dict[str, torch.Tensor]:
    arrs: Dict[str, np.ndarray] = {}
    for key, cols in region_cols.items():
        src = batch_abs if key.endswith("_abs") else batch_rel
        mat = np.zeros((len(src), len(cols)), dtype=np.float32)
        for i, row in enumerate(src):
            mat[i] = [safe_float(row.get(c, 0.0)) for c in cols]
        arrs[key] = mat

    out: Dict[str, torch.Tensor] = {}
    for k, v in arrs.items():
        out[k] = torch.from_numpy(v).to(device=device, dtype=torch.float32, non_blocking=True)
    return out


# 单批次前向并写入 z_* 与 latent24 结果
def flush_batch(
    model: RegionLatent24,
    batch_abs: List[Dict[str, str]],
    batch_rel: List[Dict[str, str]],
    region_cols: Dict[str, List[str]],
    writer: csv.DictWriter,
    device: str,
) -> int:
    with torch.inference_mode():
        x = build_batch_inputs(batch_abs, batch_rel, region_cols, device=device)
        y = model(x)

    z_brow = y["z_brow"].detach().cpu().numpy()
    z_eye = y["z_eye"].detach().cpu().numpy()
    z_mouth = y["z_mouth"].detach().cpu().numpy()
    z_jaw = y["z_jaw"].detach().cpu().numpy()
    z_global = y["z_global"].detach().cpu().numpy()
    latent = y["latent24"].detach().cpu().numpy()

    n = len(batch_abs)
    for i in range(n):
        src = batch_abs[i]
        out: Dict[str, object] = {
            "image_path": src.get("image_path", ""),
            "image_name": src.get("image_name", ""),
            "face_found": int(float(src.get("face_found", 0) or 0)),
            "face_detect_conf": safe_float(src.get("face_detect_conf", 0.0)),
            "landmark_conf": safe_float(src.get("landmark_conf", 0.0)),
            "error": src.get("error", ""),
        }
        for j in range(4):
            out[f"z_brow_{j:02d}"] = float(z_brow[i, j])
            out[f"z_eye_{j:02d}"] = float(z_eye[i, j])
            out[f"z_jaw_{j:02d}"] = float(z_jaw[i, j])
        for j in range(10):
            out[f"z_mouth_{j:02d}"] = float(z_mouth[i, j])
        for j in range(2):
            out[f"z_global_{j:02d}"] = float(z_global[i, j])
        for j in range(24):
            out[f"latent_{j:02d}"] = float(latent[i, j])
        writer.writerow(out)
    return n


# 主流程：读取 ABS/REL -> 区域融合网络前向 -> 输出 latent24
def main() -> None:
    args = parse_args()
    if not args.abs_file.exists():
        raise FileNotFoundError(f"ABS file not found: {args.abs_file}")
    if not args.rel_file.exists():
        raise FileNotFoundError(f"REL file not found: {args.rel_file}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = args.device

    # 构建并初始化融合网络（未训练随机权重）
    model = RegionLatent24().to(device)
    model.apply(init_weights)
    model.eval()

    region_cols = build_region_columns()["regions"]
    out_cols = build_output_columns()

    processed = 0
    mismatch = 0
    t0 = time.time()
    batch_abs: List[Dict[str, str]] = []
    batch_rel: List[Dict[str, str]] = []

    print(
        f"[INFO] device={device} cuda_available={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA'}"
    )
    print(f"[INFO] abs_file={args.abs_file}")
    print(f"[INFO] rel_file={args.rel_file}")
    print(f"[INFO] output={args.output}")

    # 同步遍历 ABS/REL，两者按 image_path 对齐
    with open_text(args.abs_file, "rt") as fa, open_text(args.rel_file, "rt") as fr, open_text(args.output, "wt") as fo:
        ra = csv.DictReader(fa)
        rr = csv.DictReader(fr)

        needed_abs = [c for k, cols in region_cols.items() if k.endswith("_abs") for c in cols] + [
            "image_path",
            "image_name",
            "face_found",
            "face_detect_conf",
            "landmark_conf",
            "error",
        ]
        needed_rel = [c for k, cols in region_cols.items() if k.endswith("_rel") for c in cols] + ["image_path", "image_name"]
        miss_abs = [c for c in needed_abs if c not in (ra.fieldnames or [])]
        miss_rel = [c for c in needed_rel if c not in (rr.fieldnames or [])]
        if miss_abs:
            raise RuntimeError(f"ABS missing columns, sample: {miss_abs[:5]}")
        if miss_rel:
            raise RuntimeError(f"REL missing columns, sample: {miss_rel[:5]}")

        writer = csv.DictWriter(fo, fieldnames=out_cols)
        writer.writeheader()

        for abs_row, rel_row in zip(ra, rr):
            if args.max_rows > 0 and (processed + len(batch_abs)) >= args.max_rows:
                break
            if abs_row.get("image_path", "") != rel_row.get("image_path", ""):
                mismatch += 1
            batch_abs.append(abs_row)
            batch_rel.append(rel_row)

            if len(batch_abs) >= args.batch_size:
                processed += flush_batch(model, batch_abs, batch_rel, region_cols, writer, device=device)
                batch_abs = []
                batch_rel = []
                if processed % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"[INFO] processed={processed} speed={processed/max(dt,1e-6):.2f} rows/s mismatches={mismatch}")
        if batch_abs:
            processed += flush_batch(model, batch_abs, batch_rel, region_cols, writer, device=device)

    elapsed = time.time() - t0
    # 输出运行摘要，记录结构和维度
    summary = {
        "abs_file": str(args.abs_file),
        "rel_file": str(args.rel_file),
        "output": str(args.output),
        "rows_processed": processed,
        "row_mismatches_abs_rel_image_path": mismatch,
        "elapsed_seconds": round(elapsed, 3),
        "avg_rows_per_sec": round(processed / max(elapsed, 1e-6), 4),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "architecture": {
            "brow_abs_encoder": "37->32->16",
            "brow_rel_encoder": "37->32->16",
            "eye_abs_encoder": "45->32->16",
            "eye_rel_encoder": "45->32->16",
            "mouth_abs_encoder": "79->64->32",
            "mouth_rel_encoder": "79->64->32",
            "jaw_abs_encoder": "28->32->16",
            "jaw_rel_encoder": "28->32->16",
            "global_abs_encoder": "3->8->8",
            "global_rel_encoder": "1->8->8",
            "gate_fusion_out": {"brow": 16, "eye": 16, "mouth": 32, "jaw": 16, "global": 8},
            "projection": {"z_brow": 4, "z_eye": 4, "z_mouth": 10, "z_jaw": 4, "z_global": 2},
            "latent_total_dim": 24,
        },
        "region_dims": {
            "brow_abs": len(region_cols["brow_abs"]),
            "eye_abs": len(region_cols["eye_abs"]),
            "mouth_abs": len(region_cols["mouth_abs"]),
            "jaw_abs": len(region_cols["jaw_abs"]),
            "global_abs": len(region_cols["global_abs"]),
            "brow_rel": len(region_cols["brow_rel"]),
            "eye_rel": len(region_cols["eye_rel"]),
            "mouth_rel": len(region_cols["mouth_rel"]),
            "jaw_rel": len(region_cols["jaw_rel"]),
            "global_rel": len(region_cols["global_rel"]),
        },
        "note": "Current latent values come from randomly initialized network (seeded). Train or load learned weights for task-specific semantics.",
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.save_model:
        model_path = args.output.with_suffix(args.output.suffix + ".model.pt")
        torch.save({"state_dict": model.state_dict(), "seed": args.seed}, model_path)
        print(f"[DONE] model={model_path}")

    print(f"[DONE] processed={processed} elapsed={elapsed:.1f}s avg={processed/max(elapsed,1e-6):.2f} rows/s")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
