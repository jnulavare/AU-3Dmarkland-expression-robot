#!/usr/bin/env python3
#GPU鎻愬彇REL鏁版嵁
"""
Build relative expression features from existing ABS outputs using GPU tensor ops.

Definitions:
- AU_rel    = AU_abs(img) - AU_abs(neutral)
- LMK_rel   = LMK_abs_norm(img) - LMK_abs_norm(neutral)
- DIST_rel  = DIST_abs(img) - DIST_abs(neutral)
- ENERGY_rel = sum_k |DIST_rel_k|

REL_input_vec = concat(AU_rel(15), LMK_rel(150), DIST_rel(24), ENERGY_rel(1)) -> 190 dims
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    # 璇诲彇 REL 璁＄畻鍙傛暟
    p = argparse.ArgumentParser(description="Build REL vectors from ABS file and neutral image.")
    p.add_argument("--abs-file", type=Path, default=Path(r"D:\code\AU+landmark\dataset\x2c_data_bundle\ABS_input_vec_X2C_gpu.csv.gz"))
    p.add_argument("--neutral-image", type=Path, default=Path(r"D:\code\X2CNet-main\assets\ameca_neutral.jpg"))
    p.add_argument("--output", type=Path, default=Path(r"D:\code\AU+landmark\dataset\x2c_data_bundle\REL_input_vec_X2C_gpu.csv.gz"))
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--log-every", type=int, default=20000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch-home", type=Path, default=Path(r"D:\torch_cache"))
    return p.parse_args()


def lmk_abs_cols() -> List[str]:
    cols: List[str] = []
    for i in range(50):
        cols.extend(
            [
                f"lmk_abs_norm_{i:02d}_x",
                f"lmk_abs_norm_{i:02d}_y",
                f"lmk_abs_norm_{i:02d}_z",
            ]
        )
    return cols


def rel_lmk_cols() -> List[str]:
    cols: List[str] = []
    for i in range(50):
        cols.extend(
            [
                f"lmk_rel_{i:02d}_x",
                f"lmk_rel_{i:02d}_y",
                f"lmk_rel_{i:02d}_z",
            ]
        )
    return cols


def get_au_abs_cols(au_names: Sequence[str]) -> List[str]:
    return [f"{name}_abs_intensity" for name in au_names]


def get_au_rel_cols(au_names: Sequence[str]) -> List[str]:
    return [f"{name}_rel" for name in au_names]


def get_dist_rel_cols(dist_names: Sequence[str]) -> List[str]:
    return [f"{name}_rel" for name in dist_names]


def open_text(path: Path, mode: str):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def safe_float(v: object) -> float:
    if v is None:
        return 0.0
    if isinstance(v, float):
        return v
    s = str(v).strip()
    if s == "":
        return 0.0
    return float(s)


def build_output_columns(au_names: Sequence[str], dist_names: Sequence[str]) -> List[str]:
    meta_cols = [
        "image_path",
        "image_name",
        "image_width",
        "image_height",
        "face_found",
        "face_detect_conf",
        "landmark_conf",
        "error",
    ]
    cols = list(meta_cols)
    cols.extend(get_au_rel_cols(au_names))
    cols.extend(rel_lmk_cols())
    cols.extend(get_dist_rel_cols(dist_names))
    cols.append("ENERGY_rel")
    cols.extend([f"rel_{i:03d}" for i in range(190)])
    return cols


def extract_neutral_abs_vectors(neutral_image: Path, device: str, torch_home: Path):
    # 鍏堝 neutral 鍥惧儚鎻愬彇 ABS 鍩哄噯鍚戦噺
    os.environ["TORCH_HOME"] = str(torch_home)
    torch_home.mkdir(parents=True, exist_ok=True)

    import face_alignment as fa
    import torch

    import extract_abs_input_vec_gpu as abs_gpu

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model = fa.FaceAlignment(
        fa.LandmarksType.THREE_D,
        device=device,
        face_detector="sfd",
        flip_input=False,
        verbose=False,
    )
    row, _, _ = abs_gpu.extract_one(
        path=neutral_image,
        dataset_root=neutral_image.parent,
        model=model,
        tracked_bbox=None,
        tracked_conf=0.0,
    )
    if int(row.get("face_found", 0)) != 1:
        raise RuntimeError(f"Neutral face extraction failed: {neutral_image}  error={row.get('error','')}")

    au_names = abs_gpu.AU_NAMES
    dist_names = abs_gpu.DIST_NAMES
    au_vec = np.array([safe_float(row[f"{a}_abs_intensity"]) for a in au_names], dtype=np.float32)
    lmk_vec = np.array([safe_float(row[c]) for c in lmk_abs_cols()], dtype=np.float32)
    dist_vec = np.array([safe_float(row[d]) for d in dist_names], dtype=np.float32)

    info = {
        "face_detect_conf": safe_float(row["face_detect_conf"]),
        "landmark_conf": safe_float(row["landmark_conf"]),
        "yaw": safe_float(row["yaw"]),
        "pitch": safe_float(row["pitch"]),
        "roll": safe_float(row["roll"]),
    }
    return au_names, dist_names, au_vec, lmk_vec, dist_vec, info


def process_batch(
    rows: List[Dict[str, str]],
    au_abs_cols: Sequence[str],
    lmk_cols: Sequence[str],
    dist_names: Sequence[str],
    au_neutral_t,
    lmk_neutral_t,
    dist_neutral_t,
    device: str,
) -> List[Dict[str, object]]:
    # 鎵归噺璁＄畻 AU/LMK/DIST 鐨勭浉瀵瑰€硷紝骞舵嫾鎺?190 缁?REL_input_vec
    import torch

    b = len(rows)
    au_np = np.zeros((b, len(au_abs_cols)), dtype=np.float32)
    lmk_np = np.zeros((b, len(lmk_cols)), dtype=np.float32)
    dist_np = np.zeros((b, len(dist_names)), dtype=np.float32)

    for i, row in enumerate(rows):
        au_np[i] = [safe_float(row[c]) for c in au_abs_cols]
        lmk_np[i] = [safe_float(row[c]) for c in lmk_cols]
        dist_np[i] = [safe_float(row[c]) for c in dist_names]

    au_t = torch.from_numpy(au_np).to(device=device, dtype=torch.float32, non_blocking=True)
    lmk_t = torch.from_numpy(lmk_np).to(device=device, dtype=torch.float32, non_blocking=True)
    dist_t = torch.from_numpy(dist_np).to(device=device, dtype=torch.float32, non_blocking=True)

    au_rel_t = au_t - au_neutral_t
    lmk_rel_t = lmk_t - lmk_neutral_t
    dist_rel_t = dist_t - dist_neutral_t
    energy_t = torch.sum(torch.abs(dist_rel_t), dim=1, keepdim=True)
    rel_vec_t = torch.cat([au_rel_t, lmk_rel_t, dist_rel_t, energy_t], dim=1)

    au_rel = au_rel_t.detach().cpu().numpy()
    lmk_rel = lmk_rel_t.detach().cpu().numpy()
    dist_rel = dist_rel_t.detach().cpu().numpy()
    energy = energy_t.detach().cpu().numpy()
    rel_vec = rel_vec_t.detach().cpu().numpy()

    out: List[Dict[str, object]] = []
    au_rel_cols = get_au_rel_cols([c.replace("_abs_intensity", "") for c in au_abs_cols])
    lmk_rel_names = rel_lmk_cols()
    dist_rel_cols = get_dist_rel_cols(dist_names)

    for i, src in enumerate(rows):
        row: Dict[str, object] = {
            "image_path": src.get("image_path", ""),
            "image_name": src.get("image_name", ""),
            "image_width": int(float(src.get("image_width", 0) or 0)),
            "image_height": int(float(src.get("image_height", 0) or 0)),
            "face_found": int(float(src.get("face_found", 0) or 0)),
            "face_detect_conf": safe_float(src.get("face_detect_conf", 0.0)),
            "landmark_conf": safe_float(src.get("landmark_conf", 0.0)),
            "error": src.get("error", ""),
        }
        for j, c in enumerate(au_rel_cols):
            row[c] = float(au_rel[i, j])
        for j, c in enumerate(lmk_rel_names):
            row[c] = float(lmk_rel[i, j])
        for j, c in enumerate(dist_rel_cols):
            row[c] = float(dist_rel[i, j])
        row["ENERGY_rel"] = float(energy[i, 0])
        for j in range(190):
            row[f"rel_{j:03d}"] = float(rel_vec[i, j])
        out.append(row)
    return out


def main() -> None:
    # 涓绘祦绋嬶細璇诲彇 ABS锛屾寜 neutral 璁＄畻 REL锛屾祦寮忓啓鍑虹粨鏋?
    args = parse_args()
    if not args.abs_file.exists():
        raise FileNotFoundError(f"ABS file not found: {args.abs_file}")
    if not args.neutral_image.exists():
        raise FileNotFoundError(f"Neutral image not found: {args.neutral_image}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_HOME"] = str(args.torch_home)
    args.torch_home.mkdir(parents=True, exist_ok=True)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = args.device

    t0 = time.time()
    au_names, dist_names, au_neutral, lmk_neutral, dist_neutral, neutral_info = extract_neutral_abs_vectors(
        args.neutral_image, device=device, torch_home=args.torch_home
    )
    au_abs_cols = get_au_abs_cols(au_names)
    lmk_cols = lmk_abs_cols()

    au_neutral_t = torch.from_numpy(au_neutral).to(device=device, dtype=torch.float32)
    lmk_neutral_t = torch.from_numpy(lmk_neutral).to(device=device, dtype=torch.float32)
    dist_neutral_t = torch.from_numpy(dist_neutral).to(device=device, dtype=torch.float32)

    out_cols = build_output_columns(au_names, dist_names)
    out_count = 0
    face_ok = 0
    batch_rows: List[Dict[str, str]] = []

    print(
        f"[INFO] device={device} cuda_available={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA'}"
    )
    print(f"[INFO] abs_file={args.abs_file}")
    print(f"[INFO] neutral_image={args.neutral_image}")
    print(f"[INFO] output={args.output}")

    with open_text(args.abs_file, "rt") as f_in, open_text(args.output, "wt") as f_out:
        reader = csv.DictReader(f_in)
        missing = [c for c in (au_abs_cols + lmk_cols + list(dist_names)) if c not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"ABS file missing required columns, first missing: {missing[:5]}")

        writer = csv.DictWriter(f_out, fieldnames=out_cols)
        writer.writeheader()

        for src_row in reader:
            batch_rows.append(src_row)
            if len(batch_rows) >= args.batch_size:
                out_rows = process_batch(
                    rows=batch_rows,
                    au_abs_cols=au_abs_cols,
                    lmk_cols=lmk_cols,
                    dist_names=dist_names,
                    au_neutral_t=au_neutral_t,
                    lmk_neutral_t=lmk_neutral_t,
                    dist_neutral_t=dist_neutral_t,
                    device=device,
                )
                for r in out_rows:
                    writer.writerow(r)
                    out_count += 1
                    face_ok += int(r["face_found"])
                batch_rows = []
                if out_count % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"[INFO] processed={out_count} speed={out_count/max(dt,1e-6):.2f} rows/s")

        if batch_rows:
            out_rows = process_batch(
                rows=batch_rows,
                au_abs_cols=au_abs_cols,
                lmk_cols=lmk_cols,
                dist_names=dist_names,
                au_neutral_t=au_neutral_t,
                lmk_neutral_t=lmk_neutral_t,
                dist_neutral_t=dist_neutral_t,
                device=device,
            )
            for r in out_rows:
                writer.writerow(r)
                out_count += 1
                face_ok += int(r["face_found"])

    elapsed = time.time() - t0
    summary = {
        "abs_file": str(args.abs_file),
        "neutral_image": str(args.neutral_image),
        "output": str(args.output),
        "rows_processed": out_count,
        "face_found_count": face_ok,
        "elapsed_seconds": round(elapsed, 3),
        "avg_rows_per_sec": round(out_count / max(elapsed, 1e-6), 4),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dimensions": {
            "AU_rel": 15,
            "LMK_rel": 150,
            "DIST_rel": 24,
            "ENERGY_rel": 1,
            "REL_input_vec": 190,
        },
        "neutral_abs_info": neutral_info,
        "notes": [
            "AU_rel = AU_abs(img) - AU_abs(neutral)",
            "LMK_rel = LMK_abs_norm(img) - LMK_abs_norm(neutral)",
            "DIST_rel = DIST_abs(img) - DIST_abs(neutral)",
            "ENERGY_rel = sum(abs(DIST_rel))",
        ],
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    neutral_path = args.output.with_suffix(args.output.suffix + ".neutral_abs.json")
    neutral_dump = {
        "neutral_image": str(args.neutral_image),
        "au_names": au_names,
        "dist_names": list(dist_names),
        "AU_abs_neutral": au_neutral.tolist(),
        "LMK_abs_norm_neutral": lmk_neutral.tolist(),
        "DIST_abs_neutral": dist_neutral.tolist(),
        "neutral_abs_info": neutral_info,
    }
    with neutral_path.open("w", encoding="utf-8") as f:
        json.dump(neutral_dump, f, ensure_ascii=False, indent=2)

    print(f"[DONE] processed={out_count} elapsed={elapsed:.1f}s avg={out_count/max(elapsed,1e-6):.2f} rows/s")
    print(f"[DONE] summary={summary_path}")
    print(f"[DONE] neutral_abs={neutral_path}")


if __name__ == "__main__":
    main()

