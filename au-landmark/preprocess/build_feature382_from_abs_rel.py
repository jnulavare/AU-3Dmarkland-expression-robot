#!/usr/bin/env python3
from __future__ import annotations

"""Build compare1 feature dataset: ABS+REL -> 382-dim feature vectors."""

import argparse
import csv
import gzip
import json
import shutil
import time
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build compare1 feature382 (ABS+REL) for direct 382->30 regression.")
    p.add_argument(
        "--abs-file",
        type=Path,
        default=Path(r"D:\code\AU+landmark\dataset\x2c_data_bundle\ABS_input_vec_X2C_gpu.csv.gz"),
    )
    p.add_argument(
        "--rel-file",
        type=Path,
        default=Path(r"D:\code\AU+landmark\dataset\x2c_data_bundle\REL_input_vec_X2C_gpu.csv.gz"),
    )
    p.add_argument(
        "--target-file",
        type=Path,
        default=Path(r"D:\code\AU+landmark\dataset\x2c_data_bundle\metadata_normalize.jsonl"),
    )
    p.add_argument("--output-dir", type=Path, default=Path(r"D:\code\AU+landmark\dataset\x2c_data_compare1"))
    p.add_argument("--feature-file-name", type=str, default="FEATURE382_X2C_gpu.csv.gz")
    p.add_argument("--target-file-name", type=str, default="metadata_normalize.jsonl")
    p.add_argument("--log-every", type=int, default=20000)
    p.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    p.add_argument("--no-copy-target", action="store_true", help="Do not copy metadata_normalize.jsonl to output dir.")
    return p.parse_args()


def open_text(path: Path, mode: str):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def safe_float(v: object) -> float:
    if v is None:
        return 0.0
    s = str(v).strip()
    if s == "":
        return 0.0
    return float(s)


def lmk_cols_abs(start_idx: int, end_idx: int) -> List[str]:
    out: List[str] = []
    for i in range(start_idx, end_idx + 1):
        out.extend([f"lmk_abs_norm_{i:02d}_x", f"lmk_abs_norm_{i:02d}_y", f"lmk_abs_norm_{i:02d}_z"])
    return out


def lmk_cols_rel(start_idx: int, end_idx: int) -> List[str]:
    out: List[str] = []
    for i in range(start_idx, end_idx + 1):
        out.extend([f"lmk_rel_{i:02d}_x", f"lmk_rel_{i:02d}_y", f"lmk_rel_{i:02d}_z"])
    return out


def au_abs_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_abs_intensity" for n in names]


def au_rel_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_rel" for n in names]


def dist_rel_cols(names: Sequence[str]) -> List[str]:
    return [f"{n}_rel" for n in names]


def build_feature_layout() -> List[Tuple[str, str, List[str]]]:
    # Final order is fixed and shared by training + explainability configs.
    return [
        ("brow", "abs", au_abs_cols(BROW_AU) + lmk_cols_abs(0, 9) + BROW_DIST),
        ("brow", "rel", au_rel_cols(BROW_AU) + lmk_cols_rel(0, 9) + dist_rel_cols(BROW_DIST)),
        ("eye", "abs", au_abs_cols(EYE_AU) + lmk_cols_abs(10, 21) + EYE_DIST),
        ("eye", "rel", au_rel_cols(EYE_AU) + lmk_cols_rel(10, 21) + dist_rel_cols(EYE_DIST)),
        ("mouth", "abs", au_abs_cols(MOUTH_AU) + lmk_cols_abs(22, 41) + MOUTH_DIST),
        ("mouth", "rel", au_rel_cols(MOUTH_AU) + lmk_cols_rel(22, 41) + dist_rel_cols(MOUTH_DIST)),
        ("jaw", "abs", au_abs_cols(JAW_AU) + lmk_cols_abs(42, 49) + JAW_DIST),
        ("jaw", "rel", au_rel_cols(JAW_AU) + lmk_cols_rel(42, 49) + dist_rel_cols(JAW_DIST)),
        ("global", "abs", ["yaw", "pitch", "roll"]),
        ("global", "rel", ["ENERGY_rel"]),
    ]


def ensure_required_columns(fieldnames: List[str], required_cols: Sequence[str], source_name: str) -> None:
    missing = [c for c in required_cols if c not in fieldnames]
    if missing:
        raise RuntimeError(f"{source_name} missing columns, first missing: {missing[:10]}")


def main() -> None:
    # 1) Validate input files and prepare output paths.
    args = parse_args()
    if not args.abs_file.exists():
        raise FileNotFoundError(f"abs file not found: {args.abs_file}")
    if not args.rel_file.exists():
        raise FileNotFoundError(f"rel file not found: {args.rel_file}")
    if not args.target_file.exists():
        raise FileNotFoundError(f"target file not found: {args.target_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_feature = args.output_dir / args.feature_file_name
    output_target = args.output_dir / args.target_file_name
    summary_path = output_feature.with_suffix(output_feature.suffix + ".summary.json")
    columns_map_path = output_feature.with_suffix(output_feature.suffix + ".columns.json")

    # 2) Build feature schema and feature index mapping.
    layout = build_feature_layout()
    feature_dim = sum(len(cols) for _, _, cols in layout)
    feature_cols = [f"feat_{i:03d}" for i in range(feature_dim)]

    meta_cols = [
        "image_path",
        "image_name",
        "face_found",
        "face_detect_conf",
        "landmark_conf",
        "error",
    ]

    region_ranges: Dict[str, Dict[str, int]] = {}
    columns_map: List[Dict[str, object]] = []
    feat_idx = 0
    for region, source, cols in layout:
        start = feat_idx
        for c in cols:
            columns_map.append(
                {
                    "feat_idx": feat_idx,
                    "feat_name": f"feat_{feat_idx:03d}",
                    "region": region,
                    "source": source,
                    "source_column": c,
                }
            )
            feat_idx += 1
        end = feat_idx - 1
        if region not in region_ranges:
            region_ranges[region] = {"start": start, "end": end, "dim": end - start + 1}
        else:
            region_ranges[region]["end"] = end
            region_ranges[region]["dim"] = region_ranges[region]["end"] - region_ranges[region]["start"] + 1

    t0 = time.time()
    processed = 0
    mismatch_path = 0
    mismatch_name = 0

    # 3) Stream ABS/REL rows, merge per row, and write feature file.
    with open_text(args.abs_file, "rt") as fa, open_text(args.rel_file, "rt") as fr, open_text(output_feature, "wt") as fo:
        ra = csv.DictReader(fa)
        rr = csv.DictReader(fr)
        abs_fields = list(ra.fieldnames or [])
        rel_fields = list(rr.fieldnames or [])

        abs_req: List[str] = []
        rel_req: List[str] = []
        for _, source, cols in layout:
            if source == "abs":
                abs_req.extend(cols)
            else:
                rel_req.extend(cols)
        abs_req.extend(meta_cols)
        rel_req.extend(["image_path", "image_name"])

        ensure_required_columns(abs_fields, abs_req, "ABS")
        ensure_required_columns(rel_fields, rel_req, "REL")

        writer = csv.DictWriter(fo, fieldnames=meta_cols + feature_cols)
        writer.writeheader()

        for abs_row, rel_row in zip_longest(ra, rr, fillvalue=None):
            if abs_row is None or rel_row is None:
                raise RuntimeError("ABS and REL row count mismatch")
            if args.max_rows > 0 and processed >= args.max_rows:
                break

            if abs_row.get("image_path", "") != rel_row.get("image_path", ""):
                mismatch_path += 1
            if abs_row.get("image_name", "") != rel_row.get("image_name", ""):
                mismatch_name += 1

            row_out: Dict[str, object] = {
                "image_path": abs_row.get("image_path", ""),
                "image_name": abs_row.get("image_name", ""),
                "face_found": int(float(abs_row.get("face_found", 0) or 0)),
                "face_detect_conf": safe_float(abs_row.get("face_detect_conf", 0.0)),
                "landmark_conf": safe_float(abs_row.get("landmark_conf", 0.0)),
                "error": abs_row.get("error", ""),
            }

            values: List[float] = []
            for _, source, cols in layout:
                src_row = abs_row if source == "abs" else rel_row
                values.extend([safe_float(src_row.get(c, 0.0)) for c in cols])

            if len(values) != feature_dim:
                raise RuntimeError(f"feature dim mismatch while building row: {len(values)} != {feature_dim}")

            for i, v in enumerate(values):
                row_out[f"feat_{i:03d}"] = float(v)
            writer.writerow(row_out)

            processed += 1
            if processed % max(1, args.log_every) == 0:
                dt = time.time() - t0
                print(
                    f"[INFO] processed={processed} speed={processed/max(dt,1e-6):.2f} rows/s "
                    f"path_mismatch={mismatch_path} name_mismatch={mismatch_name}"
                )

    # 4) Copy target jsonl to compare1 directory for self-contained training inputs.
    if not args.no_copy_target:
        shutil.copy2(args.target_file, output_target)

    elapsed = time.time() - t0
    # 5) Persist build summary + full feature column mapping.
    summary = {
        "abs_file": str(args.abs_file),
        "rel_file": str(args.rel_file),
        "target_file": str(args.target_file),
        "output_feature_file": str(output_feature),
        "output_target_file": str(output_target) if not args.no_copy_target else None,
        "rows_processed": processed,
        "feature_dim": feature_dim,
        "feature_layout": [{"region": r, "source": s, "dim": len(c)} for r, s, c in layout],
        "region_feature_ranges": region_ranges,
        "row_mismatch": {
            "image_path_mismatch": mismatch_path,
            "image_name_mismatch": mismatch_name,
        },
        "elapsed_seconds": round(elapsed, 3),
        "avg_rows_per_sec": round(processed / max(elapsed, 1e-6), 4),
        "notes": [
            "This dataset is for direct 382->30 regression.",
            "feature382 order = brow(abs+rel), eye(abs+rel), mouth(abs+rel), jaw(abs+rel), global(abs+rel).",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    columns_map_path.write_text(json.dumps(columns_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] feature_file={output_feature}")
    print(f"[DONE] rows={processed} dim={feature_dim} elapsed={elapsed:.1f}s avg={processed/max(elapsed,1e-6):.2f} rows/s")
    if not args.no_copy_target:
        print(f"[DONE] copied_target={output_target}")
    print(f"[DONE] summary={summary_path}")
    print(f"[DONE] columns_map={columns_map_path}")


if __name__ == "__main__":
    main()
