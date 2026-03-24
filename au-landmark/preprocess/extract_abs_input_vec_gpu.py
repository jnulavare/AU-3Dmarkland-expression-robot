#!/usr/bin/env python3
#gpu璁粌锛屾彁鍙朅BS鏁版嵁
"""
GPU ABS feature extraction for X2C.

Backend:
- face-alignment (3D landmarks) on CUDA

Outputs:
- streaming CSV / CSV.GZ / JSONL / JSONL.GZ
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


# AU 杈撳嚭缁村害瀹氫箟锛?5缁达級
AU_NAMES: List[str] = [
    "AU1",
    "AU2",
    "AU4",
    "AU5",
    "AU6",
    "AU7",
    "AU10",
    "AU12",
    "AU14",
    "AU15",
    "AU17",
    "AU20",
    "AU23",
    "AU25",
    "AU26",
]

# 鍑犱綍璺濈杈撳嚭缁村害瀹氫箟锛?4缁达級
DIST_NAMES: List[str] = [
    "brow_left_eye_dist",
    "brow_right_eye_dist",
    "brow_inner_dist",
    "brow_outer_height_diff",
    "eye_left_open",
    "eye_right_open",
    "eye_left_width",
    "eye_right_width",
    "eye_left_ratio",
    "eye_right_ratio",
    "mouth_width",
    "mouth_open",
    "mouth_left_corner_to_nose",
    "mouth_right_corner_to_nose",
    "mouth_left_corner_raise",
    "mouth_right_corner_raise",
    "upper_lip_to_lower_lip",
    "upper_lip_to_nose",
    "lower_lip_to_chin",
    "jaw_open",
    "chin_to_nose",
    "chin_to_upper_lip",
    "mouth_center_to_nose",
    "mouth_center_to_chin",
]

# brows (10) + eyes (12) + mouth (20) + jaw (8) = 50
SELECTED_68_50: List[int] = list(range(17, 27)) + list(range(36, 48)) + list(range(48, 68)) + [
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    16,
]

PNP_68_IDX = [30, 8, 36, 45, 48, 54]
PNP_MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ],
    dtype=np.float64,
)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def to_intensity(value: float, lo: float, hi: float, reverse: bool = False) -> float:
    if hi <= lo:
        return 0.0
    x = (value - lo) / (hi - lo)
    x = clamp(x, 0.0, 1.0)
    if reverse:
        x = 1.0 - x
    return 5.0 * x


def fold_angle_90(angle_deg: float) -> float:
    a = (angle_deg + 180.0) % 360.0 - 180.0
    if a > 90.0:
        a -= 180.0
    elif a < -90.0:
        a += 180.0
    return float(a)


def dist3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def mean_point(pts: np.ndarray, idxs: Sequence[int]) -> np.ndarray:
    return pts[np.array(idxs, dtype=np.int32)].mean(axis=0)


def estimate_pose(landmarks_xy: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, float, float, float]:
    img_pts = np.array([landmarks_xy[i] for i in PNP_68_IDX], dtype=np.float64)
    focal = float(max(width, height))
    camera_matrix = np.array(
        [[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        PNP_MODEL_POINTS,
        img_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return np.eye(3, dtype=np.float64), 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat([rot_mat, tvec])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [float(v) for v in euler.flatten()]
    pitch = fold_angle_90(pitch)
    yaw = fold_angle_90(yaw)
    roll = fold_angle_90(roll)
    return rot_mat, yaw, pitch, roll


def compute_distances(pts: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    p = pts
    left_brow = mean_point(p, [17, 18, 19, 20, 21])
    right_brow = mean_point(p, [22, 23, 24, 25, 26])
    left_eye_c = mean_point(p, [36, 39, 37, 41])
    right_eye_c = mean_point(p, [42, 45, 43, 47])

    left_outer_brow = p[17]
    right_outer_brow = p[26]
    left_inner_brow = p[21]
    right_inner_brow = p[22]

    left_eye_top = 0.5 * (p[37] + p[38])
    left_eye_bot = 0.5 * (p[40] + p[41])
    right_eye_top = 0.5 * (p[43] + p[44])
    right_eye_bot = 0.5 * (p[46] + p[47])
    left_eye_outer, left_eye_inner = p[36], p[39]
    right_eye_inner, right_eye_outer = p[42], p[45]

    mouth_left, mouth_right = p[48], p[54]
    upper_lip, lower_lip = p[62], p[66]
    mouth_center = 0.5 * (upper_lip + lower_lip)
    nose_tip = p[30]
    nose_base = p[33]
    chin = p[8]

    eye_left_open = dist3(left_eye_top, left_eye_bot)
    eye_right_open = dist3(right_eye_top, right_eye_bot)
    eye_left_width = dist3(left_eye_outer, left_eye_inner)
    eye_right_width = dist3(right_eye_inner, right_eye_outer)
    mouth_open = dist3(upper_lip, lower_lip)

    d: Dict[str, float] = {
        "brow_left_eye_dist": dist3(left_brow, left_eye_c),
        "brow_right_eye_dist": dist3(right_brow, right_eye_c),
        "brow_inner_dist": dist3(left_inner_brow, right_inner_brow),
        "brow_outer_height_diff": abs(float(left_outer_brow[1] - right_outer_brow[1])),
        "eye_left_open": eye_left_open,
        "eye_right_open": eye_right_open,
        "eye_left_width": eye_left_width,
        "eye_right_width": eye_right_width,
        "eye_left_ratio": eye_left_open / max(eye_left_width, 1e-6),
        "eye_right_ratio": eye_right_open / max(eye_right_width, 1e-6),
        "mouth_width": dist3(mouth_left, mouth_right),
        "mouth_open": mouth_open,
        "mouth_left_corner_to_nose": dist3(mouth_left, nose_tip),
        "mouth_right_corner_to_nose": dist3(mouth_right, nose_tip),
        "mouth_left_corner_raise": max(0.0, float(nose_tip[1] - mouth_left[1])),
        "mouth_right_corner_raise": max(0.0, float(nose_tip[1] - mouth_right[1])),
        "upper_lip_to_lower_lip": mouth_open,
        "upper_lip_to_nose": dist3(upper_lip, nose_tip),
        "lower_lip_to_chin": dist3(lower_lip, chin),
        "jaw_open": dist3(nose_base, chin),
        "chin_to_nose": dist3(chin, nose_tip),
        "chin_to_upper_lip": dist3(chin, upper_lip),
        "mouth_center_to_nose": dist3(mouth_center, nose_tip),
        "mouth_center_to_chin": dist3(mouth_center, chin),
    }

    aux = {
        "brow_inner_eye_avg": 0.5 * (dist3(left_inner_brow, left_eye_c) + dist3(right_inner_brow, right_eye_c)),
        "brow_outer_eye_avg": 0.5 * (dist3(left_outer_brow, left_eye_c) + dist3(right_outer_brow, right_eye_c)),
        "eye_open_avg": 0.5 * (eye_left_open + eye_right_open),
        "corner_raise_avg": 0.5 * (d["mouth_left_corner_raise"] + d["mouth_right_corner_raise"]),
        "mouth_width": d["mouth_width"],
        "mouth_open": d["mouth_open"],
    }
    return d, aux


def compute_au_from_geometry(d: Dict[str, float], aux: Dict[str, float]) -> Dict[str, float]:
    au: Dict[str, float] = {}
    au["AU1"] = to_intensity(aux["brow_inner_eye_avg"], lo=0.04, hi=0.16)
    au["AU2"] = to_intensity(aux["brow_outer_eye_avg"], lo=0.04, hi=0.20)
    au["AU4"] = to_intensity(aux["brow_inner_eye_avg"], lo=0.04, hi=0.16, reverse=True)
    au["AU5"] = to_intensity(aux["eye_open_avg"], lo=0.01, hi=0.08)
    au["AU6"] = to_intensity(aux["corner_raise_avg"], lo=0.0, hi=0.10)
    au["AU7"] = to_intensity(aux["eye_open_avg"], lo=0.01, hi=0.08, reverse=True)
    au["AU10"] = to_intensity(d["upper_lip_to_nose"], lo=0.05, hi=0.18, reverse=True)
    au["AU12"] = to_intensity(0.7 * aux["mouth_width"] + 0.3 * aux["corner_raise_avg"], lo=0.08, hi=0.45)
    au["AU14"] = to_intensity(aux["mouth_width"], lo=0.10, hi=0.45)
    au["AU15"] = to_intensity(aux["corner_raise_avg"], lo=0.0, hi=0.10, reverse=True)
    au["AU17"] = to_intensity(d["lower_lip_to_chin"], lo=0.05, hi=0.18, reverse=True)
    au["AU20"] = to_intensity(aux["mouth_width"], lo=0.12, hi=0.55)
    au["AU23"] = to_intensity(aux["mouth_width"] / max(aux["mouth_open"], 1e-6), lo=2.0, hi=25.0)
    au["AU25"] = to_intensity(aux["mouth_open"], lo=0.005, hi=0.18)
    au["AU26"] = to_intensity(d["jaw_open"], lo=0.08, hi=0.55)
    for k in au:
        au[k] = float(clamp(au[k], 0.0, 5.0))
    return au


def build_columns() -> List[str]:
    cols = [
        "image_path",
        "image_name",
        "image_width",
        "image_height",
        "face_found",
        "face_bbox",
        "face_bbox_x",
        "face_bbox_y",
        "face_bbox_w",
        "face_bbox_h",
        "face_detect_conf",
        "landmark_conf",
        "error",
        "yaw",
        "pitch",
        "roll",
    ]
    cols.extend([f"{au}_abs_intensity" for au in AU_NAMES])
    for i in range(50):
        cols.extend([f"lmk_abs_norm_{i:02d}_x", f"lmk_abs_norm_{i:02d}_y", f"lmk_abs_norm_{i:02d}_z"])
    cols.extend(DIST_NAMES)
    cols.extend([f"abs_{i:03d}" for i in range(192)])
    return cols


def build_default_row() -> Dict[str, object]:
    row: Dict[str, object] = {
        "image_path": "",
        "image_name": "",
        "image_width": 0,
        "image_height": 0,
        "face_found": 0,
        "face_bbox": json.dumps([-1, -1, -1, -1]),
        "face_bbox_x": -1,
        "face_bbox_y": -1,
        "face_bbox_w": -1,
        "face_bbox_h": -1,
        "face_detect_conf": 0.0,
        "landmark_conf": 0.0,
        "error": "",
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
    }
    for au in AU_NAMES:
        row[f"{au}_abs_intensity"] = 0.0
    for i in range(50):
        row[f"lmk_abs_norm_{i:02d}_x"] = 0.0
        row[f"lmk_abs_norm_{i:02d}_y"] = 0.0
        row[f"lmk_abs_norm_{i:02d}_z"] = 0.0
    for d in DIST_NAMES:
        row[d] = 0.0
    for i in range(192):
        row[f"abs_{i:03d}"] = 0.0
    return row


def parse_args() -> argparse.Namespace:
    # 璇诲彇 ABS 鎶藉彇鍙傛暟
    p = argparse.ArgumentParser(description="GPU ABS feature extraction (face-alignment CUDA)")
    p.add_argument("--dataset-root", type=Path, default=Path(r"E:\DD\Git\X2C"))
    p.add_argument("--image-dirs", nargs="*", default=None)
    p.add_argument("--output", type=Path, default=Path(r"D:\code\AU+landmark\dataset\x2c_data_bundle\ABS_input_vec_X2C_gpu.csv.gz"))
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--torch-home", type=Path, default=Path(r"D:\torch_cache"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--detector", type=str, default="sfd")
    p.add_argument("--redetect-interval", type=int, default=30, help="Re-run full detection every N frames in a folder.")
    return p.parse_args()


def sort_dir_name(name: str) -> Tuple[str, int]:
    digits = "".join(c for c in name if c.isdigit())
    return (name.rstrip(digits), int(digits) if digits else -1)


def discover_image_dirs(dataset_root: Path, explicit_dirs: Sequence[str] | None) -> List[Path]:
    if explicit_dirs:
        dirs = [dataset_root / d for d in explicit_dirs]
    else:
        dirs = [p for p in dataset_root.iterdir() if p.is_dir() and p.name.lower().startswith("image")]
        dirs.sort(key=lambda p: sort_dir_name(p.name))
    return [d for d in dirs if d.exists()]


def collect_images(image_dirs: Sequence[Path]) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths: List[Path] = []
    for folder in image_dirs:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
        files.sort(key=lambda p: p.name)
        paths.extend(files)
    return paths


class Writer:
    def __init__(self, output_path: Path, columns: Sequence[str]):
        self.output_path = output_path
        self.columns = list(columns)
        name = str(output_path).lower()
        if name.endswith(".jsonl.gz"):
            self.mode = "jsonl_gz"
            self.f = gzip.open(output_path, "wt", encoding="utf-8")
        elif name.endswith(".jsonl"):
            self.mode = "jsonl"
            self.f = output_path.open("w", encoding="utf-8")
        elif name.endswith(".csv.gz"):
            self.mode = "csv_gz"
            self.f = gzip.open(output_path, "wt", encoding="utf-8", newline="")
            self.csvw = csv.DictWriter(self.f, fieldnames=self.columns)
            self.csvw.writeheader()
        else:
            self.mode = "csv"
            self.f = output_path.open("w", encoding="utf-8", newline="")
            self.csvw = csv.DictWriter(self.f, fieldnames=self.columns)
            self.csvw.writeheader()

    def write(self, row: Dict[str, object]) -> None:
        if self.mode.startswith("jsonl"):
            self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            self.csvw.writerow(row)

    def close(self) -> None:
        self.f.close()


def extract_one(
    path: Path,
    dataset_root: Path,
    model,
    tracked_bbox: np.ndarray | None = None,
    tracked_conf: float = 0.0,
) -> Tuple[Dict[str, object], np.ndarray | None, float]:
    # 鍗曞紶鍥炬娊鍙栵細face bbox + pose + AU_abs + LMK_abs + DIST_abs + ABS_input_vec
    row = build_default_row()
    row["image_name"] = path.name
    try:
        row["image_path"] = str(path.relative_to(dataset_root)).replace("\\", "/")
    except ValueError:
        row["image_path"] = str(path)

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        row["error"] = "imread_failed"
        return row

    h, w = img_bgr.shape[:2]
    row["image_width"] = int(w)
    row["image_height"] = int(h)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if tracked_bbox is not None:
        out = model.get_landmarks_from_image(
            img_rgb,
            detected_faces=[tracked_bbox],
            return_bboxes=True,
            return_landmark_score=True,
        )
    else:
        out = model.get_landmarks_from_image(img_rgb, return_bboxes=True, return_landmark_score=True)
    if out is None:
        row["error"] = "no_output"
        return row, None, 0.0

    landmarks_all, scores_all, bboxes_all = out
    if landmarks_all is None or len(landmarks_all) == 0:
        row["error"] = "no_face"
        return row, None, 0.0

    # choose highest detector score
    best_i = 0
    best_score = -1.0
    for i, bb in enumerate(bboxes_all):
        s = float(bb[4]) if len(bb) > 4 else float(tracked_conf)
        if s > best_score:
            best_score = s
            best_i = i

    lmk = np.asarray(landmarks_all[best_i], dtype=np.float64)  # (68,3)
    lmk_score = np.asarray(scores_all[best_i], dtype=np.float64) if scores_all is not None else None
    bb = np.asarray(bboxes_all[best_i], dtype=np.float64)

    x1, y1, x2, y2 = [int(v) for v in bb[:4]]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    row["face_bbox_x"] = x1
    row["face_bbox_y"] = y1
    row["face_bbox_w"] = bw
    row["face_bbox_h"] = bh
    row["face_bbox"] = json.dumps([x1, y1, bw, bh])
    row["face_detect_conf"] = float(best_score if best_score >= 0 else 0.0)
    if lmk_score is not None and lmk_score.size > 0:
        row["landmark_conf"] = float(np.clip(np.mean(lmk_score), 0.0, 1.0))
    row["face_found"] = 1

    rot_mat, yaw, pitch, roll = estimate_pose(lmk[:, :2], w, h)
    row["yaw"] = yaw
    row["pitch"] = pitch
    row["roll"] = roll

    pts = lmk - lmk.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(pts * pts, axis=1))))
    if scale < 1e-8:
        scale = 1.0
    pts = pts / scale
    pts = pts @ rot_mat

    pts50 = pts[np.array(SELECTED_68_50, dtype=np.int32)]
    flat_lmk = pts50.reshape(-1)
    for i in range(50):
        row[f"lmk_abs_norm_{i:02d}_x"] = float(pts50[i, 0])
        row[f"lmk_abs_norm_{i:02d}_y"] = float(pts50[i, 1])
        row[f"lmk_abs_norm_{i:02d}_z"] = float(pts50[i, 2])

    dists, aux = compute_distances(pts)
    for k in DIST_NAMES:
        row[k] = float(dists[k])
    aus = compute_au_from_geometry(dists, aux)
    for au in AU_NAMES:
        row[f"{au}_abs_intensity"] = float(aus[au])

    pose_vec = np.array([yaw, pitch, roll], dtype=np.float64)
    au_vec = np.array([aus[a] for a in AU_NAMES], dtype=np.float64)
    dist_vec = np.array([dists[d] for d in DIST_NAMES], dtype=np.float64)
    abs_vec = np.concatenate([pose_vec, au_vec, flat_lmk, dist_vec], axis=0)
    if abs_vec.shape[0] != 192:
        raise RuntimeError(f"ABS vector size mismatch: {abs_vec.shape[0]}")
    for i, v in enumerate(abs_vec.tolist()):
        row[f"abs_{i:03d}"] = float(v)

    next_bbox = bb[:4].copy()
    next_conf = float(best_score if best_score >= 0 else tracked_conf)
    return row, next_bbox, next_conf


def main() -> None:
    # 涓绘祦绋嬶細閬嶅巻鍥剧墖骞舵祦寮忓啓鍑?ABS 鏂囦欢
    args = parse_args()
    os.environ["TORCH_HOME"] = str(args.torch_home)
    args.torch_home.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch  # local import so TORCH_HOME is applied
    import face_alignment as fa

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but device=cuda requested.")

    model = fa.FaceAlignment(
        fa.LandmarksType.THREE_D,
        device=args.device,
        face_detector=args.detector,
        flip_input=False,
        verbose=False,
    )

    image_dirs = discover_image_dirs(args.dataset_root, args.image_dirs)
    if not image_dirs:
        raise RuntimeError(f"No image dirs found under {args.dataset_root}")
    images = collect_images(image_dirs)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError("No images found.")

    print(f"[INFO] device={args.device} cuda_available={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA'}")
    print(f"[INFO] torch_home={args.torch_home}")
    print(f"[INFO] dataset_root={args.dataset_root} dirs={len(image_dirs)} images={len(images)}")
    print(f"[INFO] output={args.output}")

    writer = Writer(args.output, build_columns())
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    tracked_bbox: np.ndarray | None = None
    tracked_conf = 0.0
    tracked_steps = 0
    current_dir = None

    try:
        for i, path in enumerate(images, start=1):
            if current_dir != path.parent:
                current_dir = path.parent
                tracked_bbox = None
                tracked_conf = 0.0
                tracked_steps = 0

            use_track = tracked_bbox is not None and tracked_steps < args.redetect_interval
            try:
                row, next_bbox, next_conf = extract_one(
                    path,
                    args.dataset_root,
                    model,
                    tracked_bbox=tracked_bbox if use_track else None,
                    tracked_conf=tracked_conf,
                )
                if int(row["face_found"]) == 1:
                    ok_count += 1
                    tracked_bbox = next_bbox
                    tracked_conf = next_conf
                    tracked_steps = tracked_steps + 1 if use_track else 1
                else:
                    fail_count += 1
                    tracked_bbox = None
                    tracked_conf = 0.0
                    tracked_steps = 0
            except Exception as exc:
                row = build_default_row()
                row["image_path"] = str(path)
                row["image_name"] = path.name
                row["error"] = str(exc)
                fail_count += 1
                tracked_bbox = None
                tracked_conf = 0.0
                tracked_steps = 0
            writer.write(row)

            if i % args.log_every == 0:
                dt = time.time() - t0
                print(f"[INFO] processed={i}/{len(images)} ok={ok_count} fail={fail_count} speed={i/max(dt,1e-6):.2f} img/s")
    finally:
        writer.close()

    elapsed = time.time() - t0
    summary = {
        "dataset_root": str(args.dataset_root),
        "output": str(args.output),
        "images_processed": len(images),
        "ok_count": ok_count,
        "fail_count": fail_count,
        "elapsed_seconds": round(elapsed, 3),
        "avg_img_per_sec": round(len(images) / max(elapsed, 1e-6), 4),
        "device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "abs_dim": 192,
        "au_names": AU_NAMES,
        "dist_names": DIST_NAMES,
        "selected_landmark_indices": SELECTED_68_50,
        "notes": [
            "Landmarks are face-alignment 68-point 3D output.",
            "AU_abs_intensity is geometry-derived proxy (0-5), not OpenFace AU regression.",
        ],
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] processed={len(images)} ok={ok_count} fail={fail_count} elapsed={elapsed:.1f}s avg={len(images)/max(elapsed,1e-6):.2f} img/s")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()

