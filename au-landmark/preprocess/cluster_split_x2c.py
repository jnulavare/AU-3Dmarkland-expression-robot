#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


SPLIT_NAMES = ["train", "val", "test"]
TARGET_RATIOS = np.array([0.70, 0.15, 0.15], dtype=np.float64)


# 读取划分脚本参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster-based train/val/test split for X2C.")
    p.add_argument("--dataset-root", type=Path, default=Path(r"E:\DD\Git\X2C"))
    p.add_argument("--latent-file", type=Path, default=Path(r"D:\code\test2\x2c_data_bundle\LATENT24_X2C_gpu.csv.gz"))
    p.add_argument("--metadata-file", type=Path, default=Path(r"D:\code\test2\x2c_data_bundle\metadata.jsonl"))
    p.add_argument("--n-clusters", type=int, default=320, help="Recommended 200-500.")
    p.add_argument("--cluster-device", type=str, default="cuda")
    p.add_argument("--cluster-iters", type=int, default=40)
    p.add_argument("--cluster-batch", type=int, default=20000)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--assign-restarts", type=int, default=64)
    p.add_argument("--output-train", type=Path, default=None)
    p.add_argument("--output-val", type=Path, default=None)
    p.add_argument("--output-test", type=Path, default=None)
    p.add_argument("--report-json", type=Path, default=None)
    return p.parse_args()


# 读取 metadata 中的 30 维电机值映射：idx -> ctrl30
def load_metadata_ctrl_map(metadata_file: Path) -> Dict[int, np.ndarray]:
    ctrl_map: Dict[int, np.ndarray] = {}
    with metadata_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            name = obj["file_name"]
            stem = Path(name).stem
            idx = int(stem)
            ctrl = np.asarray(obj["ctrl_value"], dtype=np.float32)
            if ctrl.shape[0] != 30:
                raise RuntimeError(f"ctrl_value dim != 30 at {name}")
            ctrl_map[idx] = ctrl
    return ctrl_map


# 读取 latent24 并与 ctrl30 对齐，得到聚类特征输入
def load_latent_and_ctrl(
    latent_file: Path,
    ctrl_map: Dict[int, np.ndarray],
    dataset_root: Path,
) -> Tuple[List[str], List[np.ndarray], np.ndarray, np.ndarray]:
    image_paths: List[str] = []
    ctrl_values: List[np.ndarray] = []
    latent_rows: List[np.ndarray] = []

    with gzip.open(latent_file, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        latent_cols = [f"latent_{i:02d}" for i in range(24)]
        missing = [c for c in ["image_path", "image_name"] + latent_cols if c not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"latent file missing columns: {missing[:5]}")

        for row in reader:
            image_name = row["image_name"]
            idx = int(Path(image_name).stem)
            if idx not in ctrl_map:
                raise RuntimeError(f"metadata missing ctrl for {image_name}")

            rel_path = row["image_path"].replace("/", "\\")
            abs_path = str(dataset_root / rel_path)
            image_paths.append(abs_path)
            ctrl_values.append(ctrl_map[idx])
            latent_rows.append(np.asarray([float(row[c]) for c in latent_cols], dtype=np.float32))

    latent_arr = np.vstack(latent_rows).astype(np.float32)
    ctrl_arr = np.vstack(ctrl_values).astype(np.float32)
    return image_paths, ctrl_values, latent_arr, ctrl_arr


# 按 cluster 聚合统计（数量、和、平方和）
def build_cluster_stats(labels: np.ndarray, z: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = z.shape
    counts = np.bincount(labels, minlength=n_clusters).astype(np.int64)
    sums = np.zeros((n_clusters, d), dtype=np.float64)
    sqs = np.zeros((n_clusters, d), dtype=np.float64)
    np.add.at(sums, labels, z)
    np.add.at(sqs, labels, z * z)
    return counts, sums, sqs


# 对特征做标准化（替代 sklearn 标准化）
def standardize_np(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    z = ((x - mean) / std).astype(np.float32)
    return z, mean.astype(np.float32), std.astype(np.float32)


# 纯 PyTorch KMeans（支持 GPU）
def torch_kmeans(
    z: np.ndarray,
    n_clusters: int,
    device: str,
    max_iters: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    x = torch.from_numpy(z).to(device=device, dtype=torch.float32)
    n, d = x.shape
    if n_clusters > n:
        raise RuntimeError(f"n_clusters({n_clusters}) > n_samples({n})")

    perm = torch.randperm(n, device=device)
    centers = x[perm[:n_clusters]].clone()

    labels = torch.zeros(n, dtype=torch.long, device=device)
    prev_inertia = None

    for it in range(max_iters):
        # assignment (chunked)
        inertia = 0.0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = x[start:end]
            dists = torch.cdist(xb, centers, p=2) ** 2
            l = torch.argmin(dists, dim=1)
            labels[start:end] = l
            inertia += float(dists[torch.arange(end - start, device=device), l].sum().item())

        # update
        sums = torch.zeros((n_clusters, d), device=device, dtype=torch.float32)
        counts = torch.bincount(labels, minlength=n_clusters).to(torch.float32)
        sums.index_add_(0, labels, x)

        empty = counts == 0
        counts_safe = counts.clone()
        counts_safe[empty] = 1.0
        new_centers = sums / counts_safe.unsqueeze(1)

        if bool(empty.any()):
            re_idx = torch.randperm(n, device=device)[: int(empty.sum().item())]
            new_centers[empty] = x[re_idx]

        shift = float(torch.mean(torch.norm(new_centers - centers, dim=1)).item())
        centers = new_centers

        if prev_inertia is not None:
            rel_drop = abs(prev_inertia - inertia) / max(prev_inertia, 1e-8)
            if rel_drop < 1e-5 and shift < 1e-4:
                break
        prev_inertia = inertia

    return labels.detach().cpu().numpy().astype(np.int32)


# 评分函数：比例偏差 + 方差偏差
def score_stats(
    split_n: np.ndarray,
    split_sum: np.ndarray,
    split_sq: np.ndarray,
    total_n: int,
    global_var: np.ndarray,
    count_weight: float = 20000.0,
    var_weight: float = 1.0,
) -> float:
    ratios = split_n.astype(np.float64) / max(total_n, 1)
    count_loss = float(np.sum((ratios - TARGET_RATIOS) ** 2))

    var_loss = 0.0
    valid = 0
    for s in range(3):
        n = split_n[s]
        if n <= 1:
            continue
        mean = split_sum[s] / n
        var = split_sq[s] / n - mean * mean
        var = np.maximum(var, 1e-8)
        var_loss += float(np.mean(np.abs(var - global_var)))
        valid += 1
    if valid > 0:
        var_loss /= valid
    else:
        var_loss = 10.0

    return count_weight * count_loss + var_weight * var_loss


# 单次贪心簇分配（cluster -> train/val/test）
def greedy_cluster_assignment(
    counts: np.ndarray,
    sums: np.ndarray,
    sqs: np.ndarray,
    global_var: np.ndarray,
    total_n: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    k, d = sums.shape

    order = np.arange(k)
    rng.shuffle(order)
    order = sorted(order.tolist(), key=lambda c: (-int(counts[c]), rng.random()))

    split_n = np.zeros(3, dtype=np.int64)
    split_sum = np.zeros((3, d), dtype=np.float64)
    split_sq = np.zeros((3, d), dtype=np.float64)
    assign = np.full(k, -1, dtype=np.int8)

    for c in order:
        best_s = 0
        best_score = None
        for s in range(3):
            split_n[s] += counts[c]
            split_sum[s] += sums[c]
            split_sq[s] += sqs[c]
            sc = score_stats(split_n, split_sum, split_sq, total_n, global_var)
            split_n[s] -= counts[c]
            split_sum[s] -= sums[c]
            split_sq[s] -= sqs[c]
            if best_score is None or sc < best_score:
                best_score = sc
                best_s = s

        assign[c] = best_s
        split_n[best_s] += counts[c]
        split_sum[best_s] += sums[c]
        split_sq[best_s] += sqs[c]

    final_score = score_stats(split_n, split_sum, split_sq, total_n, global_var)
    return assign, final_score


# 多次重启，选择最优簇分配方案
def pick_best_assignment(
    counts: np.ndarray,
    sums: np.ndarray,
    sqs: np.ndarray,
    global_var: np.ndarray,
    total_n: int,
    base_seed: int,
    restarts: int,
) -> Tuple[np.ndarray, float]:
    best_assign = None
    best_score = None
    for i in range(restarts):
        assign, sc = greedy_cluster_assignment(
            counts=counts,
            sums=sums,
            sqs=sqs,
            global_var=global_var,
            total_n=total_n,
            seed=base_seed + i * 17,
        )
        if best_score is None or sc < best_score:
            best_score = sc
            best_assign = assign
    return best_assign, float(best_score)


# 统计最终 split 的方差接近程度
def compute_split_stats(x: np.ndarray, split_idx: np.ndarray) -> Dict[str, object]:
    stats: Dict[str, object] = {}
    n = x.shape[0]
    global_var = np.var(x, axis=0)
    stats["global_var_mean"] = float(np.mean(global_var))
    stats["global_var_min"] = float(np.min(global_var))
    stats["global_var_max"] = float(np.max(global_var))

    for s, name in enumerate(SPLIT_NAMES):
        idx = np.where(split_idx == s)[0]
        xs = x[idx]
        var = np.var(xs, axis=0) if len(xs) > 1 else np.zeros(x.shape[1], dtype=np.float64)
        rel_diff = np.abs(var - global_var) / np.maximum(np.abs(global_var), 1e-8)
        stats[name] = {
            "count": int(len(idx)),
            "ratio": float(len(idx) / n),
            "var_mean": float(np.mean(var)),
            "var_diff_mean_abs": float(np.mean(np.abs(var - global_var))),
            "var_rel_diff_mean": float(np.mean(rel_diff)),
            "var_rel_diff_max": float(np.max(rel_diff)),
        }
    return stats


# 输出与原格式兼容的 split pkl 文件
def dump_split_pkl(path: Path, image_paths: List[str], ctrl_values: List[np.ndarray], indices: np.ndarray) -> None:
    data = {
        "img_path": [image_paths[i] for i in indices.tolist()],
        "ctrl_values": [np.asarray(ctrl_values[i], dtype=np.float32) for i in indices.tolist()],
    }
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    # 解析输出路径
    args = parse_args()

    dataset_root = args.dataset_root
    output_train = args.output_train or (dataset_root / "train_split.pkl")
    output_val = args.output_val or (dataset_root / "val_split.pkl")
    output_test = args.output_test or (dataset_root / "test_split.pkl")
    report_json = args.report_json or (dataset_root / "cluster_split_report.json")

    # 加载数据并构建 54 维特征（latent24 + ctrl30）
    t0 = time.time()
    print(f"[INFO] loading metadata: {args.metadata_file}")
    ctrl_map = load_metadata_ctrl_map(args.metadata_file)
    print(f"[INFO] metadata rows: {len(ctrl_map)}")

    print(f"[INFO] loading latent + ctrl: {args.latent_file}")
    image_paths, ctrl_values, latent24, ctrl30 = load_latent_and_ctrl(args.latent_file, ctrl_map, dataset_root)
    n = latent24.shape[0]
    if n != len(ctrl_values):
        raise RuntimeError("latent rows and ctrl rows mismatch")
    print(f"[INFO] samples={n} latent_dim={latent24.shape[1]} ctrl_dim={ctrl30.shape[1]}")

    feat54 = np.concatenate([latent24, ctrl30], axis=1).astype(np.float32)

    # 标准化后执行 GPU KMeans 聚类
    print("[INFO] standardizing features")
    z, z_mean, z_std = standardize_np(feat54)
    global_var_z = np.var(z, axis=0).astype(np.float64)

    cluster_device = args.cluster_device
    if cluster_device.startswith("cuda") and not torch.cuda.is_available():
        cluster_device = "cpu"
    print(
        f"[INFO] clustering with TorchKMeans n_clusters={args.n_clusters} "
        f"device={cluster_device} iters={args.cluster_iters}"
    )
    labels = torch_kmeans(
        z=z,
        n_clusters=args.n_clusters,
        device=cluster_device,
        max_iters=args.cluster_iters,
        batch_size=args.cluster_batch,
        seed=args.random_seed,
    )

    counts, sums, sqs = build_cluster_stats(labels, z.astype(np.float64), args.n_clusters)
    non_empty_clusters = int(np.sum(counts > 0))
    print(f"[INFO] non_empty_clusters={non_empty_clusters}")

    # 按 cluster 做 group-aware 划分，目标比例 70/15/15
    print(f"[INFO] assigning clusters with {args.assign_restarts} restarts")
    assign, best_score = pick_best_assignment(
        counts=counts,
        sums=sums,
        sqs=sqs,
        global_var=global_var_z,
        total_n=n,
        base_seed=args.random_seed,
        restarts=args.assign_restarts,
    )
    print(f"[INFO] best_assignment_score={best_score:.6f}")

    split_idx = assign[labels]
    split_counts = np.array([(split_idx == s).sum() for s in range(3)], dtype=np.int64)
    print(
        f"[INFO] split_counts train/val/test = {split_counts[0]}/{split_counts[1]}/{split_counts[2]} "
        f"ratios={split_counts / n}"
    )

    idx_train = np.where(split_idx == 0)[0]
    idx_val = np.where(split_idx == 1)[0]
    idx_test = np.where(split_idx == 2)[0]

    # 写出 train/val/test split 文件
    print(f"[INFO] writing split files to {dataset_root}")
    dump_split_pkl(output_train, image_paths, ctrl_values, idx_train)
    dump_split_pkl(output_val, image_paths, ctrl_values, idx_val)
    dump_split_pkl(output_test, image_paths, ctrl_values, idx_test)

    # 生成划分报告（比例、方差、cluster 分布）
    stats = compute_split_stats(feat54.astype(np.float64), split_idx)
    report = {
        "dataset_root": str(dataset_root),
        "latent_file": str(args.latent_file),
        "metadata_file": str(args.metadata_file),
        "n_samples": int(n),
        "feature_dim": 54,
        "standardize_mean_first5": z_mean[:5].tolist(),
        "standardize_std_first5": z_std[:5].tolist(),
        "cluster_count_requested": int(args.n_clusters),
        "cluster_count_non_empty": non_empty_clusters,
        "cluster_size_min": int(counts[counts > 0].min()) if non_empty_clusters > 0 else 0,
        "cluster_size_max": int(counts.max()) if counts.size > 0 else 0,
        "cluster_size_mean": float(counts[counts > 0].mean()) if non_empty_clusters > 0 else 0.0,
        "split_counts": {
            "train": int(split_counts[0]),
            "val": int(split_counts[1]),
            "test": int(split_counts[2]),
        },
        "split_ratios": {
            "train": float(split_counts[0] / n),
            "val": float(split_counts[1] / n),
            "test": float(split_counts[2] / n),
        },
        "target_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "variance_stats_feature54": stats,
        "best_assignment_score": float(best_score),
        "cluster_device_used": cluster_device,
        "cluster_iters": int(args.cluster_iters),
        "cluster_batch": int(args.cluster_batch),
        "random_seed": int(args.random_seed),
        "assign_restarts": int(args.assign_restarts),
        "outputs": {
            "train_split": str(output_train),
            "val_split": str(output_val),
            "test_split": str(output_test),
        },
    }

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"[DONE] elapsed={elapsed:.1f}s")
    print(f"[DONE] train={output_train}")
    print(f"[DONE] val={output_val}")
    print(f"[DONE] test={output_test}")
    print(f"[DONE] report={report_json}")


if __name__ == "__main__":
    main()
