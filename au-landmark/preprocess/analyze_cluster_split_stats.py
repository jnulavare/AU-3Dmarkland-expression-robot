#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import cluster_split_x2c as cs


# 读取统计脚本参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze split stats for latent24/ctrl30/cluster distribution.")
    p.add_argument("--dataset-root", type=Path, default=Path(r"E:\DD\Git\X2C"))
    p.add_argument("--latent-file", type=Path, default=Path(r"D:\code\test2\x2c_data_bundle\LATENT24_X2C_gpu.csv.gz"))
    p.add_argument("--metadata-file", type=Path, default=Path(r"D:\code\test2\x2c_data_bundle\metadata.jsonl"))
    p.add_argument("--report-file", type=Path, default=Path(r"E:\DD\Git\X2C\cluster_split_report.json"))
    p.add_argument("--train-pkl", type=Path, default=Path(r"E:\DD\Git\X2C\train_split.pkl"))
    p.add_argument("--val-pkl", type=Path, default=Path(r"E:\DD\Git\X2C\val_split.pkl"))
    p.add_argument("--test-pkl", type=Path, default=Path(r"E:\DD\Git\X2C\test_split.pkl"))
    p.add_argument("--out-latent-csv", type=Path, default=Path(r"E:\DD\Git\X2C\split_latent24_mean_var.csv"))
    p.add_argument("--out-ctrl-csv", type=Path, default=Path(r"E:\DD\Git\X2C\split_ctrl30_mean_var.csv"))
    p.add_argument("--out-cluster-json", type=Path, default=Path(r"E:\DD\Git\X2C\split_cluster_distribution.json"))
    return p.parse_args()


# 统一路径格式，避免分隔符差异
def norm_path(p: str) -> str:
    return str(Path(p)).replace("/", "\\").lower()


# 加载 split 文件中的图片路径集合
def load_split_set(pkl_path: Path) -> set[str]:
    obj = pickle.load(open(pkl_path, "rb"))
    return {norm_path(p) for p in obj["img_path"]}


# 导出每维度 mean/var 到 CSV
def write_feature_stats_csv(path: Path, split_stats: Dict[str, Tuple[np.ndarray, np.ndarray]], prefix: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "dim", "feature", "mean", "var"])
        for split_name, (mean_arr, var_arr) in split_stats.items():
            for i, (m, v) in enumerate(zip(mean_arr.tolist(), var_arr.tolist())):
                w.writerow([split_name, i, f"{prefix}_{i:02d}", float(m), float(v)])


# 统计 cluster size 的分布摘要（最小/最大/分位数）
def size_summary(sizes: np.ndarray) -> Dict[str, float]:
    if sizes.size == 0:
        return {
            "cluster_count": 0,
            "size_min": 0,
            "size_max": 0,
            "size_mean": 0.0,
            "size_median": 0.0,
            "size_p10": 0.0,
            "size_p25": 0.0,
            "size_p75": 0.0,
            "size_p90": 0.0,
        }
    return {
        "cluster_count": int(sizes.size),
        "size_min": int(np.min(sizes)),
        "size_max": int(np.max(sizes)),
        "size_mean": float(np.mean(sizes)),
        "size_median": float(np.median(sizes)),
        "size_p10": float(np.percentile(sizes, 10)),
        "size_p25": float(np.percentile(sizes, 25)),
        "size_p75": float(np.percentile(sizes, 75)),
        "size_p90": float(np.percentile(sizes, 90)),
    }


# cluster size 直方图统计
def histogram_counts(sizes: np.ndarray, bins: List[int]) -> Dict[str, int]:
    # bins as upper-inclusive buckets
    out: Dict[str, int] = {}
    prev = 1
    remaining = sizes.copy()
    for b in bins:
        c = int(np.sum((sizes >= prev) & (sizes <= b)))
        out[f"{prev}-{b}"] = c
        prev = b + 1
    out[f"{prev}+"] = int(np.sum(sizes >= prev))
    return out


def main() -> None:
    # 读取已有划分报告参数（聚类簇数、迭代、seed 等）
    args = parse_args()

    report = json.load(open(args.report_file, "r", encoding="utf-8"))
    n_clusters = int(report["cluster_count_requested"])
    cluster_iters = int(report.get("cluster_iters", 60))
    cluster_batch = int(report.get("cluster_batch", 25000))
    seed = int(report.get("random_seed", 42))
    cluster_device = str(report.get("cluster_device_used", "cuda"))
    if cluster_device.startswith("cuda") and not torch.cuda.is_available():
        cluster_device = "cpu"

    # 加载 latent24 + ctrl30，并对齐 split 索引
    ctrl_map = cs.load_metadata_ctrl_map(args.metadata_file)
    image_paths, ctrl_values, latent24, ctrl30 = cs.load_latent_and_ctrl(args.latent_file, ctrl_map, args.dataset_root)
    n = latent24.shape[0]

    path_to_idx = {norm_path(p): i for i, p in enumerate(image_paths)}
    train_set = load_split_set(args.train_pkl)
    val_set = load_split_set(args.val_pkl)
    test_set = load_split_set(args.test_pkl)

    split_idx = np.full(n, -1, dtype=np.int8)
    for p in train_set:
        if p in path_to_idx:
            split_idx[path_to_idx[p]] = 0
    for p in val_set:
        if p in path_to_idx:
            split_idx[path_to_idx[p]] = 1
    for p in test_set:
        if p in path_to_idx:
            split_idx[path_to_idx[p]] = 2

    if np.any(split_idx < 0):
        missing = int(np.sum(split_idx < 0))
        raise RuntimeError(f"{missing} samples are not covered by train/val/test splits")

    # 计算 latent24 / ctrl30 在 train/val/test 各维 mean/var
    # per-dimension mean/var
    latent_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    ctrl_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    split_names = ["train", "val", "test"]
    for s, name in enumerate(split_names):
        idx = np.where(split_idx == s)[0]
        latent_stats[name] = (latent24[idx].mean(axis=0), latent24[idx].var(axis=0))
        ctrl_stats[name] = (ctrl30[idx].mean(axis=0), ctrl30[idx].var(axis=0))

    write_feature_stats_csv(args.out_latent_csv, latent_stats, "latent")
    write_feature_stats_csv(args.out_ctrl_csv, ctrl_stats, "ctrl")

    # 复现聚类标签，统计每个集合覆盖的 cluster 数量与 size 分布
    # cluster labels (recompute with saved params)
    feat54 = np.concatenate([latent24, ctrl30], axis=1).astype(np.float32)
    z, _, _ = cs.standardize_np(feat54)
    labels = cs.torch_kmeans(
        z=z,
        n_clusters=n_clusters,
        device=cluster_device,
        max_iters=cluster_iters,
        batch_size=cluster_batch,
        seed=seed,
    )

    global_counts = np.bincount(labels, minlength=n_clusters)
    non_empty_global = np.where(global_counts > 0)[0]
    cluster_json: Dict[str, object] = {
        "n_samples": int(n),
        "n_clusters_requested": n_clusters,
        "n_clusters_non_empty_global": int(non_empty_global.size),
        "global_cluster_size_summary": size_summary(global_counts[global_counts > 0]),
        "split": {},
        "cluster_params": {
            "device_used": cluster_device,
            "cluster_iters": cluster_iters,
            "cluster_batch": cluster_batch,
            "seed": seed,
        },
    }

    bins = [5, 10, 20, 50, 100, 200, 500, 1000]
    for s, name in enumerate(split_names):
        idx = np.where(split_idx == s)[0]
        split_labels = labels[idx]
        u, c = np.unique(split_labels, return_counts=True)
        summary = size_summary(c)
        cluster_json["split"][name] = {
            "sample_count": int(idx.size),
            "cluster_count": int(u.size),
            "cluster_size_summary": summary,
            "cluster_size_histogram": histogram_counts(c, bins),
        }

    # 写出统计结果文件
    with args.out_cluster_json.open("w", encoding="utf-8") as f:
        json.dump(cluster_json, f, ensure_ascii=False, indent=2)

    print(f"[DONE] latent stats: {args.out_latent_csv}")
    print(f"[DONE] ctrl stats:   {args.out_ctrl_csv}")
    print(f"[DONE] clusters:     {args.out_cluster_json}")
    for s in split_names:
        info = cluster_json["split"][s]
        print(
            f"[INFO] {s}: samples={info['sample_count']} clusters={info['cluster_count']} "
            f"size_mean={info['cluster_size_summary']['size_mean']:.2f} "
            f"size_min={info['cluster_size_summary']['size_min']} "
            f"size_max={info['cluster_size_summary']['size_max']}"
        )


if __name__ == "__main__":
    main()
