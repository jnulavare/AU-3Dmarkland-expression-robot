#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


# 构建 30 维电机控制值的归一化区间定义
def build_ranges() -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    ranges.extend([(0.0, 1.0)] * 4)        # 0-3 Brows
    ranges.extend([(-1.0, 2.0)] * 4)       # 4-7 Eyelids
    ranges.extend([(-130.0, 130.0)] * 2)   # 8-9 Gaze
    ranges.extend([(-20.0, 20.0)] * 3)     # 10-12 Head
    ranges.extend([(0.0, 1.0)] * 14)       # 13-26 Mouth
    ranges.extend([(-10.0, 10.0)] * 2)     # 27-28 Neck
    ranges.extend([(0.0, 1.0)] * 1)        # 29 Nose
    if len(ranges) != 30:
        raise RuntimeError(f"Range length mismatch: {len(ranges)}")
    return ranges


# 单值归一化并裁剪到 [0,1]
def normalize_value(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    y = (x - lo) / (hi - lo)
    if y < 0.0:
        return 0.0
    if y > 1.0:
        return 1.0
    return float(y)


def main() -> None:
    # 读取输入输出路径参数
    parser = argparse.ArgumentParser(description="Normalize X2C metadata ctrl_value to [0,1].")
    parser.add_argument("--input", type=Path, default=Path("metadata.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("metadata_normalize.jsonl"))
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("metadata_normalize.stats.json"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    ranges = build_ranges()
    count = 0
    bad_rows = 0
    mins = [1.0] * 30
    maxs = [0.0] * 30

    # 全量遍历 metadata.jsonl 并按维度归一化
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            file_name = obj.get("file_name")
            ctrl = obj.get("ctrl_value")
            if not isinstance(ctrl, list) or len(ctrl) != 30:
                bad_rows += 1
                continue

            norm = []
            for i, v in enumerate(ctrl):
                lo, hi = ranges[i]
                nv = normalize_value(float(v), lo, hi)
                norm.append(nv)
                if nv < mins[i]:
                    mins[i] = nv
                if nv > maxs[i]:
                    maxs[i] = nv

            out = {"file_name": file_name, "ctrl_value": norm}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    # 输出统计信息，便于检查归一化范围
    stats = {
        "input": str(args.input),
        "output": str(args.output),
        "total_written": count,
        "bad_rows_skipped": bad_rows,
        "dimension": 30,
        "ranges_used": build_ranges(),
        "normalized_min_per_dim": mins,
        "normalized_max_per_dim": maxs,
    }
    with args.stats_output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote={count} rows  bad_rows={bad_rows}")
    print(f"[DONE] output={args.output}")
    print(f"[DONE] stats={args.stats_output}")


if __name__ == "__main__":
    main()
