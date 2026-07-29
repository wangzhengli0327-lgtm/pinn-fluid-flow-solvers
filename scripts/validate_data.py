from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EXPECTED_FILES = {
    "Burgers_star_data.txt": ("x", "t", "u"),
    "Cavity_star_data_Re100.txt": ("x", "y", "speed"),
}


def validate(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path} 至少需要三列数据，实际形状为 {data.shape}")
    data = data[:, :3]
    if data.shape[0] < 3:
        raise ValueError(f"{path} 至少需要三行数据")
    if not np.isfinite(data).all():
        raise ValueError(f"{path} 包含非有限数值")
    return data.shape[0], data.min(axis=0), data.max(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查项目参考数据的格式和数值范围。")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="缺少任一参考数据文件时返回失败。",
    )
    args = parser.parse_args()

    failures: list[str] = []
    for filename, columns in EXPECTED_FILES.items():
        path = args.data_dir / filename
        if not path.exists():
            message = f"缺少：{path}"
            print(message)
            if args.require_all:
                failures.append(message)
            continue
        try:
            rows, lower, upper = validate(path)
        except (OSError, ValueError) as exc:
            failures.append(str(exc))
            print(f"失败：{exc}")
            continue
        ranges = ", ".join(
            f"{name}=[{minimum:.6g}, {maximum:.6g}]"
            for name, minimum, maximum in zip(columns, lower, upper)
        )
        print(f"通过：{path}，{rows} 行，{ranges}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
