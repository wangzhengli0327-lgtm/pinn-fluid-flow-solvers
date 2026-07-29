from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DEFAULT_URL = (
    "https://raw.githubusercontent.com/maziarraissi/PINNs/"
    "master/appendix/Data/burgers_shock.mat"
)


def convert_mat_to_txt(source: Path, output: Path) -> tuple[int, float, float]:
    data = loadmat(source)
    missing = {"x", "t", "usol"} - set(data)
    if missing:
        raise KeyError(f"参考数据缺少字段：{', '.join(sorted(missing))}")

    x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
    t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
    solution = np.real(np.asarray(data["usol"], dtype=np.complex128))
    if solution.shape == (x.size, t.size):
        solution = solution.T
    elif solution.shape != (t.size, x.size):
        raise ValueError(
            f"参考解形状应为 {(x.size, t.size)} 或 {(t.size, x.size)}，实际为 {solution.shape}"
        )

    x_grid, t_grid = np.meshgrid(x, t)
    table = np.column_stack((x_grid.reshape(-1), t_grid.reshape(-1), solution.reshape(-1)))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, table, header="x t u", comments="", fmt="%.10e")
    return table.shape[0], float(table[:, 2].min()), float(table[:, 2].max())


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并转换伯格斯方程公开参考数据。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/Burgers_star_data.txt"),
        help="转换后的三列文本数据路径。",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="原始矩阵数据下载地址。")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="pinn-burgers-") as temp_dir:
        source = Path(temp_dir) / "burgers_shock.mat"
        print(f"正在下载：{args.url}")
        urllib.request.urlretrieve(args.url, source)
        rows, value_min, value_max = convert_mat_to_txt(source, args.output)

    print(f"已写入：{args.output}")
    print(f"数据行数：{rows}")
    print(f"速度范围：[{value_min:.6f}, {value_max:.6f}]")


if __name__ == "__main__":
    main()
