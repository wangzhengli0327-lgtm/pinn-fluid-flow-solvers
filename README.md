# 基于物理信息神经网络的流体流动求解器

[中文说明](README.md) · [English](README_EN.md)

[![代码质量检查](https://github.com/wangzhengli0327-lgtm/pinn-fluid-flow-solvers/actions/workflows/quality.yml/badge.svg)](https://github.com/wangzhengli0327-lgtm/pinn-fluid-flow-solvers/actions/workflows/quality.yml)

本项目使用物理信息神经网络（PINN）求解典型的非线性流体流动问题，覆盖偏微分方程正问题、二维不可压缩流动和方程参数反演。训练过程同时受到控制方程、初始条件、边界条件和观测数据约束，可以在较少标注数据的条件下学习满足物理规律的近似解。

![四项任务结果总览](assets/project-overview.png)

## 主要特点

- 使用 PyTorch 自动微分计算控制方程所需的一阶和二阶导数。
- 将方程残差、初始条件、边界条件和数据误差组合为训练目标。
- 采用 Adam 预训练与 L-BFGS 精细优化相结合的训练策略。
- 支持 CPU 与 GPU 自动选择，并提供 `debug`、`light`、`full` 三种运行规模。
- 固定随机种子，自动保存模型权重、训练历史、评估指标和结果图像。
- 四个任务拥有独立入口，同时复用统一的数据、网络、采样、训练和绘图模块。

## 四项任务

| 任务 | 研究问题 | 输入与输出 | 数据用途 | 主要评价内容 |
| --- | --- | --- | --- | --- |
| 任务一 | 具有解析解的一维伯格斯方程正问题 | 输入位置与时间，输出速度 | 不需要外部数据 | 预测解与解析解的相对二范数误差 |
| 任务二 | 一维伯格斯方程正问题 | 输入位置与时间，输出速度 | 参考解用于评价，也可参与训练 | 预测解与数值参考解的相对二范数误差 |
| 任务三 | 雷诺数为 100 的稳态顶盖驱动方腔流 | 输入二维坐标，输出速度与压力 | 速度模用于评价，也可参与训练 | 预测速度模与参考值的相对二范数误差 |
| 任务四 | 伯格斯方程扩散系数反演 | 输入位置与时间，输出速度并识别系数 | 观测数据用于训练与评价 | 系数相对误差和速度场相对二范数误差 |

### 任务一：具有解析解的伯格斯方程

求解带黏性项的一维伯格斯方程：

$$
\frac{\partial u}{\partial t}
+u\frac{\partial u}{\partial x}
-\nu\frac{\partial^2u}{\partial x^2}=0.
$$

计算区域为 $x\in[0,1]$、$t\in[0,1]$，默认雷诺数为 $500$，黏性系数为 $\nu=1/500$。程序内置解析解，不依赖外部数据。

入口：`task1_burgers_exact.py`

### 任务二：一维伯格斯方程正问题

求解经典一维伯格斯方程，黏性系数为 $\nu=0.01/\pi$，计算区域为 $x\in[-1,1]$、$t\in[0,1]$，初始条件为 $u(x,0)=-\sin(\pi x)$。左右边界满足周期条件。参考数据默认用于评价；添加 `--use-data` 后也可采样监督点参与训练。

入口：`task2_burgers_forward.py`

### 任务三：稳态顶盖驱动方腔流

求解单位方腔内雷诺数为 $100$ 的二维稳态不可压缩流动。控制方程包括连续性方程和两个方向的纳维–斯托克斯动量方程。方腔顶部以单位速度向右运动，其余壁面满足无滑移条件。网络同时预测两个速度分量和压力。

入口：`task3_cavity_Re100.py`

### 任务四：伯格斯方程参数反演

根据离散观测数据反演伯格斯方程中的未知扩散系数：

$$
\frac{\partial u}{\partial t}
+u\frac{\partial u}{\partial x}
-\lambda\frac{\partial^2u}{\partial x^2}=0.
$$

网络同时拟合速度观测值、约束方程残差，并将 $\lambda$ 作为可训练参数共同优化。程序通过正值映射保证识别出的扩散系数始终大于零，参考真值为 $\lambda=0.01/\pi$。

入口：`task4_burgers_inverse.py`

## 结果预览

下图来自本项目实际运行结果，完整训练会在 `figures/` 中生成更多预测场、误差场、剖面对比、损失曲线和流线图。

| 任务一：解析解对照 | 任务二：伯格斯方程预测 |
| --- | --- |
| ![任务一预测结果](assets/results/task1-prediction.png) | ![任务二预测结果](assets/results/task2-prediction.png) |
| **任务三：方腔流流线** | **任务四：参数反演** |
| ![任务三流线结果](assets/results/task3-streamlines.png) | ![任务四反演结果](assets/results/task4-inverse.png) |

## 环境安装

建议使用 Python 3.10 或更高版本：

```powershell
python -m venv .venv
python -m pip install -r requirements.txt
```

如需 GPU 加速，请根据设备和驱动环境安装相匹配的 PyTorch 版本。

## 参考数据

任务二和任务四使用同一份伯格斯方程公开参考解。运行以下命令即可下载原始数据并转换为项目所需格式：

```powershell
python scripts/prepare_burgers_data.py
```

任务三需要 `Cavity_star_data_Re100.txt` 作为评价参考。由于尚未确认到与本项目文件完全一致且允许再分发的公开来源，仓库不会自动下载或提交该文件。数据列格式、来源说明和完整校验命令见 [数据准备说明](data/README.md)。

## 快速运行

调试模式使用较少的采样点和训练轮数，适合检查环境、数据路径和程序流程：

```powershell
python task1_burgers_exact.py --mode debug
python task2_burgers_forward.py --mode debug
python task3_cavity_Re100.py --mode debug
python task4_burgers_inverse.py --mode debug
```

三种运行规模：

- `debug`：快速检查程序是否可以运行。
- `light`：以较低计算成本进行较完整的训练。
- `full`：使用更大的采样规模和训练轮数获得最终结果。

查看任一程序支持的参数：

```powershell
python task3_cavity_Re100.py --help
```

## 项目结构

```text
src/
├── data_utils.py            数据读取、张量转换和分批预测
├── gradients.py             自动微分封装
├── metrics.py               均方误差和相对二范数误差
├── models.py                全连接神经网络与随机种子设置
├── plot_utils.py            标量场、剖面、矢量场和流线绘制
├── sampling.py              区域、初始点和边界点采样
└── train_utils.py           设备选择、训练历史和结果保存

scripts/
├── prepare_burgers_data.py  下载并转换伯格斯方程公开数据
└── validate_data.py         校验参考数据格式

task1_burgers_exact.py       任务一入口
task2_burgers_forward.py     任务二入口
task3_cavity_Re100.py        任务三入口
task4_burgers_inverse.py     任务四入口
```

仓库仅收录求解器代码、说明文档和精选结果图，不包含模型权重、训练日志、课程报告或大体积参考数据集。

## 结果输出与复现

程序会自动创建以下目录：

| 目录 | 内容 |
| --- | --- |
| `figures/` | 预测场、误差场、剖面对比、损失曲线、矢量图和流线图 |
| `checkpoints/` | 训练后的模型权重和反演参数 |
| `logs/` | 训练历史、最终评价指标和运行配置 |

默认随机种子为 `2024`，可通过 `--seed` 修改。不同硬件和 PyTorch 版本可能产生轻微数值差异。正式实验建议使用 `light` 或 `full` 模式，并保存命令行参数、依赖版本和生成日志。

## 参与改进与引用

欢迎通过议题报告问题、提出建议，或提交合并请求改进算法、文档和实验配置。具体流程见 [参与贡献说明](CONTRIBUTING.md)。

如果本项目对你的研究或学习有帮助，请使用 [CITATION.cff](CITATION.cff) 中的信息进行引用。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。你可以在保留版权与许可声明的前提下使用、修改和分发代码。
