# 基于物理信息神经网络的流体流动求解器

本项目使用物理信息神经网络求解非线性流体流动基准问题，包含以下四项实验：

1. 求解具有解析解的伯格斯方程正问题。
2. 求解一维伯格斯方程正问题。
3. 求解雷诺数为 100 的稳态顶盖驱动方腔流。
4. 反演识别伯格斯方程中的扩散系数。

## 项目结构

```
src/                         公共模型、采样、梯度、评价指标和绘图模块
task1_burgers_exact.py       具有解析解的伯格斯方程求解程序
task2_burgers_forward.py     伯格斯方程正问题求解程序
task3_cavity_Re100.py        顶盖驱动方腔流求解程序
task4_burgers_inverse.py     伯格斯方程参数反演程序
requirements.txt             运行依赖清单
```

本仓库仅收录原创代码和运行说明，不包含生成的图像、模型权重、训练日志、报告、课程资料及参考数据集。

## 环境安装

建议使用 3.10 或更高版本的编程语言解释器。

```
python -m venv .venv
python -m pip install -r requirements.txt
```

如需使用显卡加速，请根据设备环境另行安装支持显卡的深度学习框架版本。

## 参考数据

任务二、任务三和任务四需要外部参考数据，用于结果评价或监督采样。请在本地创建 `data/` 目录，并放入以下文件：

```
data/Burgers_star_data.txt
data/Cavity_star_data_Re100.txt
```

每个文本文件都应包含一行表头以及至少三列数值数据。仓库不提供这些参考数据集。

## 运行方法

可使用调试模式进行短时训练测试：

```
python task1_burgers_exact.py --mode debug
python task2_burgers_forward.py --mode debug
python task3_cavity_Re100.py --mode debug
python task4_burgers_inverse.py --mode debug
```

任务二至任务四需要对应的参考数据文件。可通过 `--help` 参数查看任一程序支持的训练与数据选项：

```
python task3_cavity_Re100.py --help
```

训练结果会保存在本地的 `figures/`、`checkpoints/` 和 `logs/` 目录中，这些生成目录不会纳入版本管理。
