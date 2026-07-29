# 参考数据准备

本目录只保存数据说明，不提交大体积参考数据文件。

## 伯格斯方程参考数据

任务二和任务四使用公开的伯格斯方程参考解。可在项目根目录运行：

```powershell
python scripts/prepare_burgers_data.py
```

脚本会从 Maziar Raissi 等人的公开物理信息神经网络仓库下载
`burgers_shock.mat`，并转换为本项目使用的三列文本文件：

```text
data/Burgers_star_data.txt
```

三列依次为位置 `x`、时间 `t` 和速度 `u`。

公开原始数据：
https://github.com/maziarraissi/PINNs/blob/master/appendix/Data/burgers_shock.mat

## 方腔流参考数据

任务三需要：

```text
data/Cavity_star_data_Re100.txt
```

三列依次为横坐标 `x`、纵坐标 `y` 和速度模 `speed`。当前没有确认到可公开、
可再分发且与本项目文件完全一致的数据来源，因此仓库不自动下载或重新分发该文件。
获得合法数据后，也可以通过 `--data` 参数传入其他存放位置。

不使用速度模观测进行训练时，任务三仍会用该文件评价预测结果。

## 格式检查

检查已经准备的数据：

```powershell
python scripts/validate_data.py
```

要求两份数据都存在并通过检查：

```powershell
python scripts/validate_data.py --require-all
```
