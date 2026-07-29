# 参与贡献

感谢你关注本项目。欢迎通过议题反馈缺陷、提出新想法，或通过合并请求改进算法、文档和实验配置。

## 开始之前

1. 在议题列表中搜索是否已经存在相同问题。
2. 对较大的功能改动，建议先创建功能建议议题并说明使用场景。
3. 不要提交课程报告、未经授权的数据集、模型权重、训练日志或大体积生成文件。

## 本地开发

```powershell
python -m venv .venv
python -m pip install -r requirements.txt
python -m compileall -q src scripts task1_burgers_exact.py task2_burgers_forward.py task3_cavity_Re100.py task4_burgers_inverse.py
```

如需运行任务二至任务四，请先按照 [数据准备说明](data/README.md) 准备参考数据。建议先使用 `--mode debug` 验证完整流程。

## 提交修改

1. 从最新主分支创建一个用途明确的分支。
2. 只修改与议题有关的文件，并保持现有代码风格。
3. 对行为变化补充必要的说明、测试或结果图。
4. 提交前执行语法检查和相关调试任务。
5. 在合并请求中说明修改目的、验证方式和可能影响。

请勿在议题、日志或合并请求中公开访问令牌、私有数据和个人敏感信息。
