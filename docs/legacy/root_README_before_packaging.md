# RoNIN 惯性里程计项目

当前使用处理好的 RoNIN NPZ 估计窗口平均水平速度。输入为 IMU 角速度和加速度，标签由窗口两端参考 XY 位置除以真实时间差生成。

## 当前入口

- cnn_baseline/：CNN、CNN-GRU、CNN-GRU + 时间注意力、平均速度加载器、训练、评估、测试与说明。
- ronin-master/：提供的 RoNIN 源码。
- scripts/run_ronin_npz_baseline.py：官方 ResNet18 的 NPZ 适配训练及轨迹评估。
- scripts/summarize_ronin_run.py：ResNet 结果整理。
- RoNIN_Team_Delivery/：152 条正式 NPZ、固定划分、质量报告与上游文档。
- results/ronin_resnet18_npz_20261025/：已完成的 ResNet 实验，含配置和源码快照。
- results/cnn_comparison/：旧单帧速度标签实验，仅作历史材料。
- cnn_baseline/run_fixed_comparison.py：用完全相同的训练参数、窗口顺序和评估协议重跑 ResNet18 与 CNN-GRU；新结果位于 results/fixed_resnet18_vs_cnn_gru_20261025/。
- cnn_baseline/model_cnn_gru_attention.py：CNN-GRU + 加性时间注意力模型；train_attention_fixed.py 接入已验收的固定训练协议，--check-only 可只检查配置。

旧 source/ 中三个 CNN 相关模块及 scripts/train_cnn_comparison.py、scripts/evaluate_cnn_comparison.py 已迁入并更新到 cnn_baseline/，不再保留旧活动入口。

## CNN 数据协议

features[start:end] → [B,6,200] → 网络 → [B,2]。

y = (positions[end,:2] - positions[start,:2]) / (timestamps[end] - timestamps[start])

window_size=200，step_size=10。NPZ targets 仅用于训练异常筛选，不再直接使用 targets[end] 监督。--model cnn_gru_attention 选择时间注意力模型。

新评估代码会拒绝无新标签协议的旧 checkpoint；旧指标不是新协议结果。正式训练通过 cnn_baseline/train.py 执行，详细命令见 cnn_baseline/README.md。

## 环境

使用 E:\conda_envs\torch_env\python.exe，依赖 NumPy、PyTorch、Matplotlib。正式数据已包含在 RoNIN_Team_Delivery/dataset，无需重复下载或转换。

