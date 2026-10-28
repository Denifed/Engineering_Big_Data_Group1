# RoNIN ResNet18 基线实验结果

本次使用交付的 NPZ 数据，调用 `ronin-master/source/model_resnet1d.py` 中未修改的官方 ResNet18 网络训练，并在全部 35 条 seen_test、32 条 unseen_test 序列评估。数据预处理正确性作为本次实验前提。

## 训练结果

- 完成轮数：13 / 30；最佳轮数：7。
- 最佳验证 MSE：0.07193645。
- 训练轮次累计耗时：16.33 分钟；训练与最终评估总耗时：18.51 分钟（不含最初数据加载）。
- CUDA GPU：NVIDIA GeForce RTX 4060 Laptop GPU；Python：E:\conda_envs\torch_env\python.exe；PyTorch：2.14.0+cu126。
- 训练 72 条、验证 13 条；窗口 200，步长 10；训练 batch 256；Adam 1e-4；MSE；随机种子 42。
- 每轮完整遍历训练窗口；训练速度异常筛选 >10 m/s；验证和测试不筛选。
- 水平随机旋转、起点随机偏移；float32；TF32 关闭。
- 验证集连续 6 轮无改善时早停，测试集未用于选模。

## 测试结果

| 测试集 | 窗口数 | MSE | RMSE (m/s) | MAE (m/s) | 官方 ATE (m) | 官方 60s RTE (m) |
|---|---:|---:|---:|---:|---:|---:|
| seen_test | 388,054 | 0.030022 | 0.173270 | 0.113597 | 5.596 | 3.560 |
| unseen_test | 364,722 | 0.059067 | 0.243038 | 0.156749 | 7.420 | 5.269 |

速度 MSE/RMSE/MAE 对所有窗口和两个分量加权计算。ATE/RTE 先按序列计算，再对序列等权平均。`sequence_metrics.csv` 另存欧氏距离 ATE 和终点误差。

## 重要协议说明

1. 标签是整个窗口的平均速度 `(p[start+200]-p[start])/(t[start+200]-t[start])`，对应官方 ResNet 的 StridedSequenceDataset。旧 CNN/CNN-GRU 实验使用 `targets[end]` 单步速度，因此不能直接横向比较旧 CSV 数字；公平比较须重训并使用相同标签协议。
2. 使用交付的 72/13/35/32 划分，并非源码自带名单；所有 NPZ 姿态来源为 game_rv；训练增加了项目约定的 >10 m/s 异常筛选。这是官方网络在当前项目数据协议上的基线，不能直接宣称达到论文原条件复现。
3. 网络和官方 metric.py 未修改。NPZ 适配、GPU 常驻窗口批处理、边界安全的随机偏移及实验管理由独立入口实现。导出 NPZ 缺失原始末帧位置，序列末端窗口数与直接 HDF5 路径可能有极小差异。
4. 轨迹按官方 `recon_traj_with_preds` 公式重建：预测速度乘平均输出间隔后累计，以真实首位置初始化，端点延拓后线性插值到原始时间。未做额外平移、旋转或 SE(2) 最优拟合。此公式已与本地官方函数逐元素核对。
5. 官方 ATE/RTE 使用每个坐标的平方误差均值开方；二维欧氏 RMSE 是其 sqrt(2) 倍。RTE 用 12,000 帧，即约 60 秒；不足 60 秒沿用官方缩放。不能混用不同定义。
6. 每个预测代表约 1 秒平均速度，间隔约 0.05 秒。重建沿用官方约定，不是把重叠窗口各积分一整秒。
7. cuDNN benchmark 开启，未强制逐位确定性；种子、数据指纹、代码快照和完整配置已保存。

## 文件入口

- `index.html`：本地可视化总览，可切换全部测试序列。
- `training_curves.png`：训练和验证损失、学习率。
- `test_summary.png`、`ate_distribution.png`：测试汇总和轨迹误差分布。
- `seen_test/*.png`、`unseen_test/*.png`：每条序列的轨迹、速度、误差图。
- `seen_test/*_predictions.npz`、`unseen_test/*_predictions.npz`：窗口索引、时刻、预测/真实速度及完整轨迹。
- `test_metrics.csv`、`sequence_metrics.csv`：汇总和逐序列指标。
- `checkpoint_best.pt`、`checkpoint_last.pt`：最佳与最后一轮权重、优化器及配置。
- `config.json`、`training_log.csv`、`run.log`、`summary.json`：配置与完整运行记录。
- `source_snapshot/`、`data_fingerprints.csv`：代码、名单快照和数据 SHA256。
- `acceptance_checks.json`、`result_verification.json`：加载、重建和结果复核证据。

## 重跑命令

在项目根目录运行，务必使用一个新的输出目录：

```powershell
& 'E:\conda_envs\torch_env\python.exe' -B -u scripts/run_ronin_npz_baseline.py --out results/ronin_resnet18_npz_new_run --epochs 30 --patience 6 --batch-size 256
```

原始交付数据、旧实验结果与官方源码均保留。
