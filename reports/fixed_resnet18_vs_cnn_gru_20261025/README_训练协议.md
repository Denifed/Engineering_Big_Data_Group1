# 固定训练协议

本次重跑两项实验：主要参考基线 RoNIN ResNet18、直接对照 CNN-GRU。两者重新初始化，共用同一数据缓存、训练循环和评估代码。

| 项目 | 两模型的共同设置 |
|---|---|
| 训练/验证/测试 | 固定交付名单：72/13/35 seen/32 unseen |
| 网络输入 | 六通道 IMU，[B,6,200] |
| 预测与标签 | 窗口平均水平速度 [vx,vy]，从位置端点和真实时间差计算 |
| 窗口/步长 | 200/10 |
| batch | 256，保留末尾不足 batch 的样本 |
| 训练预算 | 完整 30 轮，不早停 |
| 优化器 | Adam，lr=1e-4，betas=(0.9,0.999)，eps=1e-8 |
| 权重衰减/AMSGrad | 0 / false |
| 损失 | 两个速度分量的 MSE |
| 随机种子 | 42 |
| 窗口顺序 | 第 e 轮使用独立 CPU generator，seed=42+e-1；逐轮校验 SHA256 |
| 数据增强/标准化 | 均关闭 |
| 学习率调度/梯度裁剪 | 均关闭 |
| 数据筛选 | 训练过滤跨越瞬时速度 >10 m/s 区间的窗口；验证、测试不筛选 |
| 数值设置 | float32，TF32 关闭，确定性算法开启，cuDNN benchmark 关闭 |
| 选模 | 验证 MSE 最低的权重；相同分数保留更早轮次 |
| 最终测试 | 共同 seen/unseen 名单、相同 batch 512 |
| 轨迹与指标 | 同一速度积分/插值重建；官方坐标误差 ATE、60 秒 RTE |
| 汇总 | 速度按窗口及分量平均，轨迹指标先逐序列再等权平均 |

## 保留的模型结构差异

- RoNIN ResNet18：官方网络定义，4,634,882 参数，包括原有 FC Dropout=0.5。
- CNN-GRU：三层 CNN、单层单向 GRU、最终隐藏状态回归，78,370 参数，无 Dropout。
- 两者具有不同的容量、初始化规则和网络内部结构；共同训练预算不意味着相同计算量。
- 这是主要参考基线与时序模型对照。后续评估 Attention 的直接消融应比较 CNN-GRU 与 CNN-GRU-Attention。

## 证据与输出

- protocol.json：机器可读的共同协议及 SHA256。
- dataset_manifest.json：共享训练/验证窗口索引、标签摘要和序列清单。
- data_order_manifest.json：逐轮样本顺序摘要。
- data_fingerprints.csv、source_fingerprints.json、source_snapshot/：数据与代码证据。
- resnet18/、cnn_gru/：各自的配置、训练日志、曲线及 best/last 权重；评估后增加 seen_test/、unseen_test/ 与指标。
- comparison/：两模型与参考轨迹的共同图表，训练曲线及指标对比。
- acceptance_checks.json、result_verification.json：完成后的训练与评估复核。
- index.html、README_实验结果.md：完成后的可视化总览和结果报告。

旧实验不覆盖。本轮只有一个随机种子；后续需多种子验证稳定性，不能仅凭一次训练宣称统计显著提升。

