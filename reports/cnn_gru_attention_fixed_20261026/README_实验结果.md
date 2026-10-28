# CNN-GRU-Attention：固定协议实验

## 本次结果解读

相对直接对照 CNN-GRU，Attention 的 seen/unseen 速度 RMSE 分别增加约 0.147%/0.077%，基本接近；速度 MAE 分别降低约 1.15%/1.81%。seen 的 ATE 增加约 1.90%，60s RTE 降低约 2.42%；unseen 的 ATE 增加约 13.06%，60s RTE 增加约 2.12%。因此本次单种子实验没有观察到 Attention 在主要速度/轨迹指标上的一致改善，不能宣称加入 Attention 提升了总体性能。RoNIN ResNet18 在本次两个测试集的速度 RMSE 上仍然更低。

Attention 最佳轮次为第 6 轮（验证 MSE 0.09136217），第 30 轮验证 MSE 0.09701039；训练耗时约 48.5 分钟。训练误差继续下降而验证误差后期没有改善，存在过拟合或泛化受限的迹象。下一步应优先按相同协议增加随机种子，检查跨种子稳定性，再决定是否调整正则化、汇总方式或训练策略。任何新策略都应另设实验，不能用测试集选择权重或超参数。

最后验收见 final_audit.json：原基线目录 390 个文件指纹均未变化；训练源码及划分指纹未变化；67 个预测 NPZ、67 张 Attention 独立轨迹图、67 张注意力图、67 张三模型对比图，以及 3 张训练/汇总图，共 204 张 PNG 已检查有效。全部保存预测的指标已复算，75.3 万个测试窗口已覆盖。

Attention 模型从头完成 30 轮训练，以验证集最低 MSE 选择权重。主要参考为 RoNIN ResNet18，直接消融对照为 CNN-GRU；两者引用此前固定协议实验的已保存预测，本次没有重新训练或改写基线结果。

共同训练设置：窗口 200，步长 10，batch 256，Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，MSE，seed=42；无增强、标准化、学习率调度、早停或梯度裁剪。CUDA float32，关闭 TF32，开启确定性算法。数据文件 SHA256、划分、窗口索引、标签及 30 轮样本顺序均与参考实验核对一致。

监督目标是输入窗口的平均 XY 速度；输入只有六通道 IMU，不包含位置或标签。CNN-GRU-Attention 与 CNN-GRU 共用 CNN、GRU、回归头定义，将末状态汇总改成所有 50 个 GRU 时刻的加性时间注意力汇总，新增 2,112 个参数。

| 模型 | 参数量 | 完成轮数 | 最佳轮次 | 最佳验证 MSE |
|---|---:|---:|---:|---:|
| RoNIN ResNet18 | 4,634,882 | 30 | 13 | 0.08518216 |
| CNN-GRU | 78,370 | 30 | 8 | 0.09112469 |
| CNN-GRU-Attention | 80,482 | 30 | 6 | 0.09136217 |

| 测试集 | 模型 | 速度 RMSE (m/s) | MAE (m/s) | 官方 ATE (m) | 官方 60s RTE (m) |
|---|---|---:|---:|---:|---:|
| seen_test | RoNIN ResNet18 | 0.188584 | 0.116286 | 5.468 | 3.648 |
| seen_test | CNN-GRU | 0.208941 | 0.135871 | 7.088 | 4.209 |
| seen_test | CNN-GRU-Attention | 0.209248 | 0.134303 | 7.223 | 4.107 |
| unseen_test | RoNIN ResNet18 | 0.273576 | 0.170870 | 9.895 | 5.818 |
| unseen_test | CNN-GRU | 0.297669 | 0.193547 | 9.047 | 6.227 |
| unseen_test | CNN-GRU-Attention | 0.297897 | 0.190054 | 10.229 | 6.360 |

训练/验证/seen/unseen 划分为 72/13/35/32 条交付序列；测试不筛选异常速度。速度指标按所有窗口的两个分量平均；ATE/RTE 使用官方坐标误差定义、先逐序列计算再等权平均，60s 对应 12,000 帧。轨迹共用 reconstruct()，仅用首个真值位置初始化，不进行轨迹旋转拟合。测试集不参与选模。

基线来源：`E:\codex_ws\handoff_cnn_baseline\results\fixed_resnet18_vs_cnn_gru_20261025`。保存的 Attention 预测已经重新读取并独立复算指标；基线预测与当前数据的索引、标签、时间、真值完全一致，基线汇总指标也已经复核。

目录入口：index.html；comparison/ 为三模型共同轨迹、训练曲线及指标；cnn_gru_attention/seen_test 和 unseen_test 为 Attention 的 67 条轨迹、速度、误差图与逐窗口预测 NPZ；每个 NPZ 额外包含 [N,50] attention_weights，对应图为 *_attention.png。权重对应 CNN 降采样后的 GRU 时刻，并非原始 200 帧的逐帧解释，也不能单独证明因果关系或性能提升。

本实验只有 seed=42。正负变化均如实报告；单次结果不能证明统计显著性。improvement_vs_cnn_gru_percent.json 中正数表示误差下降，负数表示变差。
