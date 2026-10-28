# 固定训练参数：RoNIN ResNet18 与 CNN-GRU

两模型均从头完成 30 轮训练。

共同设置：窗口 200、步长 10、batch 256、Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、MSE、seed=42；不使用增强、标准化、调度、早停或梯度裁剪。float32，TF32 关闭，确定性算法开启。每轮窗口顺序 SHA256 已逐轮核对一致。相同 epoch 数意味着相同样本访问与优化步数，不代表相同计算量。

监督目标为窗口平均 XY 速度。训练过滤瞬时速度 >10 m/s 的位移区间，验证和测试不筛选。采用交付的 72/13/35/32 划分。两模型共用数据缓存和批处理，仅网络结构不同。ResNet 保留官方网络的 FC Dropout=0.5；CNN-GRU 没有 Dropout，这属于保留的结构差异。

| 模型 | 参数量 | 完成轮数 | 最佳轮次 | 最佳验证 MSE |
|---|---:|---:|---:|---:|
| RoNIN ResNet18 | 4,634,882 | 30 | 13 | 0.08518216 |
| CNN-GRU | 78,370 | 30 | 8 | 0.09112469 |

| 测试集 | 模型 | 速度 RMSE (m/s) | MAE (m/s) | 官方 ATE (m) | 官方 60s RTE (m) |
|---|---|---:|---:|---:|---:|
| seen_test | RoNIN ResNet18 | 0.188584 | 0.116286 | 5.468 | 3.648 |
| seen_test | CNN-GRU | 0.208941 | 0.135871 | 7.088 | 4.209 |
| unseen_test | RoNIN ResNet18 | 0.273576 | 0.170870 | 9.895 | 5.818 |
| unseen_test | CNN-GRU | 0.297669 | 0.193547 | 9.047 | 6.227 |

两者均使用验证 MSE 最小的权重，未用测试集选模。速度指标对窗口和两个分量平均；ATE/RTE 沿用官方坐标误差定义，先逐序列计算再等权平均。轨迹沿用同一 reconstruct 函数，以首位置初始化，不做旋转拟合。保存预测已独立复算指标，并核对两模型的索引、时间、标签和参考轨迹一致。

每个模型目录包含配置、日志、源码引用、best/last 权重、训练曲线、汇总及逐序列指标、seen_test/、unseen_test/。comparison/ 包含共同轨迹图、训练曲线对比、测试汇总；index.html 为总览。protocol.json、dataset_manifest.json、data_order_manifest.json、data_fingerprints.csv、source_snapshot/、result_verification.json 保存复核证据。

这是单个种子在共同协议下的对照，不是论文原始条件复现，也不能独立证明结果具有统计显著性。旧实验未覆盖。
