# CNN / CNN-GRU / CNN-GRU-Attention：固定协议三路线对照

## 本次主要发现

1. CNN-GRU 相较纯 CNN，在本次 seed=42 的 seen/unseen 测试上均降低速度 RMSE、MAE、ATE 和 RTE。速度 RMSE 分别降低约 2.97%/1.59%，ATE 降低约 8.05%/13.35%，RTE 降低约 7.24%/9.02%。这是共同训练协议下的结构对照结果，CNN 与 CNN-GRU 的参数量及回归头输入维度也不同。
2. Attention 相较 CNN-GRU，速度 MAE 在 seen/unseen 分别降低约 1.15%/1.81%，但 RMSE 分别增加约 0.15%/0.08%；seen RTE 降低约 2.42%，而 unseen ATE 增加约 13.06%。因此不能概括为逐步加入模块会让所有指标持续改善。
3. 本次 CNN 完整训练 30 轮，约 40.6 分钟，选择第 22 轮最佳验证 MSE 0.09265349 的权重。CNN-GRU 最佳第 8 轮，Attention 最佳第 6 轮；均未用测试结果选择权重。
4. 最终检查通过：全部 67 条序列、752,776 个测试窗口已评估；CNN 保存预测已复算，既有两个模型原指标已复核。137 张 PNG 和总览静态链接有效；两个参考目录共 710 个文件指纹均未改变。详见 final_audit.json。

本次将原有纯 CNN 结构从头训练 30 轮，CNN-GRU 和 Attention 复用已完成的同协议实验及预测。旧实验目录未改写，旧纯 CNN 结果未混入本表。

共同配置：窗口 200、步长 10、batch 256、Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，MSE，seed=42；无增强、标准化、学习率调度、早停或梯度裁剪。CUDA FP32，关闭 TF32，开启确定性算法。三者数据文件、72/13/35/32 序列划分、训练/验证窗口索引、窗口平均 XY 速度标签和每轮排列均已核对一致。训练集 787,920 个有效窗口，验证集 136,686 个窗口；测试集不筛选异常速度。

| 模型 | 参数量 | 完成轮数 | 最佳轮次 | 最佳验证 MSE |
|---|---:|---:|---:|---:|
| CNN | 45,218 | 30 | 22 | 0.09265349 |
| CNN-GRU | 78,370 | 30 | 8 | 0.09112469 |
| CNN-GRU-Attention | 80,482 | 30 | 6 | 0.09136217 |

| 测试集 | 模型 | 速度 RMSE (m/s) | MAE (m/s) | 官方 ATE (m) | 官方 60s RTE (m) |
|---|---|---:|---:|---:|---:|
| seen_test | CNN | 0.215343 | 0.144852 | 7.709 | 4.537 |
| seen_test | CNN-GRU | 0.208941 | 0.135871 | 7.088 | 4.209 |
| seen_test | CNN-GRU-Attention | 0.209248 | 0.134303 | 7.223 | 4.107 |
| unseen_test | CNN | 0.302468 | 0.201477 | 10.441 | 6.845 |
| unseen_test | CNN-GRU | 0.297669 | 0.193547 | 9.047 | 6.227 |
| unseen_test | CNN-GRU-Attention | 0.297897 | 0.190054 | 10.229 | 6.360 |

## 逐步结构变化对应的误差变化

下表正数表示误差降低，负数表示变差；没有预设逐步改善。

| 测试集 | 比较 | RMSE 降低比例 | ATE 降低比例 | RTE 降低比例 |
|---|---|---:|---:|---:|
| seen_test | CNN → CNN-GRU | +2.97% | +8.05% | +7.24% |
| unseen_test | CNN → CNN-GRU | +1.59% | +13.35% | +9.02% |
| seen_test | CNN-GRU → CNN-GRU-Attention | -0.15% | -1.90% | +2.42% |
| unseen_test | CNN-GRU → CNN-GRU-Attention | -0.08% | -13.06% | -2.12% |

## 结构与解释边界

三者均使用三层卷积（32/64/128 通道，核 7/5/3）。CNN 采用全局平均池化和 128→64→2 回归头；CNN-GRU 使用 hidden_size=64 的单层 GRU 最终状态和 64→64→2 回归头。因此 CNN→CNN-GRU 同时改变汇总方式、参数量和回归头输入维度，是结构对照，不能把差异全部归因于孤立插入一个 GRU。CNN-GRU→Attention 保留骨干与回归头，改为对 50 个 GRU 状态进行加性时间注意力汇总，新增 2,112 个参数。相同 seed 下相同形状的共享 CNN 初始参数已逐元素验证一致；Attention 与 GRU 的共享层初始参数也一致。

三个模型都按最低验证 MSE 选权重，未用测试集选模。速度误差按全部窗口及 XY 两分量平均；ATE/RTE 使用同一官方坐标误差定义，逐序列计算后等权平均；60s 对应 12,000 帧。轨迹采用同一 reconstruct()，只以首个参考位置初始化，不进行轨迹旋转拟合。已保存预测重新读取复算指标，两个已完成模型的原指标也逐条复核。

这是单个种子下的共同预算结构比较，不能证明统计显著性、推理效率或模块必然有效。时间成本不等同于参数量；未专门进行推理延迟测试。主要参考 RoNIN ResNet18 可在之前固定协议报告中单独展示。

## 结果目录

- cnn/：本次纯 CNN 的 best/last 权重、配置、日志、训练曲线、35 条 seen 和 32 条 unseen 轨迹图及预测 NPZ。
- comparison/：三模型共同轨迹图、训练曲线、指标总览、逐序列 CSV 和结构变化百分比。
- index.html：可选择序列的交互式总览；result_verification.json、acceptance_checks.json、final_audit.json 为复核记录。
- CNN-GRU 来源：`E:\codex_ws\handoff_cnn_baseline\results\fixed_resnet18_vs_cnn_gru_20261025`。
- Attention 来源：`E:\codex_ws\handoff_cnn_baseline\results\cnn_gru_attention_fixed_20261026`，该目录保留所有注意力权重与权重图，本次不重复训练这两个模型。
