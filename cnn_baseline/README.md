# CNN 窗口平均速度基线

CNN 活动代码统一放在本目录。支持 CNN1D、CNN-GRU 和 CNN-GRU + 时间注意力，默认仍为纯 CNN1D。

## 数据与监督定义

- 数据：项目根目录 RoNIN_Team_Delivery/dataset。
- 划分：RoNIN_Team_Delivery/splits 中的四份固定名单。
- 输入：features[start:end]，end=start+200；转置为 [6,200]，批量为 [B,6,200]。
- 标签：y=(positions[end,:2]-positions[start,:2])/(timestamps[end]-timestamps[start])，单位 m/s。
- 输出：[B,2]，对应参考坐标系中的窗口平均 vx、vy。
- NPZ 的 targets 只用于训练异常筛选，不再直接作为监督标签。
- 训练过滤：targets[start:end] 中存在速度模长 >10 m/s 的区间则跳过窗口；验证、测试不筛选。
- 时间戳严格递增；分母使用真实时间差，不假定窗口恰好 1 秒。
- 输入不包含参考位置、时间戳或参考姿态。
- 本入口不做标准化、随机旋转或时间抖动。与 ResNet 共享标签定义，不代表训练配置完全相同。

## 文件

| 文件 | 用途 |
|---|---|
| dataset.py | 统一加载器、标签协议及权重协议检查 |
| model_cnn1d.py | 三层 CNN、全局平均池化、二维速度回归 |
| model_cnn_gru.py | CNN-GRU 可选对照 |
| model_cnn_gru_attention.py | CNN-GRU + 加性时间注意力 |
| common.py | 模型注册、设备选择、训练曲线 |
| train.py | 训练、验证、早停、配置/快照/权重保存、重载校验 |
| evaluate.py | 速度指标、逐窗口预测、逐序列速度图、指标图 |
| compare_with_resnet.py | 对完成 30 轮的纯 CNN 复用 ResNet 轨迹协议，生成逐序列三方对比、ATE/RTE、可视化总览 |
| run_fixed_comparison.py | 固定共同训练协议，从头运行官方 ResNet18 与 CNN-GRU，并生成共同轨迹评估 |
| train_attention_fixed.py | 只训练 Attention 模型，复用已验收基线的协议、数据与逐轮窗口顺序 |
| evaluate_attention_fixed.py | 固定协议 Attention 全量测试、三模型轨迹对比、权重图及保存结果复核 |
| train_cnn_fixed.py | 纯 CNN 按已验收基线的固定协议从头训练，逐轮核对样本顺序 |
| evaluate_cnn_three_routes.py | CNN、CNN-GRU、CNN-GRU-Attention 共同速度/轨迹评估及可视化 |
| sequence_dataset.py | 同序列连续窗口片段，不跨过滤缺口，记录片段覆盖和时间间隔 |
| losses.py | 窗口速度 MSE 与归一化累计残差约束 |
| train_drift_comparison.py | 相同 CNN-GRU、连续片段和验证 ATE 选模，训练 lambda=0/1 配套对照 |
| evaluate_drift_comparison.py | 配套测试、持续偏差/终点误差统计、六面板轨迹图和保存结果复算 |
| test_drift.py | 连续性、时间间隔、残差损失、梯度与确定性 CUDA 检查 |
| test_pipeline.py | 标签、过滤、旧权重拒绝、真实数据一致性、端到端测试 |
| runs/ | 默认输出，每次创建新目录 |

## 运行

在项目根目录使用 PowerShell：

    & 'E:\conda_envs\torch_env\python.exe' -B .\cnn_baseline\train.py --model cnn --epochs 30 --patience 6 --out .\cnn_baseline\runs\cnn_mean_run01

默认自动使用 CUDA，batch_size=256，window_size=200，step_size=10，Adam lr=1e-4，MSE，seed=42。可用 --device cpu 切换。不指定 --out 则自动创建时间戳目录；指定目录已存在时拒绝覆盖。

需要完整跑满指定轮数时添加 --no-early-stopping；仍保存验证集最优权重和最后一轮权重。

训练后评估：

    & 'E:\conda_envs\torch_env\python.exe' -B .\cnn_baseline\evaluate.py --checkpoint .\cnn_baseline\runs\cnn_mean_run01\checkpoint_best.pt

评估默认输出到训练目录的 evaluation_时间戳 子目录。--out 可指定新目录。--model cnn_gru 可训练对照；evaluate.py 的 --checkpoint 接受多个不同模型的新权重做共同比较。

也支持 python -m cnn_baseline.train / python -m cnn_baseline.evaluate。

## 固定协议的 ResNet18 与 CNN-GRU 对照

    & 'E:\conda_envs\torch_env\python.exe' -B -u .\cnn_baseline\run_fixed_comparison.py --out .\results\fixed_resnet18_vs_cnn_gru_20261025

两者从头训练 30 轮，window=200、step=10、batch=256、Adam lr=1e-4、MSE、seed=42；不使用增强、标准化、学习率调度和早停。FP32，TF32 关闭，确定性算法开启。训练按同一窗口索引，每轮使用独立于模型随机状态的相同排列，日志保存排列 SHA256 并逐轮核对。

官方 ResNet18 与 CNN-GRU 的网络定义保持不变（包括 ResNet 原有 FC Dropout）。它们共用 GPU 数据缓存、训练循环、验证选模规则和轨迹重建/评估函数。比较相同样本访问和更新次数，不保证相同参数量或计算量。

输出目录包含 resnet18/、cnn_gru/ 和 comparison/，各模型保留独立权重、日志、seen_test/、unseen_test/；comparison/ 为共同图表，根目录 index.html 为总览。protocol.json 和 data_order_manifest.json 保存一致性证据。

--smoke 可做两轮小规模流程检查，必须指定另一个新目录；小规模结果不作为正式实验。

## CNN-GRU + Attention

与已完成的固定参数基线进行消融时，使用以下入口。先只读检查配置（不训练、不写输出）：

    & 'E:\conda_envs\torch_env\python.exe' -B .\cnn_baseline\train_attention_fixed.py --check-only

明确启动完整训练时运行：

    & 'E:\conda_envs\torch_env\python.exe' -B -u .\cnn_baseline\train_attention_fixed.py --out .\results\cnn_gru_attention_fixed_20261026

默认引用 results/fixed_resnet18_vs_cnn_gru_20261025，可用 --reference-run 指定另一套已经验收的同协议基线。入口会核对完整基线状态、共同协议 SHA256、数据文件、划分、训练/验证窗口索引和标签摘要，再逐轮核对样本排列。不重跑两个基线，不从旧权重微调。配置核对不代表已执行数据全量校验；数据校验在真正训练前完成。

训练保持 30 轮、batch 256、Adam lr=1e-4、MSE、seed=42，无增强/标准化/调度/早停，数值设置与固定基线相同。输出中的 cnn_gru_attention/ 保存模型权重、日志和曲线；此入口只负责训练。测试速度和注意力可用共同评估入口：

    & 'E:\conda_envs\torch_env\python.exe' -B .\cnn_baseline\evaluate.py --checkpoint .\results\cnn_gru_attention_fixed_20261026\cnn_gru_attention\checkpoint_best.pt --save-attention

上述通用评估保存速度指标、速度曲线和注意力，不计算 ATE/RTE。固定协议正式实验训练完成后，使用以下入口一次性完成速度、ATE/RTE、三模型轨迹对比及注意力权重导出：

    & 'E:\conda_envs\torch_env\python.exe' -B -u .\cnn_baseline\evaluate_attention_fixed.py --run .\results\cnn_gru_attention_fixed_20261026

固定评估入口引用 reference.json 中的基线结果，核对共同评估代码、数据指纹、完整 30 轮顺序与最佳验证权重。逐序列核对基线预测的标签、时间、位置与当前数据一致，复算基线指标；Attention 预测保存后重新读取复核。它不修改原基线结果。输出在本次训练目录，已有评估目录时拒绝覆盖。

cnn_gru_attention/seen_test 与 unseen_test 各保存轨迹/速度/误差图、attention 图和逐窗口预测 NPZ（含 [N,50] 权重）；comparison/ 保存三模型对比、训练曲线和指标，index.html 为交互式总览。统一使用固定基线的 reconstruct() 和官方 ATE/60s RTE 定义。单个种子下的结果只表示该次实验，不能独立证明统计显著性。

默认结构：[B,6,200] → CNN [B,128,50] → 转置 [B,50,128] → 单向 GRU [B,50,64] → 时间注意力 [B,64] → Linear(64,64)、ReLU、Linear(64,2)。CNN、GRU、回归头与 CNN-GRU 对照相同，变化是用时间注意力汇总全部 GRU 输出，替代直接取最终隐藏状态。

注意力分数为 score_t = v^T tanh(W h_t + b)，softmax 沿时间维归一化，context = sum(alpha_t * h_t)。attention_size=32。50 个权重对应卷积下采样后的时间步，不是原始 200 帧。当前固定长度窗口不使用 padding 或 mask。

forward(x) 默认只返回 [B,2]，兼容共同训练/评估接口；forward(x, return_attention=True) 或 forward_with_attention(x) 返回 (prediction, weights)，默认 weights 为 [B,50]。model_config() 提供可序列化的结构信息，训练权重同时记录它。

模型调用示例：

    import torch
    from cnn_baseline.model_cnn_gru_attention import CNNGRUAttention
    model = CNNGRUAttention()
    x = torch.randn(4, 6, 200)
    velocity, weights = model(x, return_attention=True)
    assert velocity.shape == (4, 2)
    assert weights.shape == (4, 50)

默认参数量 80,482，较 CNN-GRU 增加 2,112 个参数。二者在相同初始化种子下的 CNN、GRU 和回归头初始权重一致；注意力替代最终隐藏状态汇总。代码与流程测试不代表已完成 Attention 的正式训练，不能据此宣称精度提升。

--save-attention 可选：逐窗口权重保存在 window_predictions.npz 的 attention_weights 字段，每条序列另保存中间窗口的注意力柱状图。全量测试权重未压缩时约 150 MB（200 帧窗口），不需要时省略该选项。权重展示仅供诊断，不能单独作为因果解释或性能提升证据。

三个模型共享标签、筛选、优化器和早停规则。训练样本随机排序使用独立固定种子的 generator，不受不同模型初始化消耗随机数的影响。保持同样训练预算和多种子实验后，才可判断 Attention 效果。

## 输出与验收

纯 CNN 补跑并与 CNN-GRU、CNN-GRU-Attention 做固定协议对比时，使用：

    & 'E:\conda_envs\torch_env\python.exe' -B .\cnn_baseline\train_cnn_fixed.py --check-only
    & 'E:\conda_envs\torch_env\python.exe' -B -u .\cnn_baseline\train_cnn_fixed.py --out .\results\cnn_three_routes_fixed_20261026
    & 'E:\conda_envs\torch_env\python.exe' -B -u .\cnn_baseline\evaluate_cnn_three_routes.py --run .\results\cnn_three_routes_fixed_20261026

本入口保留 CNN1D 三层卷积和全局平均池化结构，复用固定协议的 GPU 窗口缓存和 train_one()。完整训练 30 轮，逐轮核对与既有 CNN-GRU 的相同样本排列。评估默认引用 results/fixed_resnet18_vs_cnn_gru_20261025 中的 CNN-GRU，以及 results/cnn_gru_attention_fixed_20261026 中的 Attention 结果；后者可用 --attention-run 指定。两个参考模型不重新训练，已有目录不改写。

三路线结果位于新目录的 cnn/、comparison/ 和 index.html。CNN→CNN-GRU 同时改变全局平均池化/GRU 汇总方式、参数量和回归头输入维度，应称为结构对照；CNN-GRU→Attention 才是在相同骨干与回归头下改变时间汇总方式的直接消融。不要把旧纯 CNN 实验的数字混入这套固定协议对比。

训练保存 config.json、dataset_manifest.json、source_snapshot/、run.log、training_log.csv、training_curves.png、checkpoint_best.pt、checkpoint_last.pt、summary.json。结束后严格重载 best 权重并验证推理一致性。

评估保存 test_metrics.csv、sequence_metrics.csv、test_comparison.png、config.json、summary.json；按测试集/模型保存 window_predictions.npz 和逐序列速度图。

MSE 对全部窗口及两个坐标分量取平均，单位 (m/s)^2；RMSE、MAE 单位 m/s。长序列贡献更多窗口。本评估入口不计算轨迹 ATE/RTE。

另用 compare_with_resnet.py 可按已有 ResNet 实验协议生成轨迹对比，结果直接写入 CNN 训练目录的 seen_test/、unseen_test/。两个目录必须独立，已有 ResNet 结果不会被覆盖。例如：

    & 'E:\conda_envs\torch_env\python.exe' -B .\cnn_baseline\compare_with_resnet.py --cnn-run .\results\ronin_resnet18_npz_20261025\cnn_mean_30epochs_20261025 --resnet-run .\results\ronin_resnet18_npz_20261025 --device cuda

该入口核对两模型的窗口索引、时间、标签、参考轨迹，并从保存的预测复算速度和轨迹指标。CNN 与当前 ResNet 的增强、容量、实际轮次不同，对比不能解释为严格控制变量的结构消融。

checkpoint 记录 label_protocol=window_mean_xy_velocity_v1。评估拒绝缺少协议的旧权重，即使参数形状相同也不接受。

## 连续窗口累计误差约束实验

固定 CNN-GRU 网络，每片段 64 个相邻窗口，每 batch 4 个片段，原始窗口仍为 200 帧、步长 10 帧。对预测减去同口径窗口平均速度标签后的残差，在前 16/32/64 个窗口上计算 sum(error*dt)/sum(dt)，平方后平均作为 bias penalty。dt 为相邻预测间隔，约 0.05 秒。它是持续偏差正则，不是精确真值位移监督，也不对真实速度做平滑。

新协议从头训练两个条件：mse_control (lambda=0) 与 bias_constraint (lambda=1)，各 30 轮，seed=42、Adam 1e-4；初始化、连续片段、逐轮排列、更新次数和选模规则相同。完整片段尾部不足 64 的部分统一排除；验证和测试不排除。两组都按完整验证集最低宏平均 ATE 选权重，测试不参与选模或权重选择。

    & 'E:\conda_envs\torch_env\python.exe' -B -m unittest cnn_baseline.test_drift -v
    & 'E:\conda_envs\torch_env\python.exe' -B -u .\cnn_baseline\train_drift_comparison.py --out .\results\drift_loss_cnn_gru_20261026
    & 'E:\conda_envs\torch_env\python.exe' -B -u .\cnn_baseline\evaluate_drift_comparison.py --run .\results\drift_loss_cnn_gru_20261026

两个入口都支持 --smoke，训练时必须使用独立新目录；小规模目录不能作为正式实验。第一次 smoke 曾发现 NumPy 标量与安全 checkpoint 读取不兼容，修正后使用 smoke_v2 目录完成训练、重载和评估检查，失败目录保留为诊断记录。

正式结果目录包含两条件 best/last 权重、逐轮日志、最佳及最后一轮的验证逐序列明细、训练曲线、seen_test/、unseen_test/；comparison/ 保存共同六面板图（轨迹、vx/vy、位置误差、累计窗口速度残差、前缀平均偏差）、汇总 CSV/PNG 和变化百分比。index.html 为总览。预测 NPZ 包含预测间隔和累计残差诊断量；所有保存指标独立重新读取复算。

历史随机窗口 CNN-GRU 使用验证 MSE 选模，与本次采样和验证 ATE 选模不同，只能作为背景。本次新增约束的直接结论必须来自 mse_control 与 bias_constraint 的配套比较。两者参数量均为 78,370，推理输入和网络没有改变。单个种子且测试数据曾用于提出改进假设，不能宣称新测试集统计显著性、首创或保证降漂移。

## 历史结果与测试说明

results/cnn_comparison 是旧单帧速度协议的历史结果，已停用。不能与窗口平均速度结果直接比较。修改标签后须从头训练。

    & 'E:\conda_envs\torch_env\python.exe' -B -m unittest cnn_baseline.test_pipeline -v

临时合成数据仅验证代码流程，不是课程实验结果。真实 NPZ 测试与 A 旧加载器逐元素比较。重新生成代码不代表完成新 CNN 全量训练。

