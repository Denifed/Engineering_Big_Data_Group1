# CNN 30 轮训练与 ResNet 轨迹对比

CNN 完整训练 30 轮，关闭早停。使用验证集选出的第 30 轮权重评估，验证 MSE 为 0.09222355；末轮权重另存 checkpoint_last.pt。

| 测试集 | 模型 | 速度 RMSE (m/s) | MAE (m/s) | 官方 ATE (m) | 官方 60s RTE (m) |
|---|---|---:|---:|---:|---:|
| seen_test | CNN | 0.215726 | 0.143041 | 7.618 | 4.433 |
| seen_test | RoNIN ResNet18 | 0.173270 | 0.113597 | 5.596 | 3.560 |
| unseen_test | CNN | 0.296093 | 0.195391 | 9.425 | 6.427 |
| unseen_test | RoNIN ResNet18 | 0.243038 | 0.156749 | 7.420 | 5.269 |

输入 [B,6,200]，标签为窗口两端 XY 位移除以真实时间差，步长 10。训练 72 条、验证 13 条、seen 35 条、unseen 32 条。训练窗口过滤极端速度区间，测试不筛选。

轨迹重建复用 ResNet 的 reconstruct 函数；官方 ATE/RTE 为坐标平方误差平均后开方，先计算每条序列再等权平均，不额外旋转对齐。速度指标按窗口及两个分量平均。

本次 CNN 参数 45,218，Adam lr=1e-4、batch=256、seed=42，无增强；ResNet 参数 4,634,882，包含水平旋转/时间偏移增强，实际训练 13 轮、最佳第 7 轮。共同标签和评估已核对，但训练方案及网络容量不同，不是严格的结构消融。

训练日志中耗时超过 300 秒的轮次：[26]。运行期间出现长时间日志间隔，恢复后继续训练；记录的墙钟耗时包含停顿，不宜据此比较模型训练效率。

seen_test/ 和 unseen_test/ 每条序列保存四面板图：参考/CNN/ResNet 轨迹、vx、vy、位置误差，以及 CNN 预测 NPZ。index.html 提供全部序列浏览；cnn_vs_resnet_summary.png 和 ate_distribution.png 为汇总图。

test_metrics.csv 为汇总，sequence_metrics.csv 为 CNN 逐序列指标，comparison_sequence_metrics.csv 包含两模型逐序列指标。result_verification.json 记录全部预测落盘后的指标复算及与 ResNet 的标签、索引、时间、参考轨迹一致性检查。

训练命令：使用 torch_env 的 Python 运行 cnn_baseline/train.py --model cnn --epochs 30 --no-early-stopping --device cuda --batch-size 256 --out 本目录。
评估命令：使用同一 Python 运行 cnn_baseline/compare_with_resnet.py --cnn-run 本目录 --resnet-run 上级ResNet目录 --device cuda。
