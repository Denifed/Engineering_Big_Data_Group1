# 实验报告与图表

这些文件从对应`results/<实验名>/`复制，保留原始CSV数值、JSON配置/验收记录、报告和汇总PNG。逐轨迹预测NPZ、权重、完整轨迹图库与HTML留在本地`results/`，不重复提交到普通Git。

| 研究内容 | 轻量报告目录 | 主要入口 |
|---|---|---|
| 固定协议ResNet18 / CNN-GRU | [fixed_resnet18_vs_cnn_gru_20261025](fixed_resnet18_vs_cnn_gru_20261025/) | `comparison/test_metrics.csv` |
| CNN-GRU-Attention | [cnn_gru_attention_fixed_20261026](cnn_gru_attention_fixed_20261026/) | `comparison/test_metrics.csv` |
| CNN / CNN-GRU / Attention三路线 | [cnn_three_routes_fixed_20261026](cnn_three_routes_fixed_20261026/) | `comparison/test_metrics.csv` |
| 连续片段累计偏差约束 | [drift_loss_cnn_gru_20261026](drift_loss_cnn_gru_20261026/) | `README_实验结果.md`、`comparison/` |
| BatchNorm统计诊断 | [bn_calibration_diagnosis_20261026](bn_calibration_diagnosis_20261026/) | `README_诊断结果.md` |
| 冻结CNN-GRU的C4推理 | [cnn_gru_rotation_consistency_20261026](cnn_gru_rotation_consistency_20261026/) | `README_实验结果.md`、`comparison/test_metrics.csv` |
| 旋转增强 / 一致性训练 | [cnn_gru_yaw_equivariance_20261027](cnn_gru_yaw_equivariance_20261027/) | `README_实验结果.md`、`comparison/test_metrics.csv` |
| 最早的ResNet NPZ适配与旧CNN对照 | [ronin_resnet18_npz_20261025](ronin_resnet18_npz_20261025/) | 历史记录，不能代替固定协议三路线 |

报告中的旧机器绝对路径是历史溯源信息。`source_snapshot`与完整日志在本地档案中保留。活动代码和模块README经历过开发更新，当前总览以根README及本目录正式报告为准。
