# 实验索引

原始目录名与内部结构保留，用于满足代码中的相对路径依赖。`results/`是本地完整档案，默认不进入普通Git；正式报告摘录在`reports/`。

| 目录 | 类别 | 历史记录状态 | 权重文件 | 预测NPZ |
|---|---|---|---:|---:|
| [bn_calibration_diagnosis_20261026](../reports/bn_calibration_diagnosis_20261026/) | diagnosis | summary=completed; final_audit=passed | 6 | 10 |
| `results/cnn_comparison` | legacy_single_frame_labels | 无正式验收结论 | 4 | 0 |
| [cnn_gru_attention_fixed_20261026](../reports/cnn_gru_attention_fixed_20261026/) | fixed_model_comparison | summary=completed; acceptance_checks=passed; final_audit=passed | 2 | 67 |
| [cnn_gru_rotation_consistency_20261026](../reports/cnn_gru_rotation_consistency_20261026/) | frozen_C4_inference | summary=completed; final_audit=passed | 0 | 720 |
| [cnn_gru_yaw_equivariance_20261027](../reports/cnn_gru_yaw_equivariance_20261027/) | paired_yaw_training | summary=completed; training_acceptance=passed; final_audit=passed | 6 | 402 |
| `results/cnn_gru_yaw_equivariance_smoke_20261027` | smoke_only | summary=training_completed; training_acceptance=passed | 6 | 0 |
| `results/cnn_gru_yaw_equivariance_smoke_v2_20261027` | smoke_only | summary=smoke_completed; training_acceptance=passed; final_audit=passed | 6 | 12 |
| [cnn_three_routes_fixed_20261026](../reports/cnn_three_routes_fixed_20261026/) | fixed_model_comparison | summary=completed; acceptance_checks=passed; final_audit=passed | 2 | 67 |
| [drift_loss_cnn_gru_20261026](../reports/drift_loss_cnn_gru_20261026/) | exploratory_drift_loss | summary=completed; acceptance_checks=passed; final_audit=passed | 4 | 134 |
| `results/drift_loss_cnn_gru_smoke_20261026` | smoke_only | 无正式验收结论 | 2 | 0 |
| `results/drift_loss_cnn_gru_smoke_v2_20261026` | smoke_only | summary=smoke_evaluation_completed; acceptance_checks=passed | 4 | 4 |
| [fixed_resnet18_vs_cnn_gru_20261025](../reports/fixed_resnet18_vs_cnn_gru_20261025/) | fixed_model_comparison | summary=completed; acceptance_checks=passed; final_audit=passed | 4 | 134 |
| `results/fixed_resnet18_vs_cnn_gru_smoke_20261025` | smoke_only | summary=smoke_passed; acceptance_checks=passed | 4 | 4 |
| [ronin_resnet18_npz_20261025](../reports/ronin_resnet18_npz_20261025/) | historical_npz_adapter | summary=completed; acceptance_checks=record_present | 4 | 134 |
| `results/ronin_resnet18_npz_smoke_20261025` | smoke_only | summary=smoke_passed | 2 | 2 |

## 主线及依赖

基础对照：fixed ResNet18/CNN-GRU → Attention → 纯CNN三路线共同评估。三路线中的GRU/Attention预测复用对应已验收的参考运行，并非每次对比都重新训练所有模型。

诊断与改进：连续片段偏差约束 → BN统计诊断 → 冻结C4推理 → 旋转增强/一致性训练及单次/C4评估。不同采样、选模、增强和推理次数应分别说明，不能混成一个严格控制变量对照。

`cnn_comparison`使用旧单帧标签，只保留历史用途。含smoke的目录都是小规模流程或故障检查，不能作为正式性能结果。`ronin_resnet18_npz_20261025`包含早期适配和旧CNN设置，不能代替后来的固定协议表。

## 核心文件

- 数据加载与标签：`cnn_baseline/dataset.py`。

- 模型：`model_cnn1d.py`、`model_cnn_gru.py`、`model_cnn_gru_attention.py`。

- 固定循环与顺序核对：`run_fixed_comparison.py`；CNN/Attention入口复用其协议与训练函数。

- 轨迹重建：`scripts/run_ronin_npz_baseline.py`；指标及汇总见`compare_with_resnet.py`、上游`metric.py`与各共同评估脚本。

- C4：`evaluate_rotation_consistency.py`；配对增强与一致性损失：`rotation_training.py`、`train_rotation_consistency.py`。

- `source_snapshot/`记录当时源码；summary、验收JSON与保存预测记录历史证据。此处没有重新跑全量训练或测试。
