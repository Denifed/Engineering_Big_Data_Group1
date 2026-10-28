# 第三方来源与授权记录

## RoNIN

- 本地源码：`ronin-master/`，保留移交文件及原始 `README.md`、`LICENSE`。
- 原始许可证：`ronin-master/LICENSE`，GNU General Public License Version 3。
- 实际复用：`source/model_resnet1d.py` 的 ResNet18 定义，`source/metric.py` 的轨迹指标；NPZ 训练/评估适配位于 `scripts/` 和 `cnn_baseline/`。
- 本项目结果采用自有 NPZ、窗口平均速度监督及固定划分，不应标记为官方论文数字或完整官方实验条件复现。

## 数据

`RoNIN_Team_Delivery/` 是上游移交的预处理数据及说明。技术说明将原始参考位置描述为 Tango 轨迹，特征来自经校准与参考系变换的 IMU。原始数据下载入口见上游 RoNIN README。

本快照没有证明处理后 NPZ、参考位置或训练权重可被公开再分发。它们保留在本地完整目录，默认被 `.gitignore` 排除；公开附件前需由团队确认授权。上游源码许可证不能代替数据使用条款。

## 本项目新增代码

新增项目代码按 GNU General Public License Version 3 发布（SPDX: GPL-3.0-only），条款见根目录 LICENSE。各贡献者保留各自版权，RoNIN 部分的版权与许可证不变。本项目不以 MIT/BSD/Apache 替换上游 GPL，也不添加“禁止商用”“仅供展示不可使用”等与 GPL 冲突的代码限制。数据、权重不在此次发布范围内。
