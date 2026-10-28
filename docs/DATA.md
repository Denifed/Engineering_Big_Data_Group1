# 数据、坐标与监督定义

## 正式数据和固定名单

正式数据位于 `RoNIN_Team_Delivery/dataset/`，共152个NPZ。四份名单在 `RoNIN_Team_Delivery/splits/`，训练72、验证13、seen测试35、unseen测试32条序列。名单使用相对数据根目录的路径，迁移时保持原目录层级。

物理 `dataset/train/` 中有85条，其中13条被划为验证集。上游 `quality_report/quality_summary.json` 的85条train统计来自此阶段，不能据此判断实际模型训练使用了85条。

`RoNIN_Team_Delivery/ronin_sequence_001.npz` 是额外移交样本，不属于上述152条正式名单；不能把它当作全量训练集。旧单帧标签实验、smoke和当前窗口均速实验也必须分开。

## NPZ 字段

| 字段 | 形状 | 含义与用途 |
|---|---|---|
| features | `[N,6]` | 角速度XYZ(rad/s)、加速度XYZ(m/s²)，唯一网络输入 |
| targets | `[N,2]` | 上游相邻参考位置差分速度；当前仅用于训练异常筛选 |
| positions | `[N,3]` | 参考位置(m)，用于窗口均速标签和轨迹评估 |
| timestamps | `[N]` | 时间戳(s)，用于真实时间差与积分 |
| orientations | `[N,4]` | 辅助四元数，当前模型不将其输入网络 |

数据形状与字节校验见 `manifests/data_manifest.csv`。训练监督协议为 `window_mean_xy_velocity_v1`。

## 上游处理与参考系

上游技术说明描述：使用RoNIN的 `GlobSpeedSequence(..., grv_only=True, interval=1)`；角速度减初始零偏，加速度做零偏/尺度修正，再通过game rotation vector、起始标定和初始Tango姿态映射到统一参考坐标系。加速度仍含重力相关分量。NPZ保存时将各数组截齐。

模型输出使用与参考XY位置一致的水平坐标系，不是手机机身坐标系。真实位置和速度标签来自参考轨迹，不是将含漂移的IMU直接积分后当作标签。这里保留的是上游数据说明，未从原始采集重新校验参考系统精度。

这份项目快照未包含原始HDF5→NPZ转换程序；`08_ronin_dataset.py`是上游加载器。不能仅靠下载原始数据和运行该文件就声称完整重建本快照。

## 窗口与标签

默认采样约200Hz、窗口200帧、步长10帧。输入 `features[start:end]` 不包含end帧，而监督使用起点与end位置之间的位移和真实时间差：

```python
end = start + 200
x = features[start:end].T       # [6,200]
y = (positions[end, :2] - positions[start, :2]) / (timestamps[end] - timestamps[start])
```

因此不能把NPZ的单帧 `targets[end]` 作为本协议标签，也不能将 `positions`、`targets` 或参考姿态拼入输入。网络输出均速`[vx,vy]`，不是绝对位置或下一秒位移。

## 异常筛选与增强

基础三路线仅在训练时排除窗口内存在`targets`速度模长>10m/s的窗口，过滤80个，保留787,920个训练窗口；验证与测试保留完整序列。筛选不修改NPZ。上游全部序列被标记REVIEW，不等于全部损坏，详见原始质量报告。

基础固定协议没有标准化或增强。后续旋转实验另有协议：同步旋转角速度XY、加速度XY及速度标签XY，Z和时间不变，不能只转输入或只转标签。

## 评估口径

- 速度MSE/RMSE/MAE按全部窗口及两个分量计算，长序列贡献更多窗口。
- ATE保留项目/上游的坐标分量RMSE定义，即`sqrt(mean((predicted_positions - reference_positions)**2))`，再对序列等权平均。
- RTE使用60秒相对位移误差，200Hz下对应12,000个原始帧，同样按序列平均。
- 逐时刻欧氏位置误差曲线不是同一定义的ATE；终点误差、累计速度残差和prefix偏差也应分别命名。
- 重建使用真值首位置设置原点，不做轨迹旋转拟合；仅速度积分不能提供绝对位置约束或保证长期漂移有界。
