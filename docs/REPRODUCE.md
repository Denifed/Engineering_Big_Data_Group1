# 运行与复现

所有命令都在本项目根目录执行，`python`指已激活环境中的解释器。Windows也可使用自己机器上的Conda Python绝对路径，不必沿用原机器的`E:\conda_envs\torch_env`。

## 1. 准备环境与数据

```bash
python -m pip install -r requirements.txt
python -B tools/check_package.py --data --models
```

完整本地包已带数据。GitHub克隆需要先恢复152条NPZ到`RoNIN_Team_Delivery/dataset/`；路径、大小、SHA256在`manifests/data_manifest.csv`。没有附加数据时，普通检查可运行，`--data`会明确报缺失。

`--models`只在CPU上检查三个网络的输入输出形状，不训练模型，不计算性能。`--hash`可选逐文件检查完整复制档案，读取量约7GiB，不是数据重新预处理。

固定协议训练和部分正式评估直接使用CUDA，不能因通用训练支持CPU就认为所有实验入口支持CPU。实际训练版本见`environment/`；迁移平台、GPU或PyTorch后，数值逐位一致性尚未验证。

## 2. 独立训练接口

```bash
python -B cnn_baseline/train.py --model cnn --epochs 30 --no-early-stopping --out results/my_cnn
python -B cnn_baseline/train.py --model cnn_gru --epochs 30 --no-early-stopping --out results/my_gru
python -B cnn_baseline/train.py --model cnn_gru_attention --epochs 30 --no-early-stopping --out results/my_attention
```

通用速度评估：

```bash
python -B cnn_baseline/evaluate.py --checkpoint results/my_gru/checkpoint_best.pt --out results/my_gru_velocity_eval
```

此通用入口不输出ATE/RTE；它也不执行归档固定对照所用的跨模型逐轮顺序哈希验收。需要严谨复核历史表格时，使用下一节的完整流程。

## 3. 固定协议三路线：按依赖顺序新建实验

以下输出名均区别于历史档案，且首次执行前不能已经存在。完整数据必须恢复。

### 3.1 先训练共同参考：ResNet18与CNN-GRU

```bash
python -B -u cnn_baseline/run_fixed_comparison.py --out results/reproduce_fixed
```

这一步从头完成两个模型的30轮训练及共同评估，保存协议、数据指纹、样本顺序、权重、日志与指标。原始协议为窗口200/步长10、batch256、Adam1e-4、MSE、seed42、FP32，不早停、不调度、不增强，按验证MSE选权重。

### 3.2 训练并评估Attention

```bash
python -B cnn_baseline/train_attention_fixed.py --reference-run results/reproduce_fixed --check-only
python -B -u cnn_baseline/train_attention_fixed.py --reference-run results/reproduce_fixed --out results/reproduce_attention
python -B -u cnn_baseline/evaluate_attention_fixed.py --run results/reproduce_attention
```

`--check-only`只检查配置与参考记录，不代表完成训练或重新检查全部数据。

### 3.3 补训练纯CNN，生成三路线对比

```bash
python -B cnn_baseline/train_cnn_fixed.py --reference-run results/reproduce_fixed --check-only
python -B -u cnn_baseline/train_cnn_fixed.py --reference-run results/reproduce_fixed --out results/reproduce_cnn
python -B -u cnn_baseline/evaluate_cnn_three_routes.py --run results/reproduce_cnn --attention-run results/reproduce_attention
```

这套流程的新`reference.json`会指向新参考目录。不要把历史记录的绝对路径手工批量替换掉：历史文件保存了验收与来源证据，随意修改可能使原始哈希失效。

## 4. 后续旋转实验

现有高级入口保留了固定历史参考目录约定，不能仅修改`--out`就假定所有参考路径都改变。使用完整本地包时参考档案已存在；仅有GitHub代码时，应另外恢复相应权重、指纹和预测结果。

| 入口 | 主要固定依赖 |
|---|---|
| `train_drift_comparison.py` | `results/fixed_resnet18_vs_cnn_gru_20261025/` |
| `diagnose_bn_calibration.py` | 上述固定对照与`results/drift_loss_cnn_gru_20261026/` |
| `evaluate_rotation_consistency.py` | 固定CNN-GRU与`results/bn_calibration_diagnosis_20261026/`中的BN校准权重 |
| `train_rotation_consistency.py` | 固定对照数据指纹、漂移实验dataset_manifest、历史C4档案 |
| `evaluate_yaw_training.py` / `analyze_yaw_training.py` | 新训练目录，以及历史固定CNN/GRU/Attention/ResNet与C4结果 |

### C4冻结推理

```bash
python -B -u cnn_baseline/evaluate_rotation_consistency.py --out results/reproduce_c4
python -B cnn_baseline/analyze_rotation_consistency.py --run results/reproduce_c4
```

原始/混合校准/片段校准三种BN状态，各做单次与C4，共六条件。C4对0/90/180/270度输入预测逆旋转后等权平均，不训练、不在线更新。

### 旋转增强训练与一致性约束

```bash
python -B -u cnn_baseline/train_rotation_consistency.py --out results/reproduce_yaw
python -B -u cnn_baseline/evaluate_yaw_training.py --run results/reproduce_yaw
python -B cnn_baseline/analyze_yaw_training.py --run results/reproduce_yaw
```

A为双分支原始/旋转样本监督，B额外加lambda=0.1的一致性MSE；各30轮，主权重按未增强验证MSE选取。主比较B对A才隔离一致性项，A/B对旧模型还包含增强和网络batch组织变化。C4与单次必须分别报告。

中断恢复仅对提供`--resume`的入口、同一目录和未改变源码/协议使用。现有脚本会核对哈希，不应绕过验收继续训练。

## 5. 查看已有结果与重用权重

无需运行训练即可阅读`reports/`。完整本地档案中的`results/<实验>/index.html`可直接用浏览器打开，查看所有序列图。GitHub的普通文件页不执行这些HTML。

已归档三模型权重可通过通用入口做新的速度评估：

```bash
python -B cnn_baseline/evaluate.py --checkpoint results/cnn_three_routes_fixed_20261026/cnn/checkpoint_best.pt results/fixed_resnet18_vs_cnn_gru_20261025/cnn_gru/checkpoint_best.pt results/cnn_gru_attention_fixed_20261026/cnn_gru_attention/checkpoint_best.pt --out results/restored_weights_velocity_eval
```

该命令需要完整数据和权重，输出使用新目录，仍不计算ATE/RTE。现有历史轨迹评估目录不能原地重复写入；归档`reference.json`包含旧路径，而`--run`不一定有参考路径覆盖参数。不要将该归档直接视为可随处原地重跑的配置。

## 6. 保存与复核

- `checkpoint_best.pt`是验证选出的主模型，`checkpoint_last.pt`为最后一轮；两者不能混用。旋转训练还有最佳ATE诊断权重，不用于主表选模。
- 保存预测NPZ后可独立复算指标；已保存验收记录只证明历史运行，不等于新机器已完成训练。
- `source_snapshot/`保留当时源码。活动代码后续添加过功能时，应保留快照与实验上下文，而不是声称所有历史运行都使用当前目录全部文件。
- 旧单帧标签checkpoint缺少新协议，应拒绝使用，不应仅凭张量形状相同强行载入。
