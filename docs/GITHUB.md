# 仓库与发布范围

项目仓库：https://github.com/Denifed/Engineering_Big_Data_Group1

## 已确定的发布方式

- 项目代码按GPL-3.0-only发布，保留RoNIN原始许可证、版权和来源说明。
- 发布活动代码、固定数据划分、数据结构/指纹清单、聚合指标及汇总图表。
- 不发布数据NPZ、原始/参考轨迹数组、模型权重或完整逐窗口预测。
- 完整实验档案保留在本地，普通Git提交只包含轻量内容。

许可证全文见根目录LICENSE。公开展示代码仍属于发布，不能附加“禁止商用”或“仅供展示不可使用”的限制来替代GPL。数据与权重若以后需要公开，应另行核对其适用条款。

## 获取与检查

```bash
git clone https://github.com/Denifed/Engineering_Big_Data_Group1.git
cd Engineering_Big_Data_Group1
python -B tools/check_package.py
```

无需数据即可阅读README、reports和运行基础目录/代码语法检查。训练和完整评估需要按docs/DATA.md恢复数据；高级实验还依赖本地完整历史参考结果，详见docs/REPRODUCE.md。

`.gitignore`持续排除results、数据、权重及预测。`.gitattributes`关闭换行自动转换，以保留已有源码/划分SHA256。不要使用git add -f强制加入被忽略的大文件。

`manifests/github_files.csv`记录本次轻量发布的文件及SHA256（清单自身除外）；`manifests/copied_files.csv`和packaging_summary.json则记录本地完整档案的来源，不代表那些归档文件已经上传。
