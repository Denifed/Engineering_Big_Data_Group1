# 环境记录

`recorded_training_environment.json`原样保留旋转增强正式实验的环境记录。`requirements-recorded.txt`是整理时现有环境的核心包精确版本，包括Pillow图像分析依赖。

根目录requirements覆盖活动NPZ训练/评估及结果分析需要的包；PyTorch采用功能最低版本，不自动指定CUDA wheel。精确历史构建为2.14.0+cu126、Python3.11.16、CUDA12.6、RTX4060 Laptop。其他平台/版本能否逐位复现尚未验证。

`ronin-master/requirements.txt`覆盖上游完整HDF5/姿态等工具，依赖多于当前NPZ主线。不要为了运行NPZ实验自动安装全部上游工具，也不要将已有Conda目录复制或上传到仓库。
