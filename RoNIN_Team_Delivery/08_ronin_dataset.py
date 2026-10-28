
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RoNINDataset(Dataset):
    def __init__(
            self,
            data_root,
            list_path,
            window_size=200,
            step_size=10,
            layout="gru",
            filter_outliers=False,
            outlier_threshold=10.0
    ):
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.step_size = step_size
        self.layout = layout
        self.filter_outliers = filter_outliers
        self.outlier_threshold = outlier_threshold
        self.filtered_count = 0

        if layout not in ("gru", "cnn"):
            raise ValueError("layout 必须是 gru 或 cnn")

        # 从文件读取序列名单
        with open(list_path, "r", encoding="utf-8") as f:
            names = [s.strip() for s in f if s.strip()]

        self.files = [self.data_root / name for name in names]
        self.index_map = []

        # 只读取每条序列的长度以建立索引
        for seq_id, file_path in enumerate(self.files):
            if not file_path.exists():
                raise FileNotFoundError(file_path)

            with np.load(file_path) as data:
                n = len(data["timestamps"])
                targets = data["targets"]

            # 标记极端瞬时速度
            speed = np.linalg.norm(targets, axis=1)
            bad = speed > self.outlier_threshold

            # 构建前缀和，以快速判断窗口是否包含异常帧
            bad_prefix = np.concatenate([
                [0],
                np.cumsum(bad.astype(np.int64))
            ])

            for start in range(0, n - window_size, step_size):
                end = start + window_size

                # targets[k] 对应位置 k -> k+1
                # 因此需要检查 [start, end) 之间的位移区间
                if self.filter_outliers:
                    bad_count = bad_prefix[end] - bad_prefix[start]

                    if bad_count > 0:
                        self.filtered_count += 1
                        continue

                self.index_map.append((seq_id, start))

        print(
            f"序列数: {len(self.files)}, "
            f"窗口样本数: {len(self.index_map)}"
        )

    @lru_cache(maxsize=2)
    def _load_sequence(self, seq_id):
        with np.load(self.files[seq_id]) as data:
            features = data["features"].astype(np.float32)
            positions = data["positions"].astype(np.float64)
            timestamps = data["timestamps"].astype(np.float64)

        return features, positions, timestamps

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        seq_id, start = self.index_map[index]

        features, positions, timestamps = (
            self._load_sequence(seq_id)
        )

        end = start + self.window_size

        # IMU 输入：200帧 × 6通道
        x = features[start:end]

        # 整个窗口内的平均水平速度
        dt = timestamps[end] - timestamps[start]
        if dt <= 0:
            raise ValueError("时间间隔异常")

        y = (
            positions[end, :2] - positions[start, :2]
        ) / dt

        x = torch.from_numpy(x.copy())
        y = torch.tensor(y, dtype=torch.float32)

        # ResNet1D使用(batch, 6, 200)
        # GRU使用(batch, 200, 6)
        if self.layout == "cnn":
            x = x.transpose(0, 1)

        return x, y


if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parent

    DATA_ROOT = PROJECT_DIR / "processed" / "dataset"
    SPLIT_ROOT = PROJECT_DIR / "processed" / "splits"

    train_set = RoNINDataset(
        DATA_ROOT,
        SPLIT_ROOT / "train.txt",
        window_size=200,
        step_size=10,
        layout="gru",
        filter_outliers=True,
        outlier_threshold=10.0
    )

    print("过滤窗口数量：", train_set.filtered_count)
    print("过滤后训练样本数量：", len(train_set))

    val_set = RoNINDataset(
        DATA_ROOT,
        SPLIT_ROOT / "val.txt",
        window_size=200,
        step_size=10,
        layout="gru"
    )

    loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    x, y = next(iter(loader))

    print("\n========== 数据加载测试 ==========")
    print("训练样本数：", len(train_set))
    print("验证样本数：", len(val_set))
    print("批次输入形状：", x.shape)
    print("批次标签形状：", y.shape)
    print("输入数据是否正常：",
          torch.isfinite(x).all().item())
    print("标签数据是否正常：",
          torch.isfinite(y).all().item())

    # 单独检查 ResNet 使用的张量排列
    x_cnn, y_cnn = RoNINDataset(
        DATA_ROOT,
        SPLIT_ROOT / "val.txt",
        layout="cnn"
    )[0]

    print("ResNet单个输入形状：", x_cnn.shape)
    print("ResNet单个标签形状：", y_cnn.shape)
