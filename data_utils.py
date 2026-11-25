"""Tiện ích dữ liệu dùng chung cho cả nhóm.

Dùng:
    from data_utils import load_train_val, load_test
    x_tr, y_tr, x_val, y_val = load_train_val()
    x_te, y_te = load_test()          # CHỈ gọi trong evaluate.py

Ảnh trả về có shape (N, 28, 28, 1), float32, giá trị 0-1.
Nhãn là số nguyên 0-9 (shape (N,)).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "mnist_train.csv"
TEST_CSV = DATA_DIR / "mnist_test.csv"
NUM_CLASSES = 10
IMG_SIZE = 28
SEED = 42


def load_csv(path):
    """Đọc CSV Kaggle (cột 0 = label, 784 cột pixel 0-255) -> (X thô uint8, y)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy {path}. Hãy tải MNIST CSV từ Kaggle và đặt vào thư mục data/ "
            "với tên mnist_train.csv và mnist_test.csv."
        )
    df = pd.read_csv(path)
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)
    x = df.iloc[:, 1:].to_numpy(dtype=np.uint8)
    return x, y


def normalize(x):
    """(N,784) hoặc (N,28,28) uint8 -> (N,28,28,1) float32 trong [0,1]."""
    x = np.asarray(x, dtype=np.float32) / 255.0
    return x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def load_train_val(val_ratio=0.1, seed=SEED):
    """Tách validation từ tập train (có phân tầng theo nhãn). Không đụng tới tập test."""
    from sklearn.model_selection import train_test_split

    x, y = load_csv(TRAIN_CSV)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=val_ratio, random_state=seed, stratify=y
    )
    return normalize(x_tr), y_tr, normalize(x_val), y_val


def load_test():
    """Tập test - chỉ dùng ở bước đánh giá cuối cùng."""
    x, y = load_csv(TEST_CSV)
    return normalize(x), y


def label_distribution(y):
    return pd.Series(y).value_counts().sort_index()
