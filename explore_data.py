"""Phần 1: kiểm tra shape, phân bố nhãn, vẽ ảnh mẫu."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_utils import TRAIN_CSV, TEST_CSV, ROOT, load_csv

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

x_tr, y_tr = load_csv(TRAIN_CSV)
x_te, y_te = load_csv(TEST_CSV)
print("Train (có cột label):", (x_tr.shape[0], x_tr.shape[1] + 1))
print("Test  (có cột label):", (x_te.shape[0], x_te.shape[1] + 1))
assert (x_tr.shape[0], x_tr.shape[1] + 1) == (60000, 785)
assert (x_te.shape[0], x_te.shape[1] + 1) == (10000, 785)
print("Giá trị pixel:", x_tr.min(), "-", x_tr.max())

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, y, name in zip(axes, (y_tr, y_te), ("Tập train (60.000)", "Tập test (10.000)")):
    counts = np.bincount(y, minlength=10)
    bars = ax.bar(range(10), counts, color="#4C78A8")
    ax.set_title(f"Phân bố nhãn - {name}")
    ax.set_xlabel("Chữ số")
    ax.set_xticks(range(10))
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom", fontsize=8)
axes[0].set_ylabel("Số ảnh")
plt.tight_layout()
plt.savefig(OUT / "label_distribution.png", dpi=150)
plt.close()

fig, axes = plt.subplots(5, 10, figsize=(12, 6.5))
rng = np.random.default_rng(0)
for d in range(10):
    idx = rng.choice(np.where(y_tr == d)[0], 5, replace=False)
    for r, i in enumerate(idx):
        ax = axes[r, d]
        ax.imshow(x_tr[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
        if r == 0:
            ax.set_title(str(d))
fig.suptitle("Ảnh mẫu MNIST (mỗi cột một chữ số)")
plt.tight_layout()
plt.savefig(OUT / "sample_images.png", dpi=150)
print("Đã lưu outputs/label_distribution.png và outputs/sample_images.png")
