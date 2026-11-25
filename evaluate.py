"""Phần 3: đánh giá mô hình trên tập test. CHỈ CHẠY MỘT LẦN Ở CUỐI sau khi đã chốt mô hình.

Chạy: python evaluate.py [đường_dẫn_model]
"""
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from data_utils import ROOT, load_test

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
model_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "models" / "best_model.keras"

model = keras.models.load_model(model_path)
x_te, y_te = load_test()
loss, acc = model.evaluate(x_te, y_te, verbose=0)
print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")

probs = model.predict(x_te, verbose=0)
pred = probs.argmax(1)

# Confusion matrix
cm = confusion_matrix(y_te, pred)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
ax.set_title(f"Confusion matrix (test accuracy = {acc:.2%})")
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im)
plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()

# Classification report
rep = pd.DataFrame(classification_report(y_te, pred, digits=4, output_dict=True)).T
rep.round(4).to_csv(OUT / "classification_report.csv")
print(classification_report(y_te, pred, digits=4))

# Các cặp hay nhầm
pairs = []
for i, j in combinations(range(10), 2):
    pairs.append((i, j, cm[i, j] + cm[j, i], cm[i, j], cm[j, i]))
pairs.sort(key=lambda t: -t[2])
top = pairs[:6]
print("Các cặp hay nhầm nhất:")
for i, j, tot, a, b in top:
    print(f"  {i}<->{j}: {tot} lần ({i}->{j}: {a}, {j}->{i}: {b})")

# Ảnh dự đoán sai (lấy 40 ảnh sai tự tin nhất)
wrong = np.where(pred != y_te)[0]
order = wrong[np.argsort(-probs[wrong].max(1))][:40]
fig, axes = plt.subplots(4, 10, figsize=(13, 6))
for ax in axes.ravel():
    ax.axis("off")
for ax, k in zip(axes.ravel(), order):
    ax.imshow(x_te[k, :, :, 0], cmap="gray")
    ax.set_title(f"T:{y_te[k]} D:{pred[k]}", fontsize=8, color="red")
fig.suptitle(f"Ảnh đoán sai (T = thực tế, D = dự đoán) - tổng {len(wrong)}/{len(y_te)} ảnh")
plt.tight_layout(); plt.savefig(OUT / "misclassified.png", dpi=150); plt.close()

# Nhận xét tự động cho báo cáo
worst = rep.iloc[:10].sort_values("f1-score").head(3)
best = rep.iloc[:10].sort_values("f1-score", ascending=False).head(3)
lines = [
    "# Nhận xét kết quả trên tập test",
    "",
    f"- Test accuracy: **{acc:.4f}**, test loss: **{loss:.4f}**; sai {len(wrong)}/{len(y_te)} ảnh.",
    "- Chữ số nhận diện tốt nhất (F1): " + ", ".join(f"{i} ({r['f1-score']:.4f})" for i, r in best.iterrows()) + ".",
    "- Chữ số nhận diện kém nhất (F1): " + ", ".join(f"{i} ({r['f1-score']:.4f})" for i, r in worst.iterrows()) + ".",
    "- Các cặp hay nhầm nhất:",
]
for i, j, tot, a, b in top[:5]:
    lines.append(f"  - {i} ↔ {j}: {tot} lần ({i}→{j}: {a}, {j}→{i}: {b}).")
lines += [
    "",
    "Gợi ý giải thích: các cặp như 4↔9, 3↔5, 7↔2 có nét hình học gần giống nhau "
    "(phần vòng kín, nét ngang/xiên), kết hợp với kiểu viết ẩu hoặc nét đứt khiến mô hình nhầm. "
    "Xem outputs/misclassified.png để đối chiếu từng ảnh.",
]
(OUT / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
print("Đã lưu confusion_matrix.png, classification_report.csv, misclassified.png, error_analysis.md")
