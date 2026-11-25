"""Phần 2: huấn luyện CNN, so sánh các biến thể.

Chạy:  python train_model.py            (tất cả biến thể)
       python train_model.py baseline   (chỉ một biến thể)
Mô hình tốt nhất (theo val_accuracy) được lưu vào models/best_model.keras.
Tập test KHÔNG được dùng ở đây.
"""
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_utils import ROOT, SEED, load_train_val

MODELS = ROOT / "models"
OUT = ROOT / "outputs"
MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# Mỗi biến thể: số filter, dropout, số epoch
VARIANTS = {
    "baseline":       dict(filters=(32, 64), dropout=0.0, epochs=5),
    "dropout":        dict(filters=(32, 64), dropout=0.4, epochs=5),
    "more_filters":   dict(filters=(64, 128), dropout=0.4, epochs=5),
    "dropout_12ep":   dict(filters=(32, 64), dropout=0.4, epochs=12),
}


def build_model(filters=(32, 64), dropout=0.0):
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(filters[0], 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(filters[1], 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        *([layers.Dropout(dropout)] if dropout > 0 else []),
        layers.Dense(128, activation="relu"),
        *([layers.Dropout(dropout)] if dropout > 0 else []),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_history(name, hist):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(hist["accuracy"]) + 1)
    axes[0].plot(ep, hist["accuracy"], marker="o", label="train")
    axes[0].plot(ep, hist["val_accuracy"], marker="o", label="validation")
    axes[0].set_title(f"Accuracy - {name}")
    axes[1].plot(ep, hist["loss"], marker="o", label="train")
    axes[1].plot(ep, hist["val_loss"], marker="o", label="validation")
    axes[1].set_title(f"Loss - {name}")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"history_{name}.png", dpi=150)
    plt.close()


def main(names):
    keras.utils.set_random_seed(SEED)
    x_tr, y_tr, x_val, y_val = load_train_val()
    print("Train:", x_tr.shape, "Val:", x_val.shape)
    rows, histories = [], {}
    best_acc = -1
    for name in names:
        cfg = VARIANTS[name]
        keras.utils.set_random_seed(SEED)
        model = build_model(cfg["filters"], cfg["dropout"])
        t0 = time.time()
        h = model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=cfg["epochs"],
                      batch_size=128, verbose=2)
        secs = time.time() - t0
        hist = h.history
        histories[name] = hist
        plot_history(name, hist)
        model.save(MODELS / f"model_{name}.keras")
        rows.append({
            "Biến thể": name,
            "Filters": f"{cfg['filters'][0]}-{cfg['filters'][1]}",
            "Dropout": cfg["dropout"],
            "Epochs": cfg["epochs"],
            "Params": model.count_params(),
            "Train acc": round(hist["accuracy"][-1], 4),
            "Val acc": round(hist["val_accuracy"][-1], 4),
            "Val loss": round(hist["val_loss"][-1], 4),
            "Thời gian (s)": round(secs),
        })
        if hist["val_accuracy"][-1] > best_acc:
            best_acc = hist["val_accuracy"][-1]
            best = name
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "variant_comparison.csv", index=False)
    print(table.to_string(index=False))
    model = keras.models.load_model(MODELS / f"model_{best}.keras")
    model.save(MODELS / "best_model.keras")
    json.dump({"best_variant": best, "val_accuracy": best_acc}, open(MODELS / "best_info.json", "w"))
    print("Biến thể tốt nhất:", best, "-> models/best_model.keras")


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARIANTS))
