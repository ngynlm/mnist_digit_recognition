"""Phần 5: giao diện Tkinter nhận diện chữ số viết tay.

Chạy:  python app.py          (dùng models/best_model.keras nếu có, không thì dùng mô hình GIẢ)
       python app.py --fake   (ép dùng mô hình giả để thử giao diện)
"""
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from preprocess import preprocess_image, to_model_input

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.keras"
CANVAS = 280          # 10x MNIST
BRUSH = 22            # nét dày, gần giống nét MNIST sau khi co về 28x28


class FakeModel:
    """Mô hình giả để dựng giao diện trước: trả xác suất ngẫu nhiên nhưng ổn định theo ảnh."""
    def predict(self, x, verbose=0):
        rng = np.random.default_rng(int(x.sum() * 1000) % (2**32))
        p = rng.random((len(x), 10)) ** 3
        return p / p.sum(1, keepdims=True)


def load_model(force_fake=False):
    if not force_fake and MODEL_PATH.exists():
        from tensorflow import keras
        return keras.models.load_model(MODEL_PATH), False
    return FakeModel(), True


class App:
    def __init__(self, root, model, is_fake):
        self.root, self.model, self.is_fake = root, model, is_fake
        root.title("Nhận diện chữ số viết tay - MNIST" + ("  [MÔ HÌNH GIẢ]" if is_fake else ""))
        root.resizable(False, False)

        left = tk.Frame(root, padx=10, pady=10)
        left.grid(row=0, column=0)
        tk.Label(left, text="Vẽ một chữ số (chuột trái)").pack()
        self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS, bg="black", cursor="pencil",
                                highlightthickness=1, highlightbackground="gray")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonPress-1>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "last", None))
        self.reset_image()

        btns = tk.Frame(left)
        btns.pack(pady=8)
        tk.Button(btns, text="Xóa", width=10, command=self.clear).grid(row=0, column=0, padx=3)
        tk.Button(btns, text="Mở ảnh", width=10, command=self.open_image).grid(row=0, column=1, padx=3)
        tk.Button(btns, text="Nhận diện", width=10, command=self.predict,
                  bg="#4C78A8", fg="white").grid(row=0, column=2, padx=3)

        right = tk.Frame(root, padx=10, pady=10)
        right.grid(row=0, column=1, sticky="n")
        self.result = tk.Label(right, text="?", font=("Arial", 72, "bold"), width=2)
        self.result.pack()
        self.conf = tk.Label(right, text="Độ tin cậy: --", font=("Arial", 12))
        self.conf.pack()
        tk.Label(right, text="Xác suất 10 lớp", font=("Arial", 10, "bold")).pack(pady=(10, 2))
        self.bars = []
        for d in range(10):
            row = tk.Frame(right)
            row.pack(fill="x")
            tk.Label(row, text=str(d), width=2).pack(side="left")
            c = tk.Canvas(row, width=170, height=14, bg="#eee", highlightthickness=0)
            c.pack(side="left", padx=3)
            rect = c.create_rectangle(0, 0, 0, 14, fill="#4C78A8", width=0)
            lbl = tk.Label(row, text="0.0%", width=6, anchor="e")
            lbl.pack(side="left")
            self.bars.append((c, rect, lbl))
        self.thumb = tk.Label(right, text="")
        self.thumb.pack(pady=(10, 0))
        tk.Label(right, text="Ảnh 28x28 đưa vào mô hình", font=("Arial", 8)).pack()

    # --- vẽ ---
    def reset_image(self):
        self.img = Image.new("L", (CANVAS, CANVAS), 0)
        self.pen = ImageDraw.Draw(self.img)
        self.last = None

    def draw(self, e):
        r = BRUSH / 2
        x, y = e.x, e.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.pen.ellipse([x - r, y - r, x + r, y + r], fill=255)
        if getattr(self, "last", None):
            lx, ly = self.last
            self.canvas.create_line(lx, ly, x, y, fill="white", width=BRUSH, capstyle="round")
            self.pen.line([lx, ly, x, y], fill=255, width=BRUSH)
        self.last = (x, y)

    def clear(self):
        self.canvas.delete("all")
        self.reset_image()
        self.result.config(text="?")
        self.conf.config(text="Độ tin cậy: --")
        for c, rect, lbl in self.bars:
            c.coords(rect, 0, 0, 0, 14)
            lbl.config(text="0.0%")
        self.thumb.config(image="", text="")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        try:
            self.predict_array(path)
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("Lỗi", f"Không xử lý được ảnh:\n{ex}")

    def predict(self):
        arr = np.array(self.img)
        if arr.max() == 0:
            messagebox.showinfo("Thông báo", "Hãy vẽ một chữ số trước.")
            return
        self.predict_array(arr)

    def predict_array(self, src):
        x28 = preprocess_image(src)
        if x28.max() == 0:
            messagebox.showinfo("Thông báo", "Không tìm thấy chữ số trong ảnh.")
            return
        p = self.model.predict(to_model_input(x28), verbose=0)[0]
        k = int(np.argmax(p))
        self.result.config(text=str(k))
        self.conf.config(text=f"Độ tin cậy: {p[k]:.1%}")
        for d, (c, rect, lbl) in enumerate(self.bars):
            c.coords(rect, 0, 0, 170 * float(p[d]), 14)
            c.itemconfig(rect, fill="#E45756" if d == k else "#4C78A8")
            lbl.config(text=f"{p[d]:.1%}")
        t = Image.fromarray((x28 * 255).astype(np.uint8)).resize((84, 84), Image.NEAREST)
        self.tk_thumb = ImageTk.PhotoImage(t)
        self.thumb.config(image=self.tk_thumb)


def main():
    model, is_fake = load_model(force_fake="--fake" in sys.argv)
    root = tk.Tk()
    App(root, model, is_fake)
    root.mainloop()


if __name__ == "__main__":
    main()
