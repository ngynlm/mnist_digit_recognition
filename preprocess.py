"""Phần 4: tiền xử lý ảnh bất kỳ -> ảnh 28x28 chuẩn MNIST (chữ trắng nền đen, 0-1).

Pipeline: đọc -> xám -> lọc nhiễu -> Otsu (đảo màu) -> contour -> co 20x20 giữ tỉ lệ
          -> căn giữa theo tâm khối lượng trong 28x28 -> chuẩn hóa 0-1

    from preprocess import preprocess_image
    x = preprocess_image("anh.jpg")            # (28, 28) float32 0-1
    x, steps = preprocess_image("anh.jpg", return_steps=True)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

MIN_CONTOUR_AREA = 12  # bỏ các đốm nhiễu nhỏ


def read_gray(src):
    """Nhận đường dẫn hoặc mảng numpy (xám/BGR/BGRA) -> ảnh xám uint8."""
    if isinstance(src, (str, Path)):
        # imdecode để đọc được cả đường dẫn có dấu tiếng Việt trên Windows
        data = np.fromfile(str(src), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Không đọc được ảnh: {src}")
    else:
        img = np.asarray(src)
    if img.dtype != np.uint8:
        img = np.clip(img * (255 if img.max() <= 1 else 1), 0, 255).astype(np.uint8)
    if img.ndim == 3:
        if img.shape[2] == 4:  # PNG trong suốt: ghép lên nền trắng
            alpha = img[:, :, 3:4] / 255.0
            img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def denoise(gray, method="gaussian"):
    # Kernel tỉ lệ theo cỡ ảnh: ảnh nhỏ (như 28x28) chỉ lọc nhẹ để không làm nhòe nét chữ
    k = 3 if min(gray.shape) < 64 else 5
    if method == "median":
        return cv2.medianBlur(gray, k)
    return cv2.GaussianBlur(gray, (k, k), 0)


def flatten_background(gray):
    """Đưa về chữ TỐI trên nền SÁNG và san phẳng ánh sáng không đều (chia cho nền ước lượng)."""
    if np.median(gray) < 127:          # nền tối (ảnh MNIST, canvas đen) -> đảo
        gray = 255 - gray
    k = max(7, min(gray.shape) // 5) | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (k, k), 0).astype(np.float32)
    norm = np.clip(gray.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255)
    return norm.astype(np.uint8)


def binarize(blur):
    """San phẳng nền + Otsu đảo màu -> luôn ra chữ TRẮNG trên nền ĐEN."""
    flat = flatten_background(blur)
    _, binary = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def clean_mask(binary, rel_area=0.08):
    """Giữ các contour đủ lớn (>= rel_area * contour lớn nhất) để bỏ đốm nhiễu. Trả (mask, contours)."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if not areas or max(areas) < MIN_CONTOUR_AREA:
        return np.zeros_like(binary), []
    keep = [c for c, a in zip(contours, areas) if a >= max(MIN_CONTOUR_AREA, rel_area * max(areas))]
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, keep, -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(binary, mask), keep


def find_digit_box(contours):
    """Hộp bao của các contour giữ lại. Trả (x, y, w, h) hoặc None."""
    if not contours:
        return None
    return cv2.boundingRect(np.vstack(contours))


def resize_20(crop):
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)


def center_by_mass(digit20):
    """Đặt chữ 20x20 vào khung 28x28 sao cho tâm khối lượng nằm ở (14, 14)."""
    canvas = np.zeros((28, 28), dtype=np.uint8)
    h, w = digit20.shape
    canvas[(28 - h) // 2:(28 - h) // 2 + h, (28 - w) // 2:(28 - w) // 2 + w] = digit20
    m = cv2.moments(canvas, binaryImage=False)
    if m["m00"] == 0:
        return canvas
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    shift = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
    return cv2.warpAffine(canvas, shift, (28, 28), flags=cv2.INTER_LINEAR)


def preprocess_image(src, denoise_method="gaussian", return_steps=False):
    gray = read_gray(src)
    blur = denoise(gray, denoise_method)
    binary, contours = clean_mask(binarize(blur))
    box = find_digit_box(contours)
    steps = {"1. Ảnh gốc (xám)": gray, "2. Lọc nhiễu": blur, "3. Nhị phân (Otsu, đảo, bỏ nhiễu)": binary}
    if box is None:  # ảnh trống
        out = np.zeros((28, 28), np.float32)
        steps["4. Contour + hộp bao"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        steps["5. Co về 20x20"] = np.zeros((20, 20), np.uint8)
        steps["6. Căn tâm 28x28"] = np.zeros((28, 28), np.uint8)
        return (out, steps) if return_steps else out
    x, y, w, h = box
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), max(1, min(binary.shape) // 100))
    crop = binary[y:y + h, x:x + w]
    d20 = resize_20(crop)
    centered = center_by_mass(d20)
    steps["4. Contour + hộp bao"] = vis
    steps["5. Co về 20x20"] = d20
    steps["6. Căn tâm 28x28"] = centered
    out = (centered.astype(np.float32) / 255.0)
    steps["7. Chuẩn hóa 0-1"] = out
    return (out, steps) if return_steps else out


def to_model_input(x28):
    return x28.reshape(1, 28, 28, 1).astype(np.float32)


def save_steps_figure(steps, path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.9))
    for ax, (name, img) in zip(np.atleast_1d(axes), steps.items()):
        ax.imshow(img, cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=1 if img.dtype == np.float32 else 255)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        _, st = preprocess_image(p, return_steps=True)
        out = Path(p).with_suffix("").as_posix() + "_steps.png"
        save_steps_figure(st, out, Path(p).name)
        print("Đã lưu", out)
