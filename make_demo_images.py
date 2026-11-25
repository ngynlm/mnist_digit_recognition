"""Tạo ảnh thử cho preprocess: ảnh MNIST mẫu và ảnh MÔ PHỎNG chụp giấy (nền sáng, chữ tối, nhiễu, lệch sáng).
Ảnh chụp thật của nhóm nên bỏ vào sample_images/ (đặt tên bất kỳ .jpg/.png)."""
import cv2
import numpy as np
from data_utils import TEST_CSV, ROOT, load_csv

OUT = ROOT / "sample_images"
OUT.mkdir(exist_ok=True)
x, y = load_csv(TEST_CSV)
rng = np.random.default_rng(7)
for d in (3, 5, 7):  # ảnh MNIST gốc 28x28 (chữ trắng nền đen)
    i = np.where(y == d)[0][0]
    cv2.imwrite(str(OUT / f"mnist_{d}.png"), x[i].reshape(28, 28))
for d in (2, 4, 9, 6):  # ảnh mô phỏng chụp
    i = np.where(y == d)[0][5]
    big = cv2.resize(x[i].reshape(28, 28), (160, 160), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((400, 500), np.uint8)
    canvas[110:270, 170:330] = big
    M = cv2.getRotationMatrix2D((250, 190), rng.uniform(-12, 12), 1)
    canvas = cv2.warpAffine(canvas, M, (500, 400))
    ink = canvas / 255.0
    yy, xx = np.mgrid[0:400, 0:500]
    light = 0.75 + 0.25 * (xx / 500) + 0.05 * rng.standard_normal()  # lệch sáng
    paper = np.dstack([light * c for c in (0.93, 0.97, 1.0)])      # BGR, giấy hơi ngả vàng
    img = paper * (1 - 0.85 * ink[..., None])
    img += rng.normal(0, 0.03, img.shape)
    mask = rng.random(img.shape[:2]) < 0.002
    img[mask] = rng.choice([0.0, 1.0])
    cv2.imwrite(str(OUT / f"photo_sim_{d}.jpg"), np.clip(img * 255, 0, 255).astype(np.uint8))
print(sorted(p.name for p in OUT.iterdir()))
