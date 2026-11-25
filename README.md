# Nhận diện chữ số viết tay MNIST (CNN + OpenCV + Tkinter)

## Cài đặt
```
pip install -r requirements.txt
```
Đặt `mnist_train.csv` (60.000×785) và `mnist_test.csv` (10.000×785) vào thư mục `data/`.
(Bản trong gói zip không kèm CSV vì nặng ~128MB; dùng file tải từ Kaggle của nhóm.)

## Thứ tự chạy
| Bước | Lệnh | Đầu ra |
|---|---|---|
| 1. Dữ liệu | `python explore_data.py` | `outputs/label_distribution.png`, `sample_images.png` |
| 2. Huấn luyện | `python train_model.py` | `models/*.keras`, `outputs/history_*.png`, `variant_comparison.csv` |
| 3. Đánh giá (chạy **một lần** cuối) | `python evaluate.py` | `confusion_matrix.png`, `classification_report.csv`, `misclassified.png`, `error_analysis.md` |
| 4. Tiền xử lý | `python make_demo_images.py` (tùy chọn) rồi `python demo_preprocess.py` | `outputs/preprocess_steps/*.png` |
| 5. Giao diện | `python app.py` (`--fake` để thử giao diện không cần model) | |

Ảnh tự chụp của nhóm: bỏ vào `sample_images/` rồi chạy `python demo_preprocess.py`.

## Cấu trúc
- `data_utils.py` – load/chuẩn hóa dữ liệu dùng chung (`load_train_val`, `load_test`).
- `train_model.py` – CNN + 4 biến thể; validation tách 10% từ train (phân tầng, seed 42).
- `evaluate.py` – đánh giá trên test.
- `preprocess.py` – gray → lọc nhiễu → san phẳng nền + Otsu (đảo) → contour → 20×20 → căn tâm khối lượng 28×28 → 0–1.
- `app.py` – Tkinter: canvas, Xóa / Mở ảnh / Nhận diện, xác suất 10 lớp.
