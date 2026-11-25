# Kết quả chạy (số liệu thật từ outputs/)

## Dữ liệu
Train 60.000×785, test 10.000×785. Nhãn khá cân bằng (train: 5421–6742 ảnh/chữ số; ít nhất là chữ 5, nhiều nhất là chữ 1). Xem `label_distribution.png`.

## So sánh biến thể (validation 6.000 ảnh)
| Biến thể | Filters | Dropout | Epochs | Params | Train acc | Val acc | Val loss |
|---|---|---|---|---|---|---|---|
| baseline | 32-64 | 0 | 5 | 225.034 | 0,9927 | 0,9873 | 0,0496 |
| dropout | 32-64 | 0,4 | 5 | 225.034 | 0,9822 | 0,9882 | 0,0453 |
| more_filters | 64-128 | 0,4 | 5 | 485.514 | 0,9863 | 0,9890 | 0,0401 |
| **dropout_12ep** | 32-64 | 0,4 | 12 | 225.034 | 0,9901 | **0,9895** | **0,0386** |

Nhận xét: baseline có train acc cao hơn val acc rõ nhất (0,9927 so với 0,9873) → dấu hiệu overfit nhẹ; thêm Dropout thu hẹp khoảng cách này. Tăng số filter hoặc số epoch đều cải thiện thêm một chút. Chênh lệch giữa các biến thể chỉ ~0,2% (≈ 12 ảnh trên 6.000), chạy một seed nên nên coi chúng là xấp xỉ nhau.

## Test (mô hình dropout_12ep, đánh giá 1 lần)
Accuracy **99,19%**, loss 0,0221, sai 81/10.000 ảnh. F1 thấp nhất: chữ 9 (0,9875), 5 (0,9883), 7 (0,9889).
Cặp hay nhầm nhất: 7↔9 (8 lần), 2↔7, 3↔5, 4↔9, 5↔6 (7 lần mỗi cặp). Xem `error_analysis.md`, `misclassified.png`.

## Tiền xử lý
7/7 ảnh mẫu đoán đúng (3 ảnh MNIST + 4 ảnh **mô phỏng** chụp giấy có lệch sáng, nhiễu muối tiêu, xoay nhẹ). Đây KHÔNG phải ảnh chụp thật; nhóm cần thử thêm ảnh tự chụp.
