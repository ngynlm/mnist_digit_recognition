# Nhận xét kết quả trên tập test

- Test accuracy: **0.9921**, test loss: **0.0228**; sai 79/10000 ảnh.
- Chữ số nhận diện tốt nhất (F1): 1 (0.9956), 0 (0.9954), 8 (0.9944).
- Chữ số nhận diện kém nhất (F1): 9 (0.9885), 6 (0.9890), 5 (0.9894).
- Các cặp hay nhầm nhất:
  - 2 ↔ 7: 9 lần (2→7: 5, 7→2: 4).
  - 3 ↔ 5: 8 lần (3→5: 2, 5→3: 6).
  - 4 ↔ 6: 6 lần (4→6: 5, 6→4: 1).
  - 4 ↔ 9: 6 lần (4→9: 1, 9→4: 5).
  - 7 ↔ 9: 6 lần (7→9: 2, 9→7: 4).

Gợi ý giải thích: các cặp như 4↔9, 3↔5, 7↔2 có nét hình học gần giống nhau (phần vòng kín, nét ngang/xiên), kết hợp với kiểu viết ẩu hoặc nét đứt khiến mô hình nhầm. Xem outputs/misclassified.png để đối chiếu từng ảnh.