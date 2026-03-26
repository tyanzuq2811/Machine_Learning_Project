# Tổng Quan Đề Tài

## Phát Hiện Viêm Phổi Từ Ảnh X-quang Ngực Bằng Học Máy Lai

## 1. Giới thiệu
Đề tài xây dựng hệ thống hỗ trợ phân loại ảnh X-quang ngực thành hai nhóm:
- NORMAL: phổi bình thường
- PNEUMONIA: viêm phổi

Mục tiêu cốt lõi của bài toán là tối ưu khả năng phát hiện ca bệnh thật (độ nhạy cao), đồng thời duy trì độ chính xác tổng thể và tính ổn định khi áp dụng trên dữ liệu chưa thấy.

## 2. Mục tiêu nghiên cứu
- Xây dựng pipeline đầy đủ từ dữ liệu gốc đến mô hình suy luận.
- Kết hợp đặc trưng học sâu (ResNet50) và đặc trưng texture cổ điển (GLCM).
- Áp dụng Stacking Ensemble để nâng cao hiệu năng so với mô hình đơn lẻ.
- Trực quan hóa quy trình qua dashboard và triển khai webapp để dự đoán ảnh mới.

## 3. Dữ liệu và chiến lược chia tập

### 3.1 Thông tin dữ liệu
| Thuộc tính | Giá trị |
|---|---:|
| Tổng số ảnh | 5,856 |
| Tổng số bệnh nhân | 2,790 |
| NORMAL | 1,583 |
| PNEUMONIA | 4,273 |

Nhận xét: dữ liệu mất cân bằng lớp, do đó khi đánh giá mô hình cần ưu tiên các chỉ số lâm sàng như Recall, Specificity, AUC-ROC thay vì chỉ nhìn Accuracy.

### 3.2 Nguyên tắc chia tập
- Chia theo Patient ID để tránh rò rỉ dữ liệu giữa train, validation, test.
- Tỷ lệ chia mục tiêu: 80% train, 10% validation, 10% test.
- Đảm bảo phân phối lớp tương đối ổn định giữa các tập.

## 4. Kiến trúc pipeline kỹ thuật
Pipeline trong thư mục [chest_xray](chest_xray) gồm các bước chính:

1. Phân tích và chia lại dữ liệu theo bệnh nhân.
2. Tiền xử lý ảnh:
	- CLAHE tăng tương phản cục bộ
	- Resize về 224 x 224
	- Chuyển ảnh xám thành 3 kênh để tương thích ResNet50
3. Tăng cường dữ liệu trên train:
	- Lật ngang
	- Xoay nhẹ trong ngưỡng nhỏ
4. Trích xuất đặc trưng lai:
	- ResNet50 frozen: đặc trưng học sâu
	- GLCM: đặc trưng texture
5. Chuẩn hóa và giảm chiều:
	- StandardScaler
	- PCA giữ xấp xỉ 95% phương sai
6. Huấn luyện Stacking Ensemble:
	- Tầng base: SVM, Random Forest, XGBoost
	- Tầng meta: Logistic Regression
7. Đánh giá bằng các chỉ số y tế và ma trận nhầm lẫn.

## 5. Kết quả mô hình nổi bật

### 5.1 Kết quả Stacking trên tập test
| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 95.94% |
| Precision | 98.12% |
| Recall (Sensitivity) | 96.31% |
| F1-score | 97.21% |
| AUC-ROC | 0.9888 |
| Specificity | 94.90% |
| Chênh lệch Val-Test | 0.87% |

### 5.2 Diễn giải ngắn
- Recall cao cho thấy mô hình giảm tốt nguy cơ bỏ sót ca bệnh.
- AUC-ROC rất cao cho thấy khả năng phân tách NORMAL và PNEUMONIA rõ ràng.
- Chênh lệch Val-Test thấp cho thấy mô hình có tính khái quát tốt.

## 6. Thành phần triển khai trong dự án
- [chest_xray](chest_xray): mã nguồn chính cho tiền xử lý, trích xuất đặc trưng, huấn luyện và đánh giá.
- [chest_xray/dashboard_app](chest_xray/dashboard_app): dashboard Dash trực quan hóa pipeline 7 bước.
- [chest_xray/webapp](chest_xray/webapp): ứng dụng web phục vụ suy luận ảnh mới.
- [chest_xray/models_stacking](chest_xray/models_stacking): scaler, PCA, classifier đã huấn luyện.
- [chest_xray/results_stacking](chest_xray/results_stacking): tệp kết quả đánh giá mô hình.

## 7. Hướng dẫn chạy nhanh

### 7.1 Yêu cầu môi trường
- Python 3.10 trở lên
- Cài thư viện từ [requirements.txt](requirements.txt)

### 7.2 Cài đặt
```bash
pip install -r requirements.txt
```

### 7.3 Chạy dashboard EDA
```bash
python chest_xray/eda_dashboard.py
```

### 7.4 Chạy ứng dụng web suy luận
```bash
python chest_xray/webapp/main.py
```

## 8. Giá trị ứng dụng
- Hỗ trợ sàng lọc ban đầu cho bác sĩ và kỹ thuật viên.
- Tăng tốc xử lý ảnh ở bối cảnh số lượng lớn.
- Là nền tảng mở rộng sang các bài toán bệnh phổi khác hoặc mô hình đa lớp trong tương lai.

## 9. Tài liệu liên quan
- [chest_xray/KET_QUA_PIPELINE.md](chest_xray/KET_QUA_PIPELINE.md): báo cáo kết quả chi tiết.
- [chest_xray/BAO_CAO_PHAN_TICH_DU_LIEU.md](chest_xray/BAO_CAO_PHAN_TICH_DU_LIEU.md): báo cáo phân tích dữ liệu.
- [chest_xray/BAO_CAO_TIEN_XU_LY_DU_LIEU.md](chest_xray/BAO_CAO_TIEN_XU_LY_DU_LIEU.md): báo cáo tiền xử lý dữ liệu.
