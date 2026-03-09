# KẾT QUẢ PIPELINE: PHÁT HIỆN VIÊM PHỔI TỪ ẢNH X-QUANG NGỰC

> **Ngày chạy:** 25/02/2026  
> **Thời gian thực thi:** ~22 phút  
> **Phương pháp:** ResNet50 (frozen) + Stacking Ensemble (SVM + Random Forest + XGBoost)

---

## 1. Tổng quan dữ liệu

| Thông tin | Giá trị |
|-----------|---------|
| Tổng số ảnh X-quang | **5,856 ảnh** |
| Số bệnh nhân (unique) | **2,790 người** |
| Ảnh NORMAL (phổi bình thường) | 1,583 ảnh (27.0%) |
| Ảnh PNEUMONIA (viêm phổi) | 4,273 ảnh (73.0%) |
| Tỷ lệ mất cân bằng | PNEUMONIA : NORMAL = **2.7 : 1** |

### Chia dữ liệu (theo Patient ID — không trùng bệnh nhân giữa các tập)

| Tập | Bệnh nhân | Số ảnh | NORMAL | PNEUMONIA |
|-----|-----------|--------|--------|-----------|
| **Train** (80%) | 2,231 | 4,670 | 1,269 | 3,401 |
| **Val** (10%) | 279 | 595 | 157 | 438 |
| **Test** (10%) | 280 | 591 | 157 | 434 |

> Không có bệnh nhân nào xuất hiện ở 2 tập khác nhau → **Không bị data leakage**.

---

## 2. Tiền xử lý & Trích xuất đặc trưng

### 2.1. Tiền xử lý ảnh

| Bước | Mô tả |
|------|-------|
| **CLAHE** | Tăng tương phản ảnh X-quang (clipLimit=3.0) |
| **Resize** | Đưa về kích thước 224×224 pixel |
| **3 kênh màu** | Chuyển ảnh xám → 3 kênh (phù hợp ResNet50) |

### 2.2. Tăng cường dữ liệu (Augmentation — chỉ trên tập Train)

| Kỹ thuật | Chi tiết |
|----------|----------|
| Lật ngang | Mỗi ảnh → thêm 1 bản lật |
| Xoay ±15° | Mỗi ảnh → thêm 1 bản xoay ngẫu nhiên |
| **Kết quả** | 4,670 ảnh gốc → **14,010 ảnh** (×3) |

> Không lật dọc vì ảnh X-quang phổi luôn có hướng cố định (phổi trên, bụng dưới).

### 2.3. Trích xuất đặc trưng

| Nguồn | Số chiều | Mô tả |
|-------|----------|-------|
| **ResNet50** (frozen, ImageNet) | 2,048 | Đặc trưng hình ảnh cấp cao |
| **GLCM** | 6 | Đặc trưng kết cấu (texture) |
| **Vector kết hợp** | **2,054** | Ghép 2 nguồn trên → 1 vector |

### 2.4. Giảm chiều PCA

| Trước PCA | Sau PCA | Giảm | Thông tin giữ lại |
|-----------|---------|------|--------------------|
| 2,054 chiều | **997 chiều** | **51.5%** | 95.01% variance |

> PCA loại bỏ các chiều nhiễu, giữ lại 95% thông tin quan trọng → giảm overfitting, tăng tốc training.

---

## 3. Kết quả mô hình

### 3.1. So sánh các model đơn lẻ vs Stacking

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM (RBF) | 96.11% | 97.68% | 97.00% | 97.34% |
| Random Forest | 87.82% | 86.20% | 99.31% | 92.29% |
| XGBoost | 94.59% | 97.63% | 94.93% | 96.26% |
| **⭐ Stacking Ensemble** | **95.94%** | **98.12%** | **96.31%** | **97.21%** |

**Nhận xét:**

- **SVM** đạt accuracy cao nhất (96.11%), là model đơn lẻ tốt nhất.
- **Random Forest** có recall cao nhất (99.31% — gần như không bỏ sót ca bệnh nào) nhưng precision thấp → cảnh báo nhầm nhiều.
- **XGBoost** cân bằng giữa precision và recall.
- **Stacking** kết hợp cả 3 → precision rất cao (98.12%) và recall tốt (96.31%).

### 3.2. Kết quả Stacking Ensemble (kết quả cuối cùng)

```
┌──────────────────────────────────────────────┐
│  Accuracy:     95.94%                        │
│  Precision:    98.12%                        │
│  Recall:       96.31%                        │
│  F1-Score:     97.21%                        │
│  AUC-ROC:      0.9888                        │
│  Overfitting:  +0.87% (Val 96.81% → Test 95.94%) │
└──────────────────────────────────────────────┘
```

### 3.3. Giải thích từng chỉ số — Dễ hiểu

> Tưởng tượng bạn là **bác sĩ AI** đọc 591 ảnh X-quang. Trong đó có 434 ảnh viêm phổi thật và 157 ảnh phổi bình thường. Bạn cần phân loại từng ảnh.

---

#### ① Accuracy (Độ chính xác tổng thể) = 95.94%

**Định nghĩa:** Tỷ lệ số ảnh model phân loại **ĐÚNG** trên **TỔNG SỐ** ảnh.

$$\text{Accuracy} = \frac{\text{Số ảnh đúng}}{\text{Tổng số ảnh}} = \frac{567}{591} = 95.94\%$$

**Ví dụ dễ hiểu:** Bạn làm bài kiểm tra 591 câu, trả lời đúng 567 câu → điểm 95.94/100.

**Hạn chế:** Nếu dữ liệu mất cân bằng (ví dụ 90% là PNEUMONIA), model chỉ cần đoán "PNEUMONIA" cho tất cả cũng đạt accuracy 90% mà không hề thông minh. Vì vậy accuracy **không đủ** để đánh giá, cần thêm các chỉ số bên dưới.

---

#### ② Precision (Độ chính xác dương) = 98.12%

**Định nghĩa:** Trong tất cả các lần model **nói "PNEUMONIA"**, bao nhiêu lần nó nói **ĐÚNG**?

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{418}{418 + 8} = 98.12\%$$

**Ví dụ dễ hiểu:** Model kêu chuông báo "viêm phổi" cho 426 ảnh. Trong đó:
- 418 ảnh đúng là viêm phổi ✅
- 8 ảnh thực ra là phổi bình thường (báo nhầm) ❌

→ Precision **cao** = model rất ít khi **báo nhầm** người khỏe thành bệnh.

**Khi nào quan trọng?** Khi chi phí báo nhầm cao (ví dụ: phẫu thuật nhầm, điều trị không cần thiết).

---

#### ③ Recall / Sensitivity (Độ nhạy) = 96.31% ⭐ QUAN TRỌNG NHẤT

**Định nghĩa:** Trong tất cả các ca viêm phổi **THẬT**, model phát hiện được bao nhiêu ca?

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{418}{418 + 16} = 96.31\%$$

**Ví dụ dễ hiểu:** Có 434 bệnh nhân viêm phổi thật đứng trong hàng. Model quét từng người:
- Phát hiện đúng 418 người → cho đi điều trị ✅
- **Bỏ sót 16 người** → nói "bạn khỏe, về đi" ❌ (**NGUY HIỂM!**)

→ Recall **cao** = model **ít bỏ sót** bệnh nhân thật.

> ### ⭐ Tại sao Recall là chỉ số QUAN TRỌNG NHẤT?
>
> Trong y tế, **bỏ sót bệnh** (False Negative) nguy hiểm hơn **cảnh báo nhầm** (False Positive):
>
> | Loại sai | Ví dụ | Hậu quả |
> |----------|-------|---------|
> | **Báo nhầm** (FP) | Người khỏe bị nói "viêm phổi" | Chụp lại, xét nghiệm thêm → **tốn thời gian** |
> | **Bỏ sót** (FN) | Người bệnh bị nói "bình thường" | Không điều trị → bệnh nặng hơn → **NGUY HIỂM TÍNH MẠNG** |
>
> → Hệ thống y tế luôn ưu tiên **Recall cao** (ít bỏ sót nhất có thể), chấp nhận báo nhầm vài ca.
>
> **Ngưỡng lâm sàng:** Recall > 90% mới được xem là đạt yêu cầu. Model đạt **96.31%** → ✅ Vượt ngưỡng.

---

#### ④ Specificity (Độ đặc hiệu) = 94.90%

**Định nghĩa:** Trong tất cả người **BÌNH THƯỜNG thật**, model nhận ra đúng bao nhiêu người?

$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{149}{149 + 8} = 94.90\%$$

**Ví dụ dễ hiểu:** Có 157 người phổi bình thường. Model quét:
- Xác nhận đúng 149 người khỏe → "bạn bình thường" ✅
- Cảnh báo nhầm 8 người → "bạn có thể bị viêm phổi" ❌

→ Specificity là "bản sao" của Recall nhưng nhìn từ phía người khỏe.

**Ngưỡng lâm sàng:** > 85% → Model đạt **94.90%** → ✅ Vượt ngưỡng.

---

#### ⑤ F1-Score (Điểm F1) = 97.21%

**Định nghĩa:** Trung bình **điều hòa** (harmonic mean) của Precision và Recall.

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} = 2 \times \frac{0.9812 \times 0.9631}{0.9812 + 0.9631} = 97.21\%$$

**Ví dụ dễ hiểu:** Precision và Recall giống 2 đầu bập bênh — kéo cái này lên thì cái kia xuống. F1 cho biết model **cân bằng** giữa hai chỉ số này tốt đến đâu.

- Nếu Precision = 100% nhưng Recall = 50% → F1 chỉ = 66.7% (tệ!)
- Nếu Precision = 98% và Recall = 96% → F1 = 97.2% (tốt!)

→ F1 **cao** = model vừa ít báo nhầm, vừa ít bỏ sót.

---

#### ⑥ AUC-ROC (Diện tích dưới đường cong ROC) = 0.9888

**Định nghĩa:** Đo khả năng model **phân biệt** giữa NORMAL và PNEUMONIA ở mọi ngưỡng quyết định.

**Ví dụ dễ hiểu:** Hãy tưởng tượng model cho mỗi ảnh 1 "điểm nghi ngờ" từ 0 đến 1:
- Ảnh NORMAL thường được điểm thấp (gần 0)
- Ảnh PNEUMONIA thường được điểm cao (gần 1)

AUC-ROC đo xem 2 nhóm điểm này **tách biệt** nhau rõ hay chồng lấn lên nhau.

| AUC-ROC | Đánh giá |
|---------|----------|
| 0.5 | Đoán ngẫu nhiên (tung đồng xu) — vô dụng |
| 0.7 – 0.8 | Trung bình |
| 0.8 – 0.9 | Tốt |
| 0.9 – 1.0 | Rất tốt |
| 1.0 | Hoàn hảo (không bao giờ nhầm) |

→ Model đạt **0.9888** → gần hoàn hảo, 2 nhóm gần như tách biệt hoàn toàn.

---

#### ⑦ Overfitting (Quá khớp) = +0.87%

**Định nghĩa:** Chênh lệch giữa kết quả trên tập Val (model đã dùng để điều chỉnh) và tập Test (dữ liệu hoàn toàn mới).

$$\text{Overfitting} = \text{Val Accuracy} - \text{Test Accuracy} = 96.81\% - 95.94\% = +0.87\%$$

**Ví dụ dễ hiểu:** Giống như học sinh:
- **Val** = làm bài tập ở nhà (quen đề)
- **Test** = đi thi thật (đề mới)
- Nếu ở nhà 96 điểm, đi thi 95 điểm → chênh lệch nhỏ → **học thật giỏi**
- Nếu ở nhà 97 điểm, đi thi 85 điểm → chênh lệch lớn → **học tủ** (overfitting!)

| Overfitting | Đánh giá |
|-------------|----------|
| < 2% | ✅ Rất tốt — model khái quát tốt |
| 2% – 5% | ⚠️ Chấp nhận được |
| > 5% | ❌ Overfitting — cần điều chỉnh |

→ Model đạt **+0.87%** → ✅ Không overfitting.

---

### 3.4. Bảng xếp hạng mức độ quan trọng của các chỉ số

| Hạng | Chỉ số | Tại sao? |
|------|--------|----------|
| **🥇 1** | **Recall (Sensitivity)** | Trực tiếp đo "bỏ sót bao nhiêu ca bệnh" — liên quan **tính mạng bệnh nhân** |
| **🥈 2** | **AUC-ROC** | Đánh giá tổng thể khả năng phân biệt 2 lớp, không phụ thuộc ngưỡng |
| **🥉 3** | **F1-Score** | Cân bằng giữa Recall và Precision — chỉ số "tổng hợp" đáng tin |
| 4 | **Specificity** | Đo khả năng nhận diện người khỏe — giảm hoảng loạn không cần thiết |
| 5 | **Precision** | Quan trọng nhưng trong y tế, cảnh báo nhầm ít nguy hiểm hơn bỏ sót |
| 6 | **Accuracy** | Dễ bị "lừa" bởi dữ liệu mất cân bằng — chỉ mang tính tham khảo |
| 7 | **Overfitting** | Kiểm tra model có ổn định không — quan trọng nhưng là chỉ số phụ |

---

## 4. Ma trận nhầm lẫn (Confusion Matrix)

```
                       Dự đoán
                   NORMAL    PNEUMONIA
Thực tế  NORMAL  │  149   │     8     │
      PNEUMONIA  │   16   │   418     │
```

| Loại | Số ca | Ý nghĩa |
|------|-------|---------|
| ✅ **True Negative (TN)** | 149 | Phổi bình thường → Model nói bình thường (**ĐÚNG**) |
| ✅ **True Positive (TP)** | 418 | Viêm phổi → Model nói viêm phổi (**ĐÚNG**) |
| ⚠️ **False Positive (FP)** | 8 | Phổi bình thường → Model nói viêm phổi (**SAI** — cảnh báo nhầm) |
| ❌ **False Negative (FN)** | 16 | Viêm phổi → Model nói bình thường (**SAI** — bỏ sót bệnh!) |

**Tổng kết:** 567 đúng / 24 sai trên 591 ảnh test.

---

## 5. Đánh giá theo góc nhìn y tế

### Sensitivity (Recall) = 96.31%

- Trong 434 bệnh nhân viêm phổi → phát hiện đúng **418 người**.
- Bỏ sót **16 người** bệnh (3.69%).
- ✅ **Tốt** (ngưỡng lâm sàng yêu cầu > 90%).

### Specificity = 94.90%

- Trong 157 người khỏe mạnh → xác nhận đúng **149 người**.
- Cảnh báo nhầm **8 người** khỏe (5.10%).
- ✅ **Tốt** (ngưỡng lâm sàng yêu cầu > 85%).

### Tại sao Recall quan trọng hơn Precision trong y tế?

| Loại sai | Hậu quả | Mức độ |
|----------|---------|--------|
| **False Negative** (bỏ sót bệnh) | Bệnh nhân không được điều trị → bệnh nặng hơn | **NGUY HIỂM** |
| **False Positive** (cảnh báo nhầm) | Người khỏe phải chụp thêm / khám lại | Chấp nhận được |

→ Model này bỏ sót **16/434** ca bệnh (3.7%) — mức chấp nhận được cho công cụ hỗ trợ sàng lọc.

---

## 6. Overfitting — Model có bị "học tủ" không?

| Metric | Val (tập kiểm tra trong lúc train) | Test (tập chưa từng thấy) | Chênh lệch |
|--------|-----|------|-------------|
| Accuracy | 96.81% | 95.94% | **+0.87%** |

- Chênh lệch chỉ **0.87%** → model **không bị overfitting**.
- Ngưỡng cảnh báo overfitting thường là > 5%.
- So sánh: pipeline cũ (EfficientNet B7 fine-tuning) từng có chênh lệch **10-12%** → bị overfitting nặng.

---

## 7. Thời gian thực thi

| Bước | Thời gian |
|------|-----------|
| Bước 1: Chia lại dữ liệu | ~3 giây |
| Bước 2-4: Tiền xử lý + Trích xuất features | ~13 phút |
| Bước 5-7: PCA + Training Stacking + Đánh giá | ~9.4 phút |
| **Tổng** | **~22 phút** |

---

## 8. Files đầu ra

| Thư mục | Nội dung |
|---------|----------|
| `data_resplit/` | Dữ liệu đã chia lại (symbolic links) |
| `features_resnet50/` | Vectors đặc trưng (~119 MB) |
| `models_stacking/` | Mô hình đã train (`stacking_classifier.joblib`, `scaler.joblib`, `pca.joblib`) |
| `results_stacking/` | Kết quả đánh giá (`stacking_results.json`) |

---

## 9. Cấu hình kỹ thuật

| Tham số | Giá trị |
|---------|---------|
| ResNet50 | Frozen, ImageNet weights, output 2048-d |
| PCA | 95% variance → 997 components |
| SVM | kernel=RBF, C=10, gamma=scale |
| Random Forest | 300 trees |
| XGBoost | 200 rounds, max_depth=6, lr=0.1 |
| Meta-model | Logistic Regression (class_weight=balanced) |
| Cross-validation | 5-fold (trong Stacking) |
| Random seed | 42 |

---

## 10. Kết luận

| Tiêu chí | Đánh giá |
|-----------|----------|
| Accuracy (95.94%) | ✅ Vượt ngưỡng 90% |
| Recall / Sensitivity (96.31%) | ✅ Vượt ngưỡng lâm sàng 90% |
| Specificity (94.90%) | ✅ Vượt ngưỡng 85% |
| AUC-ROC (0.9888) | ✅ Gần hoàn hảo |
| Overfitting (+0.87%) | ✅ Không overfitting |
| Thời gian train (~22 phút) | ✅ Nhanh, khả thi trên CPU |

**Pipeline ResNet50 + Stacking Ensemble cho kết quả tốt, ổn định, không overfitting, phù hợp làm công cụ hỗ trợ sàng lọc viêm phổi từ ảnh X-quang ngực.**
