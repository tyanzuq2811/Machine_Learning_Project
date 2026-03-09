# 🏥 HƯỚNG DẪN TOÀN BỘ PIPELINE: PHÁT HIỆN VIÊM PHỔI TỪ ẢNH X-QUANG

> **Dành cho:** Học sinh trung học, sinh viên năm nhất, người mới bắt đầu học AI
> **Mục tiêu:** Hiểu TOÀN BỘ quy trình từ ảnh X-quang thô → kết quả "Bình thường" / "Viêm phổi"

---

## 📌 MỤC LỤC

1. [Bài toán là gì?](#1-bài-toán-là-gì)
2. [Tổng quan Pipeline](#2-tổng-quan-pipeline)
3. [Bước 1: Chia lại dữ liệu](#3-bước-1-chia-lại-dữ-liệu)
4. [Bước 2: Tiền xử lý ảnh](#4-bước-2-tiền-xử-lý-ảnh)
5. [Bước 3: Tăng cường dữ liệu](#5-bước-3-tăng-cường-dữ-liệu)
6. [Bước 4: Trích xuất đặc trưng](#6-bước-4-trích-xuất-đặc-trưng)
7. [Bước 5: Chuẩn hóa và giảm chiều PCA](#7-bước-5-chuẩn-hóa-và-giảm-chiều-pca)
8. [Bước 6: Mô hình Stacking Ensemble](#8-bước-6-mô-hình-stacking-ensemble)
9. [Bước 7: Đánh giá mô hình](#9-bước-7-đánh-giá-mô-hình)
10. [Cách chạy Pipeline](#10-cách-chạy-pipeline)
11. [Từ điển thuật ngữ](#11-từ-điển-thuật-ngữ)

---

## 1. BÀI TOÁN LÀ GÌ?

### 🤔 Câu hỏi đặt ra

Một bác sĩ nhận được ảnh X-quang ngực của bệnh nhân. Liệu máy tính có thể **tự động** phân biệt:
- **NORMAL** (bình thường, phổi sạch)
- **PNEUMONIA** (viêm phổi, phổi có vùng mờ đục)

### 📸 Ảnh X-quang trông như thế nào?

```
     NORMAL (Bình thường)              PNEUMONIA (Viêm phổi)
  ┌──────────────────────┐          ┌──────────────────────┐
  │                      │          │                      │
  │    ┌──┐    ┌──┐      │          │    ┌──┐    ┌──┐      │
  │   ┌┘  └┐  ┌┘  └┐     │          │   ┌┘░░└┐  ┌┘░░└┐     │
  │   │    │  │    │     │          │   │░░░░│  │░░░░│     │
  │   │    │  │    │     │          │   │░░░░│  │░░░░│     │
  │   └┐  ┌┘  └┐  ┌┘     │          │   └┐░░┌┘  └┐░░┌┘     │
  │    └──┘    └──┘      │          │    └──┘    └──┘      │
  │                      │          │                      │
  └──────────────────────┘          └──────────────────────┘
     Phổi sáng, rõ ràng               Phổi mờ (░ = viêm)
```

### 📊 Bộ dữ liệu

| Thông tin | Giá trị |
|-----------|---------|
| **Nguồn** | Kaggle - Chest X-Ray Images (Kermany et al.) |
| **Tổng số ảnh** | 5,856 ảnh |
| **NORMAL** | 1,583 ảnh (27%) |
| **PNEUMONIA** | 4,273 ảnh (73%) |
| **Định dạng** | JPEG |
| **Kích thước gốc** | Đa dạng (từ 400×400 đến 2000×2000 pixel) |

> ⚠️ **Vấn đề:** Dữ liệu bị **mất cân bằng** — PNEUMONIA gấp gần 3 lần NORMAL. Nếu không xử lý, máy sẽ "lười biếng" và đoán hết là PNEUMONIA (đúng 73% mà không cần học gì!)

---

## 2. TỔNG QUAN PIPELINE

**Pipeline** (đường ống) là chuỗi các bước xử lý nối tiếp nhau, đầu ra bước trước là đầu vào bước sau.

```
📸 5,856 ẢNH X-QUANG THÔ
        │
        ▼
┌─────────────────────────────────────────────┐
│  BƯỚC 1: CHIA LẠI DỮ LIỆU                  │
│  Gộp tất cả → Chia 80/10/10 theo bệnh nhân │  ← step1_resplit_data.py
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│  BƯỚC 2: TIỀN XỬ LÝ ẢNH                    │
│  CLAHE + Resize 224×224 + 3 kênh màu       │
│                                             │
│  BƯỚC 3: TĂNG CƯỜNG DỮ LIỆU (chỉ Train)   │  ← step2_preprocess_extract.py
│  Lật ngang + Xoay ±15°                     │
│                                             │
│  BƯỚC 4: TRÍCH XUẤT ĐẶC TRƯNG             │
│  ResNet50 → 2048 số + GLCM → 6 số          │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│  BƯỚC 5: CHUẨN HÓA + PCA                   │
│  StandardScaler + PCA giữ 95% thông tin     │
│                                             │
│  BƯỚC 6: STACKING ENSEMBLE                  │  ← step3_train_stacking.py
│  SVM + Random Forest + XGBoost              │
│  → Logistic Regression (trọng tài)         │
│                                             │
│  BƯỚC 7: ĐÁNH GIÁ MÔ HÌNH                  │
│  Accuracy, Precision, Recall, F1-Score      │
└────────────────────┬────────────────────────┘
                     ▼
          🎯 "NORMAL" hoặc "PNEUMONIA"
```

### ⏱️ Thời gian dự kiến

| Bước | Script | Thời gian |
|------|--------|-----------|
| Bước 1 | step1_resplit_data.py | ~5 giây |
| Bước 2-4 | step2_preprocess_extract.py | 15-30 phút |
| Bước 5-7 | step3_train_stacking.py | 5-15 phút |
| **Tổng** | run_pipeline.sh | **~30-45 phút** |

---

## 3. BƯỚC 1: CHIA LẠI DỮ LIỆU

### 🤔 Tại sao phải chia lại?

Bộ dữ liệu gốc trên Kaggle chia sẵn thành 3 tập:

| Tập | NORMAL | PNEUMONIA | Tổng | Vấn đề |
|-----|--------|-----------|------|--------|
| Train | 1,341 | 3,875 | 5,216 | OK |
| Test | 234 | 390 | 624 | OK |
| **Val** | **8** | **8** | **16** | **⚠️ QUÁ ÍT!** |

Tập **Validation** chỉ có **16 ảnh** → Không đủ để đánh giá mô hình chính xác (giống như kiểm tra toán 1 câu mà bảo biết trình độ cả lớp — vô lý phải không?).

### ✅ Giải pháp: Gộp lại và chia mới

```
GỘP: 5,216 + 624 + 16 = 5,856 ảnh
                │
                ▼
        CHIA LẠI (Stratified)
       ┌────────┼────────┐
       ▼        ▼        ▼
    Train      Val     Test
     80%       10%      10%
   ~4,685    ~586     ~585
```

### 🔑 Hai kỹ thuật quan trọng

#### 1. Stratified Split (Chia phân tầng)

**Vấn đề:** Nếu chia ngẫu nhiên, có thể xảy ra trường hợp tập Test toàn ảnh PNEUMONIA mà không có ảnh NORMAL → đánh giá sai.

**Giải pháp:** Chia **phân tầng** = giữ nguyên tỷ lệ NORMAL/PNEUMONIA (27%/73%) trong mỗi tập.

```
Trước khi chia:    NORMAL 27%  │  PNEUMONIA 73%
                               │
Sau khi chia:                  │
  Train (80%):     NORMAL 27%  │  PNEUMONIA 73%  ✅ Giữ tỷ lệ
  Val (10%):       NORMAL 27%  │  PNEUMONIA 73%  ✅ Giữ tỷ lệ
  Test (10%):      NORMAL 27%  │  PNEUMONIA 73%  ✅ Giữ tỷ lệ
```

#### 2. Patient ID Grouping (Nhóm theo bệnh nhân)

**Vấn đề rò rỉ dữ liệu (Data Leakage):**

Một bệnh nhân có thể chụp **nhiều ảnh** X-quang. Nếu ảnh 1 của bệnh nhân A nằm ở tập Train và ảnh 2 của cùng bệnh nhân A nằm ở tập Test → máy "nhớ mặt" bệnh nhân A thay vì học đặc điểm bệnh → **kết quả ảo tưởng cao!**

**Ví dụ thực tế:**

```
❌ SAI (Không nhóm):
  Train: person5_ảnh1.jpeg, person5_ảnh2.jpeg   ← Cùng bệnh nhân 5!
  Test:  person5_ảnh3.jpeg                       ← Máy "nhớ mặt" → đúng dễ dàng
  
✅ ĐÚNG (Nhóm theo Patient ID):
  Train: person5_ảnh1.jpeg, person5_ảnh2.jpeg, person5_ảnh3.jpeg  ← Tất cả ở Train
  Test:  person8_ảnh1.jpeg                                        ← Bệnh nhân mới hoàn toàn
```

**Cách trích xuất Patient ID:**

| Tên file | Patient ID | Cách đọc |
|----------|-----------|----------|
| `person1000_bacteria_2931.jpeg` | person_1000 | Số sau "person" |
| `person1000_virus_1681.jpeg` | person_1000 | Cùng bệnh nhân, khác loại vi khuẩn |
| `IM-0115-0001.jpeg` | IM_0115 | Số sau "IM-" |
| `NORMAL2-IM-1427-0001.jpeg` | IM_1427 | Số sau "IM-" |

> 📂 **Kết quả:** Tạo thư mục `data_resplit/` chứa `train/`, `val/`, `test/` mới. Dùng **symbolic links** (tên gọi tắt trỏ đến file gốc) để không tốn thêm dung lượng ổ đĩa.

---

## 4. BƯỚC 2: TIỀN XỬ LÝ ẢNH

### 🤔 Tại sao cần tiền xử lý?

Ảnh X-quang gốc rất đa dạng:
- Kích thước khác nhau (400px đến 2000px)
- Độ sáng/tối khác nhau (tùy máy chụp)
- Một số ảnh là grayscale (1 kênh), một số là RGB (3 kênh)

Mạng ResNet50 yêu cầu đầu vào **224×224 pixel, 3 kênh, đã chuẩn hóa**. Nên ta phải biến đổi tất cả ảnh về cùng chuẩn.

### Có 4 thao tác:

### 🔍 2.1. CLAHE — Tăng tương phản thông minh

**CLAHE** = **C**ontrast **L**imited **A**daptive **H**istogram **E**qualization

Tên dài và khó, nhưng ý tưởng rất đơn giản:

**Vấn đề:** Ảnh X-quang thường mờ nhạt. Vùng phổi bị viêm chỉ hơi xám hơn vùng bình thường → khó thấy bằng mắt thường, máy tính cũng khó nhận biết.

**Giải pháp CLAHE:**

```
TRƯỚC CLAHE:                         SAU CLAHE:
┌──────────────────────┐          ┌──────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░ │          │ ▓▓░░░░░░░░░░░░░░▓▓░░ │
│ ░░░░░▒▒▒░░░░░░░░░░░░ │          │ ░░░░░████░░░░░░░░░░░ │
│ ░░░░▒▒▒▒▒░░░░░░░░░░░ │   →     │ ░░░░████████░░░░░░░░ │
│ ░░░░░▒▒▒░░░░░░░░░░░░ │          │ ░░░░░████░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░ │          │ ░░░░░░░░░░░░░░░░░░░░ │
└──────────────────────┘          └──────────────────────┘
  Mờ nhạt, khó thấy viêm           Rõ ràng! Vùng viêm nổi bật
```

**Tại sao "Adaptive" (thích ứng)?**

Thay vì tăng sáng **cả ảnh** (có thể làm mất chi tiết vùng sáng sẵn), CLAHE chia ảnh thành **ô nhỏ 8×8** và tăng tương phản **từng ô riêng biệt**:

```
┌─────┬─────┬─────┬─────┐
│ Ô 1 │ Ô 2 │ Ô 3 │ Ô 4 │   Mỗi ô được điều chỉnh
├─────┼─────┼─────┼─────┤   riêng, phù hợp với
│ Ô 5 │ Ô 6 │ Ô 7 │ Ô 8 │   mức sáng tại ô đó
├─────┼─────┼─────┼─────┤
│ Ô 9 │ Ô10 │ Ô11 │ Ô12 │   → Tăng chi tiết cục bộ
├─────┼─────┼─────┼─────┤      mà không làm "cháy" ô khác
│ Ô13 │ Ô14 │ Ô15 │ Ô16 │
└─────┴─────┴─────┴─────┘
```

**Tại sao "Contrast Limited" (giới hạn)?**

Nếu tăng quá mạnh → ảnh bị nhiễu (noise). CLAHE đặt giới hạn tăng tối đa (`clipLimit = 3.0`) để giữ ảnh tự nhiên.

### 📐 2.2. Resize — Đưa về kích thước chuẩn 224×224

Mạng ResNet50 chỉ nhận ảnh đúng 224×224 pixel:

```
Ảnh gốc (1200×1000)          Sau resize (224×224)
┌────────────────────┐       ┌────────────┐
│                    │       │            │
│                    │  →    │            │
│                    │       │            │
│                    │       └────────────┘
│                    │        Nhỏ gọn, đúng chuẩn
└────────────────────┘
```

Dùng phương pháp **LANCZOS** — thuật toán nội suy cao cấp giữ chi tiết tốt nhất khi thu nhỏ (tốt hơn bilinear, bicubic).

### 🎨 2.3. Chuyển 1 kênh → 3 kênh (Grayscale → RGB)

Ảnh X-quang là **ảnh xám** (1 kênh: chỉ có độ sáng 0-255).
ResNet50 yêu cầu **ảnh màu** (3 kênh: Red, Green, Blue).

**Giải pháp:** Sao chép kênh xám thành 3 kênh giống hệt nhau:

```
Grayscale (1 kênh):     →    RGB (3 kênh):
┌──────────┐                ┌──────────┐ ┌──────────┐ ┌──────────┐
│ 128  200  │                │ 128  200  │ │ 128  200  │ │ 128  200  │
│  50  180  │                │  50  180  │ │  50  180  │ │  50  180  │
└──────────┘                └──────────┘ └──────────┘ └──────────┘
   Kênh duy nhất                Red          Green         Blue
                              (giống nhau!)
```

> **Hỏi:** Tại sao không dùng ảnh 1 kênh luôn?
> **Đáp:** Vì ResNet50 đã được huấn luyện trên ảnh 3 kênh (ImageNet). Nếu đưa ảnh 1 kênh vào, model sẽ "bối rối" vì cấu trúc khác.

### 📏 2.4. Chuẩn hóa ImageNet

Sau khi có ảnh 224×224×3, cần chuẩn hóa giá trị pixel theo chuẩn mà ResNet50 đã học:

```
Pixel gốc: 0 → 255       Sau chuẩn hóa: khoảng -2 → +2
(ví dụ: 128)              (ví dụ: 0.05)
```

Hàm `preprocess_input()` của TensorFlow tự động thực hiện việc này.

---

## 5. BƯỚC 3: TĂNG CƯỜNG DỮ LIỆU

### 🤔 Tại sao cần tăng cường?

**Vấn đề:** Chỉ có ~4,700 ảnh train — khá ít cho machine learning. Ngoài ra, dữ liệu mất cân bằng (PNEUMONIA gấp 3 lần NORMAL).

**Giải pháp:** Tạo thêm ảnh **biến thể** từ ảnh gốc → máy học được nhiều góc nhìn hơn.

### ✅ Hai phép biến đổi được dùng

#### 1. Lật ngang (Horizontal Flip)

```
Ảnh gốc:              Sau lật ngang:
┌──────────┐          ┌──────────┐
│  ┌──┐    │          │    ┌──┐  │
│ ┌┘░░└┐   │    →     │   ┌┘░░└┐ │
│ │░░░░│   │          │   │░░░░│ │
│ └┐░░┌┘   │          │   └┐░░┌┘ │
│  └──┘    │          │    └──┘  │
└──────────┘          └──────────┘
 Vùng viêm bên trái    Vùng viêm bên phải
```

✅ **Hợp lý y khoa:** Viêm phổi có thể ở bên trái hoặc bên phải → Lật ngang vẫn tạo ra ảnh hợp lệ.

#### 2. Xoay nhẹ (Rotation ±15°)

```
Ảnh gốc:              Sau xoay 10°:
┌──────────┐          ┌──────────┐
│  ┌──┐    │          │   ┌──┐   │
│ ┌┘░░└┐   │    →     │ ┌┘░░ └┐  │
│ │░░░░│   │          │  │░░░░│  │
│ └┐░░┌┘   │          │  └┐░░┌┘  │
│  └──┘    │          │   └──┘   │
└──────────┘          └──────────┘
 Thẳng đứng            Nghiêng nhẹ
```

✅ **Hợp lý y khoa:** Bệnh nhân không bao giờ đứng hoàn toàn thẳng khi chụp → Xoay nhẹ tạo ra ảnh giống thực tế.

### ❌ Tại sao KHÔNG lật dọc?

```
Ảnh gốc:              Sau lật dọc:
┌──────────┐          ┌──────────┐
│  Vai      │          │  Bụng    │
│  Phổi    │    →     │  Vòm hoành│
│  Vòm hoành│         │  Phổi    │
│  Bụng    │          │  Vai      │
└──────────┘          └──────────┘
 Đúng giải phẫu        SAI! Vòm hoành ở trên!
```

⚠️ **Lý do y khoa:** **Vòm hoành** (cơ ngăn cách ngực và bụng) **luôn ở phía dưới** phổi. Nếu lật dọc → vòm hoành lên trên → ảnh không tồn tại trong thực tế → máy học sai.

### 📊 Kết quả sau augmentation

| | Trước | Sau (×3: gốc + flip + rotate) |
|---|---|---|
| **Ảnh Train** | ~4,700 | ~14,100 |
| **Ảnh Val** | ~580 | ~580 (không augment) |
| **Ảnh Test** | ~580 | ~580 (không augment) |

> ⚠️ **Quan trọng:** Augmentation **CHỈ áp dụng cho tập Train**. Nếu augment cả Test → đánh giá sai (vì Test phải giống thực tế nhất có thể).

---

## 6. BƯỚC 4: TRÍCH XUẤT ĐẶC TRƯNG

### 🤔 Đặc trưng (Feature) là gì?

**Con người** nhìn ảnh X-quang thấy: "Phổi mờ, có vệt trắng, hình dáng bất thường"

**Máy tính** nhìn ảnh X-quang thấy: "224×224×3 = 150,528 con số từ 0 đến 255"

→ Cần một cách **tóm tắt** 150,528 con số đó thành vài nghìn con số **có ý nghĩa** hơn. Các con số tóm tắt đó gọi là **đặc trưng (features)**.

### 🏗️ ResNet50 — Mạng trích xuất đặc trưng

**ResNet50** là một mạng nơ-ron sâu (Deep Neural Network) với 50 lớp, đã được huấn luyện trên bộ ảnh **ImageNet** (1.4 triệu ảnh, 1000 loại vật thể). Nó đã "học" được cách nhìn ảnh rất tốt.

**Cấu trúc ResNet50:**

```
📸 Ảnh 224×224×3
        │
        ▼
┌───────────────────────────────────┐
│  TẦNG ĐẦU (Layers 1-10)          │
│  Phát hiện: Cạnh, đường thẳng    │
│  /, \, ─, │, ○, □                │
│  (Giống nhau cho MỌI ảnh!)       │
├───────────────────────────────────┤
│  TẦNG GIỮA (Layers 11-30)        │
│  Phát hiện: Hình dạng phức tạp   │
│  Vòng tròn, đường cong, kết cấu  │
├───────────────────────────────────┤
│  TẦNG CUỐI (Layers 31-49)        │
│  Phát hiện: Vật thể cụ thể       │
│  Cấu trúc phổi, vùng sáng/tối   │
├───────────────────────────────────┤
│  GLOBAL AVERAGE POOLING           │
│  Tóm tắt toàn bộ → 2048 con số  │
└───────────────────┬───────────────┘
                    ▼
        Vector 2048 chiều
   (Mỗi ảnh = 1 dãy 2048 con số)
```

**Ví dụ đơn giản:**

```
Ảnh NORMAL:    [0.12, 0.05, 0.87, 0.03, ... , 0.23]  ← 2048 số
Ảnh PNEUMONIA: [0.78, 0.92, 0.15, 0.88, ... , 0.67]  ← 2048 số khác
```

Nhìn vào vector, ta thấy giá trị khác nhau → máy tính có thể phân biệt!

### 🧊 "Đóng băng" (Freeze) là gì?

**Freeze** = Giữ nguyên trọng số của ResNet50, **không train thêm**. 

Tại sao?
- ResNet50 đã học 1.4 triệu ảnh → đủ giỏi nhìn ảnh rồi
- Ta chỉ cần **dùng** kiến thức của nó, không cần dạy thêm
- Tiết kiệm rất nhiều thời gian (30 phút thay vì 5-8 giờ)

### 🧩 GLCM — Đặc trưng kết cấu bổ sung (Tùy chọn)

**GLCM** = **G**ray-**L**evel **C**o-occurrence **M**atrix (Ma trận đồng hiện mức xám)

Hiểu đơn giản: GLCM đo **kết cấu** (texture) của ảnh — bề mặt trơn hay gồ ghề, đều hay lốm đốm.

**6 đặc trưng GLCM:**

| Đặc trưng | Ý nghĩa dễ hiểu | Ví dụ |
|-----------|-----------------|-------|
| **Contrast** | Độ tương phản | Ảnh viêm: contrast cao (vùng mờ + vùng sáng) |
| **Dissimilarity** | Mức khác biệt | Phổi viêm: dissimilarity cao hơn phổi bình thường |
| **Homogeneity** | Độ đồng nhất | Phổi sạch: đồng nhất hơn phổi viêm |
| **Energy** | Tính đều đặn | Phổi sạch: energy cao (pixel đều đặn) |
| **Correlation** | Tương quan | Đo mức liên kết giữa pixel kề nhau |
| **ASM** | Trật tự tổng thể | Phổi bình thường: có trật tự hơn |

### 🔗 Siêu vector "Hybrid"

Nối 2 loại features lại:

```
ResNet50 features + GLCM features = Hybrid vector
   (2048 số)     +    (6 số)      =  (2054 số)
     ↑                   ↑
 Deep Learning     Xử lý ảnh truyền thống
 (hình dạng)       (kết cấu bề mặt)
```

> **Tại sao kết hợp?** ResNet50 giỏi nhìn hình dạng nhưng có thể bỏ sót thông tin kết cấu vi mô. GLCM bổ sung điểm yếu đó.

---

## 7. BƯỚC 5: CHUẨN HÓA VÀ GIẢM CHIỀU PCA

### 📏 5A. Chuẩn hóa StandardScaler

**Vấn đề:** 2054 đặc trưng có **biên độ khác nhau**:
- Feature A: giá trị từ 0 đến 0.001
- Feature B: giá trị từ 0 đến 1000

Nếu đưa thẳng vào PCA hoặc SVM, chúng sẽ "nghĩ" Feature B quan trọng hơn 1 triệu lần (vì số lớn hơn) — **SAI!**

**Giải pháp: Z-score Normalization**

$$z = \frac{x - \mu}{\sigma}$$

Trong đó:
- $x$ = giá trị gốc
- $\mu$ = giá trị trung bình (mean) của feature đó
- $\sigma$ = độ lệch chuẩn (standard deviation)
- $z$ = giá trị đã chuẩn hóa

**Ví dụ:**

```
Feature A (gốc): [0.001, 0.003, 0.002]  →  Feature A (chuẩn hóa): [-1.22, 1.22, 0.0]
Feature B (gốc): [100, 300, 200]         →  Feature B (chuẩn hóa): [-1.22, 1.22, 0.0]
```

Sau chuẩn hóa: Mọi feature đều có **mean = 0, std = 1** → Công bằng!

> ⚠️ **Quan trọng:** `fit()` chỉ trên tập Train, rồi `transform()` Val và Test. Nếu fit trên cả Test → data leakage (vì máy đã "nhìn trước" Test).

### 📊 5B. PCA — Giảm chiều dữ liệu

**PCA** = **P**rincipal **C**omponent **A**nalysis (Phân tích thành phần chính)

**Vấn đề:** Vector 2054 chiều quá nhiều:
- Nhiều chiều chứa **nhiễu** (noise) — thông tin rác
- Thuật toán ML dễ bị **overfitting** (học thuộc thay vì hiểu)
- Tốn thời gian tính toán

**Giải pháp PCA:** Tìm "hướng quan trọng nhất" và bỏ hướng kém quan trọng.

**Ví dụ trực quan (2D → 1D):**

```
      y
      │    * *
      │  *  *  *
      │ * * * *          PCA tìm hướng "trải dài" nhất
      │* * * *     →     rồi chiếu tất cả điểm lên hướng đó
      │ * * *
      │* *
      └──────── x        Kết quả: Giảm từ 2D → 1D
                         mà giữ được phần lớn thông tin
```

**Trong project này:**
- Đầu vào: 2054 chiều
- PCA giữ lại: **95% phương sai** (95% thông tin)
- Đầu ra: thường **~200-400 chiều** (tùy dữ liệu)

```
TRƯỚC PCA:  [0.12, -0.34, 0.78, ..., 0.56]  ← 2054 số
                                                 ↓ PCA (giữ 95% thông tin)
SAU PCA:    [1.23, -0.67, 2.34, ..., 0.89]  ← ~300 số (ước lượng)

Giảm: 2054 → ~300 = giảm ~85% kích thước!
```

**Lợi ích:**
1. ✅ Loại bỏ nhiễu (5% noise bị bỏ)
2. ✅ Chống overfitting (ít chiều = ít "thuộc bài")
3. ✅ Tăng tốc train (300 chiều nhanh hơn 2054 chiều rất nhiều)

---

## 8. BƯỚC 6: MÔ HÌNH STACKING ENSEMBLE

### 🤔 Ensemble là gì?

**Ensemble** (kết hợp) = Dùng **nhiều mô hình** cùng giải 1 bài toán, rồi lấy kết quả tổng hợp.

**Ví dụ thực tế:** Khi bạn bị ốm, bạn hỏi:
- Bác sĩ A nói: "Cảm cúm"
- Bác sĩ B nói: "Viêm phổi"
- Bác sĩ C nói: "Viêm phổi"

→ Bạn tin **đa số**: "Viêm phổi" (2/3 phiếu) → Chính xác hơn hỏi 1 bác sĩ!

### 🏗️ Stacking Ensemble — 2 tầng

**Stacking** = Không chỉ "bỏ phiếu" đơn giản, mà có **trọng tài** thông minh:

```
┌───────────────────────────────────────────────────────┐
│                   TẦNG 0: BA "GIÁM KHẢO"              │
│                                                       │
│   ┌─────────────┐ ┌───────────────┐ ┌──────────────┐  │
│   │   🔵 SVM    │ │  🟢 Random    │ │  🟡 XGBoost  │  │
│   │  (Nhân RBF) │ │    Forest     │ │              │  │
│   │             │ │  (300 cây)    │ │ (200 vòng)   │  │
│   │  "Tôi nghĩ │ │  "Tôi nghĩ   │ │ "Tôi nghĩ   │  │
│   │  PNEUMONIA  │ │  NORMAL       │ │ PNEUMONIA    │  │
│   │  (85%)"     │ │  (60%)"       │ │ (92%)"       │  │
│   └──────┬──────┘ └──────┬────────┘ └──────┬───────┘  │
│          │               │                 │          │
└──────────┼───────────────┼─────────────────┼──────────┘
           │               │                 │
           ▼               ▼                 ▼
┌───────────────────────────────────────────────────────┐
│              TẦNG 1: "TRỌNG TÀI" THÔNG MINH           │
│                                                       │
│   Logistic Regression nhận: [85%, 60%, 92%]           │
│                                                       │
│   Học được: "XGBoost giỏi hơn Random Forest           │
│   trong trường hợp này, nên tin XGBoost hơn"          │
│                                                       │
│   → Kết luận: PNEUMONIA (tin cao 90%)                 │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### 📚 Chi tiết 3 mô hình Tầng 0

#### 🔵 SVM (Support Vector Machine) — Nhân RBF

**Ý tưởng:** Tìm đường ranh giới **tốt nhất** để tách 2 lớp NORMAL vs PNEUMONIA.

```
                  ┌── Đường ranh giới (hyperplane)
                  │
  NORMAL          │         PNEUMONIA
    ○   ○         │      ●  ●  ●
      ○  ○  ○     │    ●  ●  ●
    ○  ○  ○       │      ●  ●
      ○           │    ●     ●
                  │
  SVM tìm đường thẳng tách 2 nhóm xa nhất có thể
```

**RBF (Radial Basis Function):** Khi 2 lớp không thể tách bằng đường thẳng, RBF "uốn cong" không gian để tách được.

**Ưu điểm:** Rất mạnh khi dữ liệu ít, chính xác cao.
**Nhược điểm:** Chậm khi dữ liệu lớn.

#### 🟢 Random Forest — 300 cây quyết định

**Ý tưởng:** Tạo 300 "cây hỏi-đáp" khác nhau, mỗi cây nhìn một phần dữ liệu.

```
Cây 1:                    Cây 2:                    Cây 300:
Feature 45 > 0.5?         Feature 102 > -0.3?       Feature 7 > 0.8?
  ├─ Yes → PNEUMONIA      ├─ Yes → Feature 88?      ├─ Yes → ...
  └─ No  → Feature 12?    │   ├─ > 0.1 → NORMAL     └─ No  → ...
           ├─ ...          │   └─ ≤ 0.1 → PNEUMONIA
           └─ ...          └─ No → PNEUMONIA

Kết quả: Bỏ phiếu từ 300 cây → Lớp nào nhiều phiếu hơn thắng!
```

**Ưu điểm:** Ít overfitting, hoạt động tốt "out of the box".
**Nhược điểm:** Khó giải thích cụ thể tại sao dự đoán X.

#### 🟡 XGBoost — Cây tăng cường gradient

**Ý tưởng:** Xây dựng cây **tuần tự** — mỗi cây mới **sửa sai** cho cây trước.

```
Vòng 1: Cây 1 dự đoán       → Sai 20%
Vòng 2: Cây 2 sửa sai Cây 1 → Sai 12%
Vòng 3: Cây 3 sửa sai Cây 2 → Sai 8%
...
Vòng 200: Tổng hợp 200 cây  → Sai 3%!
```

**Ưu điểm:** Thường cho kết quả tốt nhất trong ML truyền thống, thắng nhiều cuộc thi Kaggle.
**Nhược điểm:** Dễ overfitting nếu không điều chỉnh tham số tốt.

### 🏆 Tầng 1: Logistic Regression (Trọng tài)

**Logistic Regression** nhận kết quả từ 3 model Tầng 0 và **học** trọng số tối ưu:

$$P(\text{PNEUMONIA}) = \sigma(w_1 \cdot p_{SVM} + w_2 \cdot p_{RF} + w_3 \cdot p_{XGB} + b)$$

Trong đó:
- $p_{SVM}, p_{RF}, p_{XGB}$ = xác suất dự đoán PNEUMONIA từ 3 model
- $w_1, w_2, w_3$ = trọng số (model nào giỏi hơn → trọng số cao hơn)
- $\sigma$ = hàm sigmoid (biến đổi về khoảng 0-1)
- $b$ = bias

**Kết quả:** Nếu $P > 0.5$ → PNEUMONIA, ngược lại → NORMAL.

### ⚖️ Xử lý mất cân bằng (Class Weights)

Vì PNEUMONIA gấp ~3 lần NORMAL, ta dùng **class weights**:

```
NORMAL:    weight = 1.94  (tăng lên → "coi trọng" NORMAL hơn)
PNEUMONIA: weight = 0.67  (giảm xuống)
```

Khi máy dự đoán SAI 1 ảnh NORMAL → bị phạt **1.94 điểm** (nặng)
Khi máy dự đoán SAI 1 ảnh PNEUMONIA → bị phạt **0.67 điểm** (nhẹ hơn)

→ Máy sẽ cố gắng **không bỏ sót** ảnh NORMAL (vì bị phạt nặng).

---

## 9. BƯỚC 7: ĐÁNH GIÁ MÔ HÌNH

### 📊 4 Chỉ số chính

Giả sử có 100 ảnh Test, model dự đoán:

```
                    Model dự đoán
                 NORMAL   PNEUMONIA
Thực tế  NORMAL    22        5        ← 5 người khỏe bị "cảnh báo nhầm"
      PNEUMONIA     3       70        ← 3 người bệnh bị "bỏ sót" ⚠️
```

Từ bảng này, ta tính:

#### 1. Accuracy (Độ chính xác tổng thể)

$$\text{Accuracy} = \frac{\text{Đoán đúng}}{\text{Tổng}} = \frac{22 + 70}{100} = 92\%$$

**Nghĩa là:** Trong 100 ảnh, máy đoán đúng 92 ảnh.

#### 2. Precision (Độ chính xác khi nói "Viêm phổi")

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{70}{70 + 5} = 93.3\%$$

**Nghĩa là:** Khi máy nói "người này bị viêm phổi", 93.3% trường hợp là đúng. 7% bị cảnh báo nhầm (False Positive).

#### 3. Recall / Sensitivity (Độ nhạy — **QUAN TRỌNG NHẤT cho y tế**)

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{70}{70 + 3} = 95.9\%$$

**Nghĩa là:** Trong 73 người bệnh thực sự, máy phát hiện được 70 người. **Bỏ sót 3 người.**

#### 4. F1-Score (Điểm cân bằng)

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.933 \times 0.959}{0.933 + 0.959} = 94.6\%$$

**Nghĩa là:** Điểm trung bình hài hòa giữa Precision và Recall.

### ⚕️ Tại sao Recall quan trọng NHẤT trong y tế?

Có 2 loại sai lầm:

| Loại sai | Tên gọi | Hậu quả |
|----------|---------|---------|
| **Cảnh báo nhầm** | False Positive (FP) | Người khỏe bị bảo "bệnh" → Chụp thêm → **Phiền nhưng AN TOÀN** |
| **Bỏ sót bệnh** | False Negative (FN) | Người bệnh bị bảo "khỏe" → Không điều trị → **NGUY HIỂM TÍNH MẠNG** ⚠️ |

→ Trong y tế: **Thà cảnh báo nhầm 100 lần còn hơn bỏ sót 1 ca bệnh.**

→ Recall (phát hiện được bao nhiêu ca bệnh) **quan trọng hơn** Precision.

### 📉 Overfitting — Máy "học thuộc" thay vì "hiểu"

**Overfitting** = Model thuộc lòng tập Train nhưng không giải được tập Test mới.

```
Ví dụ:
  Val Accuracy:  95% ← Tốt trên tập đã thấy
  Test Accuracy: 85% ← Kém trên tập mới
  Chênh lệch:   10% = OVERFITTING CAO!

Mong muốn:
  Val Accuracy:  93%
  Test Accuracy: 92%
  Chênh lệch:   1% = ✅ Tốt! Model "hiểu" thay vì "thuộc"
```

**PCA** và **Stacking** giúp giảm overfitting rất hiệu quả.

---

## 10. CÁCH CHẠY PIPELINE

### 🚀 Chạy tự động (khuyến khích)

```bash
# Kích hoạt môi trường
conda activate ML

# Chạy toàn bộ pipeline
bash chest_xray/run_pipeline.sh
```

### 🔧 Chạy từng bước (nếu muốn kiểm tra)

```bash
conda activate ML

# Bước 1: Chia lại dữ liệu (~5 giây)
python chest_xray/step1_resplit_data.py

# Bước 2-4: Tiền xử lý + Trích xuất (15-30 phút)
python chest_xray/step2_preprocess_extract.py

# Bước 5-7: PCA + Stacking + Đánh giá (5-15 phút)
python chest_xray/step3_train_stacking.py
```

### 📁 Cấu trúc output

```
chest_xray/
├── data_resplit/              ← Bước 1: Dữ liệu đã chia lại
│   ├── train/NORMAL/
│   ├── train/PNEUMONIA/
│   ├── val/NORMAL/
│   ├── val/PNEUMONIA/
│   ├── test/NORMAL/
│   ├── test/PNEUMONIA/
│   └── split_metadata.json
│
├── features_resnet50/         ← Bước 2-4: Features đã trích xuất
│   ├── feat_train.npy
│   ├── feat_val.npy
│   ├── feat_test.npy
│   ├── y_train.npy, y_val.npy, y_test.npy
│   └── features_metadata.json
│
├── models_stacking/           ← Bước 6: Mô hình đã train
│   ├── stacking_classifier.joblib
│   ├── scaler.joblib
│   └── pca.joblib
│
└── results_stacking/          ← Bước 7: Kết quả đánh giá
    └── stacking_results.json
```

---

## 11. TỪ ĐIỂN THUẬT NGỮ

Bảng giải thích nhanh các thuật ngữ theo thứ tự A-Z:

| Thuật ngữ | Tiếng Việt | Giải thích đơn giản |
|-----------|-----------|---------------------|
| **Accuracy** | Độ chính xác | Bao nhiêu % dự đoán đúng trong tổng số |
| **Augmentation** | Tăng cường dữ liệu | Tạo thêm ảnh biến thể (lật, xoay) để máy học nhiều hơn |
| **Batch** | Lô/Gói | Chia dữ liệu thành nhóm nhỏ để xử lý dần (tiết kiệm RAM) |
| **Class Weight** | Trọng số lớp | Phạt nặng hơn khi đoán sai lớp ít ảnh (cân bằng dữ liệu) |
| **CLAHE** | Tăng tương phản thích ứng | Làm rõ chi tiết ảnh theo từng vùng nhỏ |
| **CNN** | Mạng nơ-ron tích chập | Loại AI chuyên xử lý ảnh, "nhìn" bằng cách quét từng vùng nhỏ |
| **Confusion Matrix** | Ma trận nhầm lẫn | Bảng 2×2 cho biết model đoán đúng/sai bao nhiêu |
| **Data Leakage** | Rò rỉ dữ liệu | Khi thông tin Test "lọt" vào Train → kết quả ảo tưởng cao |
| **Ensemble** | Kết hợp mô hình | Dùng nhiều model cùng giải 1 bài → kết quả tốt hơn |
| **F1-Score** | Điểm F1 | Trung bình hài hòa giữa Precision và Recall |
| **False Negative** | Âm tính giả | Người bệnh bị đoán nhầm là khỏe (**nguy hiểm!**) |
| **False Positive** | Dương tính giả | Người khỏe bị đoán nhầm là bệnh (phiền nhưng an toàn) |
| **Feature** | Đặc trưng | Các con số tóm tắt thông tin quan trọng của ảnh |
| **Freeze** | Đóng băng | Giữ nguyên trọng số model, không cho thay đổi |
| **GLCM** | Ma trận đồng hiện | Đo kết cấu (texture) bề mặt ảnh: trơn hay gồ ghề |
| **Grayscale** | Ảnh xám | Ảnh chỉ có 1 kênh (0=đen, 255=trắng) |
| **ImageNet** | - | Bộ dữ liệu 1.4 triệu ảnh, 1000 loại vật thể |
| **LANCZOS** | - | Thuật toán resize ảnh chất lượng cao |
| **Normalize** | Chuẩn hóa | Đưa giá trị về cùng thang đo (ví dụ: mean=0, std=1) |
| **Overfitting** | Quá khớp | Model "thuộc bài" tập Train nhưng không giải được bài mới |
| **Patient ID** | Mã bệnh nhân | Định danh duy nhất cho mỗi bệnh nhân |
| **PCA** | Phân tích thành phần chính | Nén dữ liệu nhiều chiều xuống ít chiều, giữ thông tin quan trọng |
| **Pipeline** | Đường ống | Chuỗi các bước xử lý nối tiếp nhau |
| **Precision** | Độ chính xác dương | Khi nói "bệnh", bao nhiêu % thực sự bệnh |
| **Random Forest** | Rừng ngẫu nhiên | 300 cây quyết định bỏ phiếu chung |
| **Recall** | Độ nhạy | Phát hiện được bao nhiêu % ca bệnh thực sự |
| **ResNet50** | - | Mạng nơ-ron 50 lớp, đã học 1.4 triệu ảnh ImageNet |
| **RGB** | Đỏ-Xanh lá-Xanh dương | Ảnh 3 kênh màu |
| **RBF Kernel** | Nhân hàm cơ sở xuyên tâm | Kỹ thuật giúp SVM tách dữ liệu phi tuyến |
| **Sigmoid** | Hàm S | Hàm toán học biến mọi số thành giá trị từ 0 đến 1 |
| **Specificity** | Độ đặc hiệu | Xác định đúng bao nhiêu % người khỏe |
| **Stacking** | Xếp chồng | Dùng kết quả nhiều model này làm đầu vào cho model khác |
| **StandardScaler** | Chuẩn hóa Z-score | Đưa mọi feature về mean=0, std=1 |
| **Stratified Split** | Chia phân tầng | Giữ nguyên tỷ lệ lớp khi chia dữ liệu |
| **SVM** | Máy vector hỗ trợ | Tìm đường ranh giới tốt nhất giữa 2 lớp |
| **Symbolic Link** | Liên kết tượng trưng | "Tên gọi tắt" trỏ đến file gốc (không copy thêm) |
| **Variance** | Phương sai | Đo mức "trải rộng" của dữ liệu |
| **XGBoost** | Tăng cường gradient cực độ | Thuật toán xây dựng cây tuần tự, mỗi cây sửa sai cây trước |

---

## ❓ CÂU HỎI THƯỜNG GẶP

### Q1: Tại sao dùng ResNet50 mà không dùng EfficientNet B4/B7?

**A:** EfficientNet B4/B7 **mạnh hơn** nhưng:
- B7: 600×600 input → 24GB dữ liệu → 10-20 giờ train
- B4: 380×380 input → 10GB dữ liệu → 5-8 giờ train
- **ResNet50: 224×224 input → ~200MB features → 30-45 phút!**

ResNet50 nhỏ gọn, nhanh, và đủ tốt cho bài toán 2 lớp đơn giản.

### Q2: Tại sao không train ResNet50 mà chỉ "đóng băng"?

**A:** Vì chỉ có ~5,000 ảnh. Nếu train (fine-tune) ResNet50 (25 triệu tham số) trên 5,000 ảnh → **overfitting cực nặng** (model thuộc bài). Đóng băng + ML classifier là cách an toàn nhất.

### Q3: Stacking có tốt hơn chỉ dùng 1 model không?

**A:** Gần như **luôn tốt hơn** vì:
- SVM giỏi tách linear
- Random Forest giỏi bắt non-linear patterns
- XGBoost giỏi tối ưu hóa tổng thể
- Kết hợp = bù nhược điểm cho nhau

### Q4: Có thể cải thiện thêm không?

**A:** Có! Một số hướng:
1. **Thêm dữ liệu** từ các bệnh viện khác
2. **Ensemble nặng hơn**: Thêm LightGBM, CatBoost
3. **Fine-tune ResNet50** nếu có GPU mạnh
4. **Dùng DenseNet121** thay ResNet50 (phổ biến trong y khoa)
5. **Threshold optimization**: Điều chỉnh ngưỡng 0.5 → 0.3 để tăng Recall

### Q5: Model này có thay thế bác sĩ được không?

**A:** **KHÔNG!** Model chỉ là **công cụ hỗ trợ**:
- Recall ~95% → vẫn bỏ sót 5% ca bệnh
- Không hiểu bối cảnh lâm sàng (triệu chứng, tiền sử)
- Bác sĩ vẫn phải xem xét kết quả cuối cùng

---

*Tài liệu được tạo cho project Nhận diện Viêm phổi từ ảnh X-quang ngực.*
*Pipeline: ResNet50 (Frozen) + GLCM + PCA + Stacking Ensemble (SVM + RF + XGBoost).*
