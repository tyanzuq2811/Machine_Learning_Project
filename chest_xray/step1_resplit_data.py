"""
BƯỚC 1: KHÁM PHÁ VÀ TÁI CẤU TRÚC DỮ LIỆU (Data Re-splitting)
================================================================
Vấn đề: Tập validation mặc định chỉ có 16 ảnh → vô nghĩa thống kê
Giải pháp: Gộp tất cả → chia lại 80/10/10 với stratified split

QUAN TRỌNG: Nhóm theo Patient ID để tránh data leakage
- PNEUMONIA: "person{ID}_bacteria_xxx.jpeg" hoặc "person{ID}_virus_xxx.jpeg"
- NORMAL: "IM-{ID}-xxxx.jpeg" hoặc "NORMAL2-IM-{ID}-xxxx.jpeg"

Nếu cùng 1 bệnh nhân có ảnh ở cả train/test → rò rỉ dữ liệu!
"""

import os
import re
import shutil
import json
import numpy as np
from collections import defaultdict

# ======================== CẤU HÌNH ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = SCRIPT_DIR

SPLITS = ['train', 'test', 'val']
CLASSES = ['NORMAL', 'PNEUMONIA']

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data_resplit')
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

print("=" * 80)
print("BƯỚC 1: TÁI CẤU TRÚC DỮ LIỆU")
print("=" * 80)

# ======================== THU THẬP DỮ LIỆU ========================
print("\n[1.1] THU THẬP TOÀN BỘ ẢNH")

def extract_patient_id(filename, class_name):
    """
    Trích xuất Patient ID từ tên file.
    
    PNEUMONIA:
      "person1000_bacteria_2931.jpeg" → patient_id = "person_1000"
      "person1000_virus_1681.jpeg"    → patient_id = "person_1000"
    NORMAL:
      "IM-0115-0001.jpeg"             → patient_id = "IM_0115"
      "NORMAL2-IM-1427-0001.jpeg"     → patient_id = "IM_1427"
    """
    if class_name == 'PNEUMONIA':
        match = re.match(r'person(\d+)_', filename)
        if match:
            return f"person_{match.group(1)}"
    else:  # NORMAL
        match = re.search(r'IM-(\d+)', filename)
        if match:
            return f"IM_{match.group(1)}"
    
    # Fallback: dùng tên file làm ID riêng
    return f"unknown_{filename}"

# Thu thập tất cả ảnh
all_images = []  # List of (file_path, class_name, patient_id)
patient_images = defaultdict(list)  # patient_id → list of (file_path, class_name)

for split in SPLITS:
    for class_name in CLASSES:
        class_dir = os.path.join(BASE_PATH, split, class_name)
        if not os.path.exists(class_dir):
            continue
        
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpeg', '.jpg', '.png'))
                 and not f.startswith('.')]
        
        for fname in files:
            fpath = os.path.join(class_dir, fname)
            pid = extract_patient_id(fname, class_name)
            all_images.append((fpath, class_name, pid))
            patient_images[pid].append((fpath, class_name))

print(f"\n📊 Tổng số ảnh: {len(all_images)}")
print(f"📊 Tổng số bệnh nhân (unique patient ID): {len(patient_images)}")

# Thống kê
normal_patients = [pid for pid, imgs in patient_images.items() 
                   if all(c == 'NORMAL' for _, c in imgs)]
pneumonia_patients = [pid for pid, imgs in patient_images.items() 
                      if all(c == 'PNEUMONIA' for _, c in imgs)]
mixed_patients = [pid for pid, imgs in patient_images.items()
                  if len(set(c for _, c in imgs)) > 1]

normal_count = sum(1 for _, c, _ in all_images if c == 'NORMAL')
pneumonia_count = sum(1 for _, c, _ in all_images if c == 'PNEUMONIA')

print(f"\n📈 Phân phối ảnh:")
print(f"  NORMAL:    {normal_count:4d} ảnh ({normal_count/len(all_images)*100:.1f}%)")
print(f"  PNEUMONIA: {pneumonia_count:4d} ảnh ({pneumonia_count/len(all_images)*100:.1f}%)")
print(f"  Tỷ lệ PNEUMONIA:NORMAL = {pneumonia_count/normal_count:.2f}:1")

print(f"\n👤 Phân phối bệnh nhân:")
print(f"  NORMAL patients:    {len(normal_patients)}")
print(f"  PNEUMONIA patients: {len(pneumonia_patients)}")
if mixed_patients:
    print(f"  ⚠️ Mixed patients:   {len(mixed_patients)} (cùng patient có cả 2 lớp)")

# Kiểm tra số ảnh/patient
imgs_per_patient = [len(imgs) for imgs in patient_images.values()]
print(f"\n📸 Ảnh/bệnh nhân:")
print(f"  Min: {min(imgs_per_patient)}, Max: {max(imgs_per_patient)}, "
      f"Mean: {np.mean(imgs_per_patient):.1f}, Median: {np.median(imgs_per_patient):.0f}")

# ======================== CHIA THEO PATIENT ID ========================
print("\n" + "=" * 80)
print("[1.2] CHIA DỮ LIỆU THEO PATIENT ID (Stratified Group Split)")
print("=" * 80)

print(f"\nTỷ lệ: Train {TRAIN_RATIO*100:.0f}% / Val {VAL_RATIO*100:.0f}% / Test {TEST_RATIO*100:.0f}%")
print("Stratified: Giữ tỷ lệ NORMAL/PNEUMONIA trong mỗi tập")
print("Grouped: Ảnh cùng 1 bệnh nhân luôn ở cùng 1 tập (tránh data leakage)")

def stratified_group_split(patient_images, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Chia dữ liệu theo patient ID, stratified theo class.
    Đảm bảo: Cùng patient → cùng tập (train/val/test)
    """
    rng = np.random.RandomState(seed)
    
    # Phân loại patients theo class
    normal_pids = []
    pneumonia_pids = []
    
    for pid, imgs in patient_images.items():
        classes = set(c for _, c in imgs)
        if 'PNEUMONIA' in classes:
            pneumonia_pids.append(pid)
        else:
            normal_pids.append(pid)
    
    # Shuffle
    rng.shuffle(normal_pids)
    rng.shuffle(pneumonia_pids)
    
    def split_list(lst, train_r, val_r):
        n = len(lst)
        n_train = int(n * train_r)
        n_val = int(n * (train_r + val_r))
        return lst[:n_train], lst[n_train:n_val], lst[n_val:]
    
    # Chia từng class riêng
    normal_train, normal_val, normal_test = split_list(normal_pids, train_ratio, val_ratio)
    pneu_train, pneu_val, pneu_test = split_list(pneumonia_pids, train_ratio, val_ratio)
    
    # Gộp
    train_pids = set(normal_train + pneu_train)
    val_pids = set(normal_val + pneu_val)
    test_pids = set(normal_test + pneu_test)
    
    return train_pids, val_pids, test_pids

train_pids, val_pids, test_pids = stratified_group_split(
    patient_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
)

# Phân ảnh theo patient split
train_images, val_images, test_images = [], [], []

for fpath, class_name, pid in all_images:
    if pid in train_pids:
        train_images.append((fpath, class_name))
    elif pid in val_pids:
        val_images.append((fpath, class_name))
    else:
        test_images.append((fpath, class_name))

print(f"\n📊 Kết quả chia (theo patient grouping):")
print(f"  {'Tập':<8} {'Patients':>10} {'Ảnh':>8} {'NORMAL':>10} {'PNEUMONIA':>12} {'Tỷ lệ':>8}")
print(f"  {'-'*56}")

for split_name, split_imgs, split_pids in [
    ('Train', train_images, train_pids),
    ('Val', val_images, val_pids),
    ('Test', test_images, test_pids)
]:
    n_normal = sum(1 for _, c in split_imgs if c == 'NORMAL')
    n_pneumonia = sum(1 for _, c in split_imgs if c == 'PNEUMONIA')
    ratio = n_pneumonia / n_normal if n_normal > 0 else 0
    print(f"  {split_name:<8} {len(split_pids):>10} {len(split_imgs):>8} "
          f"{n_normal:>10} {n_pneumonia:>12} {ratio:>7.2f}:1")

# Kiểm tra KHÔNG có patient trùng giữa các tập
assert train_pids.isdisjoint(val_pids), "Data leakage: Train ∩ Val ≠ ∅"
assert train_pids.isdisjoint(test_pids), "Data leakage: Train ∩ Test ≠ ∅"
assert val_pids.isdisjoint(test_pids), "Data leakage: Val ∩ Test ≠ ∅"
print("\n✅ Kiểm tra data leakage: PASS (không có patient trùng giữa các tập)")

# ======================== COPY ẢNH VÀO THƯ MỤC MỚI ========================
print("\n" + "=" * 80)
print("[1.3] TẠO THƯ MỤC MỚI")
print("=" * 80)

# Xóa output cũ nếu có
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Tạo cấu trúc thư mục
for split_name in ['train', 'val', 'test']:
    for class_name in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split_name, class_name), exist_ok=True)

# Copy ảnh (dùng symlink để tiết kiệm disk)
print(f"\nTạo symbolic links (tiết kiệm dung lượng, không copy thực)...")

link_count = 0
for split_name, split_imgs in [('train', train_images), ('val', val_images), ('test', test_images)]:
    for fpath, class_name in split_imgs:
        fname = os.path.basename(fpath)
        # Nếu trùng tên (từ nhiều split cũ), thêm prefix
        dst = os.path.join(OUTPUT_DIR, split_name, class_name, fname)
        if os.path.exists(dst):
            base, ext = os.path.splitext(fname)
            dst = os.path.join(OUTPUT_DIR, split_name, class_name, f"{base}_dup{link_count}{ext}")
        
        os.symlink(os.path.abspath(fpath), dst)
        link_count += 1

print(f"✅ Đã tạo {link_count} symbolic links")

# ======================== LƯU METADATA ========================
metadata = {
    'total_images': len(all_images),
    'total_patients': len(patient_images),
    'split_ratio': {'train': TRAIN_RATIO, 'val': VAL_RATIO, 'test': TEST_RATIO},
    'random_state': RANDOM_STATE,
    'splits': {}
}

for split_name, split_imgs in [('train', train_images), ('val', val_images), ('test', test_images)]:
    n_normal = sum(1 for _, c in split_imgs if c == 'NORMAL')
    n_pneumonia = sum(1 for _, c in split_imgs if c == 'PNEUMONIA')
    metadata['splits'][split_name] = {
        'total': len(split_imgs),
        'NORMAL': n_normal,
        'PNEUMONIA': n_pneumonia
    }

with open(os.path.join(OUTPUT_DIR, 'split_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n💾 Metadata lưu tại: {OUTPUT_DIR}/split_metadata.json")

# ======================== TỔNG KẾT ========================
print(f"\n{'='*80}")
print("🎉 HOÀN THÀNH BƯỚC 1: TÁI CẤU TRÚC DỮ LIỆU")
print(f"{'='*80}")
print(f"\n📁 Cấu trúc mới: {OUTPUT_DIR}/")
print(f"  ├─ train/  ({len(train_images)} ảnh)")
print(f"  │   ├─ NORMAL/")
print(f"  │   └─ PNEUMONIA/")
print(f"  ├─ val/    ({len(val_images)} ảnh)")
print(f"  │   ├─ NORMAL/")
print(f"  │   └─ PNEUMONIA/")
print(f"  └─ test/   ({len(test_images)} ảnh)")
print(f"      ├─ NORMAL/")
print(f"      └─ PNEUMONIA/")
print(f"\n⚡ Bước tiếp theo:")
print(f"   python chest_xray/step2_preprocess_extract.py")
print("=" * 80)
