"""
BƯỚC 2-4: TIỀN XỬ LÝ ẢNH + TĂNG CƯỜNG + TRÍCH XUẤT ĐẶC TRƯNG
================================================================
Bước 2: Tiền xử lý
  - CLAHE (tăng tương phản vùng phổi)
  - Resize 224x224
  - Grayscale → 3 kênh (R=G=B)
  - Normalize theo chuẩn ImageNet

Bước 3: Tăng cường dữ liệu (chỉ tập Train)
  - Horizontal flip
  - Rotation ±15°
  - KHÔNG vertical flip (lý do y khoa: vòm hoành không bao giờ đảo ngược)

Bước 4: Trích xuất đặc trưng
  - ResNet50 frozen (ImageNet weights) → 2048-dim vector
  - Tùy chọn: GLCM texture features → nối thêm

Output: features_resnet50/ chứa .npy files
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# TF import
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# ======================== CẤU HÌNH ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'data_dir': os.path.join(SCRIPT_DIR, 'data_resplit'),
    'output_dir': os.path.join(SCRIPT_DIR, 'features_resnet50'),
    'target_size': (224, 224),
    'clahe_clip_limit': 3.0,
    'clahe_tile_grid': (8, 8),
    'augment_train': True,
    'rotation_range': 15,  # ±15 độ
    'use_glcm': True,      # True = trích xuất thêm GLCM
    'batch_size': 32,
}

print("=" * 80)
print("BƯỚC 2-4: TIỀN XỬ LÝ + TĂNG CƯỜNG + TRÍCH XUẤT ĐẶC TRƯNG")
print("=" * 80)

# Kiểm tra data đã chia
if not os.path.exists(CONFIG['data_dir']):
    print("Chưa có data_resplit/. Hãy chạy step1_resplit_data.py trước!")
    sys.exit(1)

# ======================== HÀM TIỀN XỬ LÝ ========================

def apply_clahe(image_gray, clip_limit=3.0, tile_grid=(8, 8)):
    """
    CLAHE: Contrast Limited Adaptive Histogram Equalization
    
    Tại sao dùng CLAHE cho X-ray?
    - X-ray thường có contrast thấp ở vùng phổi
    - CLAHE tăng contrast CỤC BỘ (từng vùng nhỏ)
    - Giúp nhìn rõ: vùng mờ đục (viêm phổi), cấu trúc xương sườn
    - Tốt hơn histogram equalization thông thường vì không bị "cháy sáng"
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image_gray)


def preprocess_single_image(img_path, target_size=(224, 224), 
                             clip_limit=3.0, tile_grid=(8, 8)):
    """
    Xử lý 1 ảnh X-ray:
    1. Đọc ảnh grayscale
    2. Áp dụng CLAHE
    3. Resize 224x224
    4. Stack 3 kênh (R=G=B)
    5. Normalize theo ImageNet
    
    Returns: numpy array (224, 224, 3) hoặc None nếu lỗi
    """
    try:
        # 1. Đọc ảnh grayscale
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            # Thử với PIL nếu OpenCV fail
            pil_img = Image.open(img_path).convert('L')
            img_gray = np.array(pil_img)
        
        # 2. CLAHE
        img_clahe = apply_clahe(img_gray, clip_limit, tile_grid)
        
        # 3. Resize
        img_resized = cv2.resize(img_clahe, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 4. Grayscale → 3 kênh giống nhau (R=G=B)
        img_3ch = np.stack([img_resized] * 3, axis=-1).astype(np.float32)
        
        # 5. Normalize theo chuẩn ImageNet (ResNet50 yêu cầu)
        img_normalized = preprocess_input(img_3ch)
        
        return img_normalized
        
    except Exception as e:
        print(f"  ✗ Lỗi {os.path.basename(img_path)}: {e}")
        return None


def compute_glcm_features(img_path, target_size=(224, 224)):
    """
    Trích xuất đặc trưng kết cấu GLCM (Gray-Level Co-occurrence Matrix).
    
    GLCM đo: Dạng kết cấu (texture) của ảnh - thông tin mà CNN có thể bỏ sót.
    
    6 đặc trưng được trích xuất:
    - Contrast: Độ tương phản giữa pixel kề nhau
    - Dissimilarity: Sự khác biệt giữa các pixel
    - Homogeneity: Độ đồng nhất (giá trị gần = 1)
    - Energy: Tính đều đặn của texture
    - Correlation: Tương quan giữa pixel kề
    - ASM: Angular Second Moment (đo trật tự)
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
        
        # Đọc grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            pil_img = Image.open(img_path).convert('L')
            img = np.array(pil_img)
        
        # Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Quantize to 64 levels (giảm noise, tăng tốc)
        img_quantized = (img / 4).astype(np.uint8)
        
        # Tính GLCM (4 hướng: 0°, 45°, 90°, 135°)
        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(img_quantized, distances=distances, angles=angles, 
                           levels=64, symmetric=True, normed=True)
        
        # Trích xuất 6 đặc trưng × trung bình qua các hướng
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        features = []
        for prop in props:
            val = graycoprops(glcm, prop).mean()
            features.append(val)
        
        return np.array(features, dtype=np.float32)
        
    except ImportError:
        return None
    except Exception as e:
        return np.zeros(6, dtype=np.float32)


# ======================== HÀM TĂNG CƯỜNG DỮ LIỆU ========================

def augment_image(img_array, rotation_range=15):
    """
    Tăng cường dữ liệu cho 1 ảnh đã preprocess.
    
    Tạo thêm 2 biến thể:
    1. Horizontal flip (lật ngang)
    2. Random rotation (xoay ±15°)
    
    KHÔNG vertical flip (vòm hoành luôn ở dưới trong X-ray thực tế)
    
    Returns: list of augmented images (numpy arrays)
    """
    augmented = []
    
    # 1. Horizontal flip
    flipped = np.fliplr(img_array).copy()
    augmented.append(flipped)
    
    # 2. Random rotation
    h, w = img_array.shape[:2]
    angle = np.random.uniform(-rotation_range, rotation_range)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    augmented.append(rotated)
    
    return augmented


# ======================== LOAD VÀ XỬ LÝ ========================

def process_split(split_name, data_dir, target_size, augment=False):
    """
    Xử lý toàn bộ 1 split: load ảnh → preprocess → (augment) → return arrays
    """
    X_images = []
    y_labels = []
    glcm_features = []
    
    classes = ['NORMAL', 'PNEUMONIA']
    class_to_label = {'NORMAL': 0, 'PNEUMONIA': 1}
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, split_name, class_name)
        if not os.path.exists(class_dir):
            continue
        
        files = sorted([f for f in os.listdir(class_dir)
                        if f.lower().endswith(('.jpeg', '.jpg', '.png'))
                        and not f.startswith('.')])
        
        label = class_to_label[class_name]
        
        print(f"\n  {class_name}: {len(files)} ảnh")
        
        for fname in tqdm(files, desc=f"    {class_name}", leave=False):
            fpath = os.path.join(class_dir, fname)
            
            # Preprocess
            img = preprocess_single_image(
                fpath, target_size,
                CONFIG['clahe_clip_limit'], CONFIG['clahe_tile_grid']
            )
            
            if img is None:
                continue
            
            X_images.append(img)
            y_labels.append(label)
            
            # GLCM (nếu bật)
            if CONFIG['use_glcm']:
                glcm = compute_glcm_features(fpath, target_size)
                glcm_features.append(glcm if glcm is not None else np.zeros(6))
            
            # Augmentation (CHỈ cho train)
            if augment and split_name == 'train':
                aug_imgs = augment_image(img, CONFIG['rotation_range'])
                for aug_img in aug_imgs:
                    X_images.append(aug_img)
                    y_labels.append(label)
                    if CONFIG['use_glcm']:
                        glcm_features.append(glcm if glcm is not None else np.zeros(6))
    
    X = np.array(X_images, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    
    glcm_arr = None
    if CONFIG['use_glcm'] and glcm_features:
        glcm_arr = np.array(glcm_features, dtype=np.float32)
    
    return X, y, glcm_arr


# ======================== XỬ LÝ TỪNG TẬP ========================

print("\n" + "=" * 80)
print("[2] TIỀN XỬ LÝ ẢNH + TĂNG CƯỜNG DỮ LIỆU")
print("=" * 80)

# --- TRAIN (có augmentation) ---
print(f"\n{'─'*40}")
print(f"Tập TRAIN (có augmentation):")
print(f"{'─'*40}")

X_train, y_train, glcm_train = process_split(
    'train', CONFIG['data_dir'], CONFIG['target_size'], augment=CONFIG['augment_train']
)

n_normal_train = np.sum(y_train == 0)
n_pneumonia_train = np.sum(y_train == 1)
print(f"\n  Train: {X_train.shape[0]} ảnh (sau augmentation)")
print(f"     NORMAL: {n_normal_train}, PNEUMONIA: {n_pneumonia_train}")
print(f"     Shape: {X_train.shape}, Memory: {X_train.nbytes/(1024**3):.2f} GB")

# --- VAL (không augment) ---
print(f"\n{'─'*40}")
print(f"Tập VAL (không augmentation):")
print(f"{'─'*40}")

X_val, y_val, glcm_val = process_split(
    'val', CONFIG['data_dir'], CONFIG['target_size'], augment=False
)
print(f"\n  Val: {X_val.shape[0]} ảnh")

# --- TEST (không augment) ---
print(f"\n{'─'*40}")
print(f"Tập TEST (không augmentation):")
print(f"{'─'*40}")

X_test, y_test, glcm_test = process_split(
    'test', CONFIG['data_dir'], CONFIG['target_size'], augment=False
)
print(f"\n  Test: {X_test.shape[0]} ảnh")


# ======================== TRÍCH XUẤT ĐẶC TRƯNG RESNET50 ========================

print("\n" + "=" * 80)
print("[3] TRÍCH XUẤT ĐẶC TRƯNG VỚI RESNET50")
print("=" * 80)

# Load ResNet50 frozen
print("\nLoading ResNet50 (ImageNet weights, frozen)...")
base_model = ResNet50(
    weights='imagenet',
    include_top=False,        # Bỏ lớp phân loại cuối
    input_shape=(224, 224, 3),
    pooling='avg'             # Global Average Pooling → 2048-dim
)
base_model.trainable = False  # Đóng băng toàn bộ

print(f"     ResNet50 loaded")
print(f"     Layers: {len(base_model.layers)}")
print(f"     Output: {base_model.output_shape[-1]} dimensions")
print(f"     Trainable params: 0 (100% frozen)")


def extract_features_batch(model, X, batch_size=32):
    """Trích xuất features theo batch để tiết kiệm RAM"""
    features = []
    n_batches = int(np.ceil(len(X) / batch_size))
    
    for i in tqdm(range(n_batches), desc="  Extracting features"):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X))
        batch = X[start:end]
        
        batch_features = model.predict(batch, verbose=0)
        features.append(batch_features)
    
    return np.vstack(features)


# Extract features cho cả 3 tập
print("\n Trích xuất features từ Train set...")
feat_train = extract_features_batch(base_model, X_train, CONFIG['batch_size'])
print(f"   Train features: {feat_train.shape}")

print("\n Trích xuất features từ Val set...")
feat_val = extract_features_batch(base_model, X_val, CONFIG['batch_size'])
print(f"   Val features: {feat_val.shape}")

print("\n Trích xuất features từ Test set...")
feat_test = extract_features_batch(base_model, X_test, CONFIG['batch_size'])
print(f"   Test features: {feat_test.shape}")

# Giải phóng RAM (ảnh gốc không cần nữa)
del X_train, X_val, X_test
import gc
gc.collect()

# ======================== NỐI GLCM VÀO RESNET (NẾU CÓ) ========================

if CONFIG['use_glcm'] and glcm_train is not None:
    print(f"\n Nối GLCM features ({glcm_train.shape[1]} dims) vào ResNet50 ({feat_train.shape[1]} dims)...")
    
    feat_train = np.hstack([feat_train, glcm_train])
    feat_val = np.hstack([feat_val, glcm_val])
    feat_test = np.hstack([feat_test, glcm_test])
    
    print(f"   Hybrid vector: {feat_train.shape[1]} dimensions "
          f"({feat_train.shape[1]-6} ResNet50 + 6 GLCM)")

# ======================== PHÂN TÍCH CHẤT LƯỢNG FEATURES ========================

print("\n" + "=" * 80)
print("[4] PHÂN TÍCH CHẤT LƯỢNG FEATURES")
print("=" * 80)

feature_vars = np.var(feat_train, axis=0)
low_var = np.sum(feature_vars < 0.01)
low_var_ratio = low_var / len(feature_vars) * 100

print(f"\n Feature quality:")
print(f"  Total features: {len(feature_vars)}")
print(f"  Low variance (<0.01): {low_var} ({low_var_ratio:.1f}%)")
print(f"  Mean variance: {feature_vars.mean():.4f}")

if low_var_ratio > 40:
    print(f"   CẢNH BÁO: {low_var_ratio:.1f}% features có variance thấp")
else:
    print(f"   Chất lượng features tốt! Chỉ {low_var_ratio:.1f}% low-variance")


# ======================== LƯU KẾT QUẢ ========================

print("\n" + "=" * 80)
print("[5] LƯU FEATURES")
print("=" * 80)

os.makedirs(CONFIG['output_dir'], exist_ok=True)

np.save(os.path.join(CONFIG['output_dir'], 'feat_train.npy'), feat_train)
np.save(os.path.join(CONFIG['output_dir'], 'y_train.npy'), y_train)
np.save(os.path.join(CONFIG['output_dir'], 'feat_val.npy'), feat_val)
np.save(os.path.join(CONFIG['output_dir'], 'y_val.npy'), y_val)
np.save(os.path.join(CONFIG['output_dir'], 'feat_test.npy'), feat_test)
np.save(os.path.join(CONFIG['output_dir'], 'y_test.npy'), y_test)

# Metadata
meta = {
    'config': {k: str(v) if not isinstance(v, (int, float, bool)) else v 
               for k, v in CONFIG.items()},
    'feature_dim': int(feat_train.shape[1]),
    'resnet50_dim': 2048,
    'glcm_dim': 6 if CONFIG['use_glcm'] else 0,
    'shapes': {
        'train': list(feat_train.shape),
        'val': list(feat_val.shape),
        'test': list(feat_test.shape)
    },
    'class_distribution': {
        'train': {'NORMAL': int(np.sum(y_train==0)), 'PNEUMONIA': int(np.sum(y_train==1))},
        'val': {'NORMAL': int(np.sum(y_val==0)), 'PNEUMONIA': int(np.sum(y_val==1))},
        'test': {'NORMAL': int(np.sum(y_test==0)), 'PNEUMONIA': int(np.sum(y_test==1))}
    },
    'low_variance_features_pct': float(low_var_ratio)
}

with open(os.path.join(CONFIG['output_dir'], 'features_metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)

total_mb = (feat_train.nbytes + feat_val.nbytes + feat_test.nbytes) / (1024**2)

print(f"\n Đã lưu tại: {CONFIG['output_dir']}/")
print(f"  feat_train.npy: {feat_train.shape}  ({feat_train.nbytes/(1024**2):.1f} MB)")
print(f"  feat_val.npy:   {feat_val.shape}  ({feat_val.nbytes/(1024**2):.1f} MB)")
print(f"  feat_test.npy:  {feat_test.shape}  ({feat_test.nbytes/(1024**2):.1f} MB)")
print(f"  Tổng: {total_mb:.1f} MB (nhẹ hơn rất nhiều so với lưu ảnh!)")

print(f"\n{'='*80}")
print(" HOÀN THÀNH BƯỚC 2-4!")
print(f"{'='*80}")
print(f"\n Bước tiếp theo:")
print(f"   python chest_xray/step3_train_stacking.py")
print("=" * 80)
