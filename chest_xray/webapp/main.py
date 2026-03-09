"""
WEB APP: CHẨN ĐOÁN VIÊM PHỔI TỪ ẢNH X-QUANG NGỰC
====================================================
Backend FastAPI + ResNet50 (frozen) + PCA + Stacking Ensemble
"""

import os
import io
import sys
import uuid
import numpy as np
import cv2
import joblib
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

# ======================== CẤU HÌNH ========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHEST_XRAY_DIR = os.path.dirname(APP_DIR)  # chest_xray/
MODELS_DIR = os.path.join(CHEST_XRAY_DIR, 'models_stacking')
UPLOAD_DIR = os.path.join(APP_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ======================== LOAD MODELS (1 lần khi khởi động) ========================
print("=" * 60)
print("  LOADING AI MODELS...")
print("=" * 60)

# 1. Load Scaler + PCA + Stacking Classifier
print("📥 Loading StandardScaler...")
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
print("   ✅ Scaler loaded")

print("📥 Loading PCA...")
pca = joblib.load(os.path.join(MODELS_DIR, 'pca.joblib'))
print(f"   ✅ PCA loaded ({pca.n_components_} components)")

print("📥 Loading Stacking Classifier...")
stacking_clf = joblib.load(os.path.join(MODELS_DIR, 'stacking_classifier.joblib'))
print("   ✅ Stacking Classifier loaded")

# 2. Load ResNet50 (frozen feature extractor)
print("📥 Loading ResNet50 (ImageNet, frozen)...")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

resnet_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)
resnet_model.trainable = False
print(f"   ✅ ResNet50 loaded ({resnet_model.output_shape[-1]}-dim output)")

# 3. GLCM
try:
    from skimage.feature import graycomatrix, graycoprops
    HAS_GLCM = True
    print("   ✅ GLCM module available")
except ImportError:
    HAS_GLCM = False
    print("   ⚠️ GLCM not available (skimage not installed)")

print("=" * 60)
print("  ✅ ALL MODELS LOADED SUCCESSFULLY!")
print("=" * 60)

CLASS_NAMES = {0: 'NORMAL', 1: 'PNEUMONIA'}

# ======================== HÀM XỬ LÝ ẢNH ========================

def apply_clahe(image_gray, clip_limit=3.0, tile_grid=(8, 8)):
    """CLAHE tăng tương phản cục bộ cho ảnh X-ray"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image_gray)


def preprocess_image(image_bytes):
    """
    Tiền xử lý ảnh X-ray:
    1. Đọc ảnh → grayscale
    2. CLAHE tăng tương phản
    3. Resize 224x224
    4. Stack 3 kênh
    5. Normalize theo ImageNet
    """
    # Đọc ảnh từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        # Fallback: dùng PIL
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('L')
        img = np.array(pil_img)
    
    # CLAHE
    img_clahe = apply_clahe(img, clip_limit=3.0, tile_grid=(8, 8))
    
    # Resize
    img_resized = cv2.resize(img_clahe, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    
    # 3 kênh
    img_3ch = np.stack([img_resized] * 3, axis=-1).astype(np.float32)
    
    # Normalize theo ImageNet
    img_normalized = preprocess_input(img_3ch)
    
    return img_normalized, img  # Trả thêm ảnh gốc để tính GLCM


def compute_glcm_features(img_gray):
    """Trích xuất 6 đặc trưng GLCM từ ảnh grayscale"""
    if not HAS_GLCM:
        return np.zeros(6, dtype=np.float32)
    
    try:
        img_resized = cv2.resize(img_gray, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        img_quantized = (img_resized / 4).astype(np.uint8)
        
        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(img_quantized, distances=distances, angles=angles,
                           levels=64, symmetric=True, normed=True)
        
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        features = [graycoprops(glcm, p).mean() for p in props]
        return np.array(features, dtype=np.float32)
    except:
        return np.zeros(6, dtype=np.float32)


def predict_single_image(image_bytes):
    """
    Pipeline chẩn đoán đầy đủ:
    1. Tiền xử lý (CLAHE + Resize + Normalize)
    2. ResNet50 → 2048-dim features
    3. GLCM → 6-dim features
    4. Nối → 2054-dim hybrid vector
    5. StandardScaler → chuẩn hóa
    6. PCA → giảm chiều
    7. Stacking Ensemble → dự đoán
    """
    # 1. Tiền xử lý
    img_processed, img_gray = preprocess_image(image_bytes)
    
    # 2. ResNet50 feature extraction
    img_batch = np.expand_dims(img_processed, axis=0)  # (1, 224, 224, 3)
    resnet_features = resnet_model.predict(img_batch, verbose=0)  # (1, 2048)
    
    # 3. GLCM features
    glcm_features = compute_glcm_features(img_gray).reshape(1, -1)  # (1, 6)
    
    # 4. Nối features
    hybrid_features = np.hstack([resnet_features, glcm_features])  # (1, 2054)
    
    # 5. StandardScaler
    features_scaled = scaler.transform(hybrid_features)
    
    # 6. PCA
    features_pca = pca.transform(features_scaled)
    
    # 7. Stacking Ensemble predict
    prediction = stacking_clf.predict(features_pca)[0]
    
    # Xác suất (probability)
    try:
        probabilities = stacking_clf.predict_proba(features_pca)[0]
        confidence = float(probabilities[prediction])
        prob_normal = float(probabilities[0])
        prob_pneumonia = float(probabilities[1])
    except:
        confidence = 1.0
        prob_normal = 1.0 - prediction
        prob_pneumonia = float(prediction)
    
    result = {
        'prediction': CLASS_NAMES[prediction],
        'confidence': round(confidence * 100, 2),
        'probability_normal': round(prob_normal * 100, 2),
        'probability_pneumonia': round(prob_pneumonia * 100, 2),
        'is_pneumonia': bool(prediction == 1),
        'feature_dim_original': int(hybrid_features.shape[1]),
        'feature_dim_pca': int(features_pca.shape[1]),
    }
    
    return result


# ======================== FASTAPI APP ========================

app = FastAPI(
    title="Hệ thống Chẩn đoán Viêm phổi từ X-quang ngực",
    description="ResNet50 + Stacking Ensemble (SVM + RF + XGBoost)",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Trang chủ — Giao diện upload ảnh X-quang"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    API chẩn đoán viêm phổi từ ảnh X-quang.
    
    Input: File ảnh (JPEG, PNG)
    Output: JSON chứa kết quả chẩn đoán
    """
    # Kiểm tra file
    if not file.content_type or not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400,
            content={"error": "Vui lòng tải lên file ảnh (JPEG, PNG)"}
        )
    
    try:
        # Đọc file
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "File ảnh rỗng"}
            )
        
        # Lưu ảnh upload (để hiển thị trên giao diện)
        ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        saved_filename = f"{uuid.uuid4().hex[:12]}.{ext}"
        saved_path = os.path.join(UPLOAD_DIR, saved_filename)
        with open(saved_path, 'wb') as f:
            f.write(image_bytes)
        
        # Chạy pipeline chẩn đoán
        result = predict_single_image(image_bytes)
        result['image_url'] = f"/static/uploads/{saved_filename}"
        result['filename'] = file.filename
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Lỗi khi xử lý ảnh: {str(e)}"}
        )


@app.get("/api/health")
async def health_check():
    """Kiểm tra trạng thái server"""
    return {
        "status": "ok",
        "models_loaded": True,
        "resnet50": "frozen (ImageNet)",
        "classifier": "Stacking Ensemble (SVM + RF + XGBoost)",
        "pca_components": int(pca.n_components_),
    }


# ======================== CHẠY SERVER ========================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting server at http://localhost:8088")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8088)
