4"""
BƯỚC 5-7: CHUẨN HÓA + PCA + STACKING ENSEMBLE + ĐÁNH GIÁ
============================================================
Bước 5: StandardScaler + PCA (giữ 95% variance)
Bước 6: Stacking Ensemble
  - Tầng 0: SVM (RBF) + Random Forest + XGBoost
  - Tầng 1: Logistic Regression (meta-model)
Bước 7: Đánh giá trên Test set
  - Confusion Matrix
  - Accuracy, Precision, Recall, F1-Score
  - Focus: Recall (quan trọng nhất cho y tế)
"""

import os
import sys
import json
import numpy as np
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost chưa cài. Sẽ dùng GradientBoosting thay thế.")
    from sklearn.ensemble import GradientBoostingClassifier

# ======================== CẤU HÌNH ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'features_dir': os.path.join(SCRIPT_DIR, 'features_resnet50'),
    'output_dir': os.path.join(SCRIPT_DIR, 'results_stacking'),
    'models_dir': os.path.join(SCRIPT_DIR, 'models_stacking'),
    'pca_variance': 0.95,    # Giữ 95% phương sai
    'random_state': 42,
    'svm_C': 10,
    'svm_gamma': 'scale',
    'rf_n_estimators': 300,
    'xgb_n_estimators': 200,
    'xgb_max_depth': 6,
    'xgb_learning_rate': 0.1,
}

print("=" * 80)
print("BƯỚC 5-7: CHUẨN HÓA + PCA + STACKING + ĐÁNH GIÁ")
print("=" * 80)

# ======================== LOAD FEATURES ========================
print("\n[5.1] LOAD FEATURES")

if not os.path.exists(CONFIG['features_dir']):
    print("❌ Chưa có features_resnet50/. Hãy chạy step2_preprocess_extract.py trước!")
    sys.exit(1)

feat_train = np.load(os.path.join(CONFIG['features_dir'], 'feat_train.npy'))
y_train = np.load(os.path.join(CONFIG['features_dir'], 'y_train.npy'))
feat_val = np.load(os.path.join(CONFIG['features_dir'], 'feat_val.npy'))
y_val = np.load(os.path.join(CONFIG['features_dir'], 'y_val.npy'))
feat_test = np.load(os.path.join(CONFIG['features_dir'], 'feat_test.npy'))
y_test = np.load(os.path.join(CONFIG['features_dir'], 'y_test.npy'))

print(f"\n✅ Loaded features:")
print(f"  Train: {feat_train.shape}")
print(f"  Val:   {feat_val.shape}")
print(f"  Test:  {feat_test.shape}")
print(f"  Feature dim: {feat_train.shape[1]}")

# ======================== CHUẨN HÓA (StandardScaler) ========================
print("\n" + "=" * 80)
print("[5.2] CHUẨN HÓA THANG ĐO (StandardScaler)")
print("=" * 80)

print("""
📖 Tại sao cần StandardScaler?
  - PCA hoạt động dựa trên phương sai (variance)
  - Nếu feature A có giá trị 0-1000, feature B có 0-1
  - PCA sẽ "nghĩ" feature A quan trọng hơn (vì variance lớn hơn)
  - StandardScaler: chuyển mọi feature về mean=0, std=1
  - Sau đó PCA mới công bằng khi so sánh các features
""")

scaler = StandardScaler()

# FIT chỉ trên TRAIN (tránh data leakage!)
feat_train_scaled = scaler.fit_transform(feat_train)
feat_val_scaled = scaler.transform(feat_val)
feat_test_scaled = scaler.transform(feat_test)

print(f"✅ Đã chuẩn hóa:")
print(f"  Train - Mean: {feat_train_scaled.mean():.6f}, Std: {feat_train_scaled.std():.4f}")
print(f"  Val   - Mean: {feat_val_scaled.mean():.6f}, Std: {feat_val_scaled.std():.4f}")
print(f"  Test  - Mean: {feat_test_scaled.mean():.6f}, Std: {feat_test_scaled.std():.4f}")

# ======================== PCA ========================
print("\n" + "=" * 80)
print("[5.3] GIẢM CHIỀU BẰNG PCA (giữ 95% variance)")
print("=" * 80)

print(f"""
📖 PCA (Principal Component Analysis):
  - Vector hiện tại: {feat_train.shape[1]} chiều
  - Nhiều chiều = nhiều nhiễu = dễ overfitting
  - PCA tìm "hướng quan trọng nhất" của dữ liệu
  - Giữ 95% thông tin, bỏ 5% nhiễu
  - Kết quả: Nén từ {feat_train.shape[1]} chiều → vài trăm chiều
""")

pca = PCA(n_components=CONFIG['pca_variance'], random_state=CONFIG['random_state'])

# FIT chỉ trên TRAIN
X_train = pca.fit_transform(feat_train_scaled)
X_val = pca.transform(feat_val_scaled)
X_test = pca.transform(feat_test_scaled)

print(f"✅ PCA giảm chiều:")
print(f"  Trước: {feat_train.shape[1]} dimensions")
print(f"  Sau:   {X_train.shape[1]} dimensions")
print(f"  Giảm:  {(1 - X_train.shape[1]/feat_train.shape[1])*100:.1f}%")
print(f"  Variance giữ lại: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# Giải phóng RAM
del feat_train, feat_val, feat_test
del feat_train_scaled, feat_val_scaled, feat_test_scaled

# ======================== STACKING ENSEMBLE ========================
print("\n" + "=" * 80)
print("[6] XÂY DỰNG STACKING ENSEMBLE CLASSIFIER")
print("=" * 80)

print(f"""
📖 Stacking Ensemble:
  
  ┌─────────────────────────────────────────────┐
  │         TẦNG 0 (Base Models)                │
  │                                             │
  │  ┌─────────┐  ┌──────────┐  ┌──────────┐   │
  │  │  SVM    │  │ Random   │  │ XGBoost  │   │
  │  │ (RBF)  │  │ Forest   │  │          │   │
  │  └────┬────┘  └────┬─────┘  └────┬─────┘   │
  │       │            │             │          │
  └───────┼────────────┼─────────────┼──────────┘
          │            │             │
          ▼            ▼             ▼
  ┌─────────────────────────────────────────────┐
  │         TẦNG 1 (Meta-Model)                 │
  │                                             │
  │       ┌──────────────────────┐              │
  │       │ Logistic Regression  │              │
  │       │   (Trọng tài)       │              │
  │       └──────────┬───────────┘              │
  │                  │                          │
  └──────────────────┼──────────────────────────┘
                     ▼
              "NORMAL" hoặc "PNEUMONIA"
  
  Logistic Regression học cách "sửa sai" cho 3 model trước.
""")

# Tính class weights cho imbalance
n_normal = np.sum(y_train == 0)
n_pneumonia = np.sum(y_train == 1)
total = len(y_train)
class_weight = {0: total/(2*n_normal), 1: total/(2*n_pneumonia)}

print(f"📊 Class weights (xử lý imbalance):")
print(f"  NORMAL:    {class_weight[0]:.4f}")
print(f"  PNEUMONIA: {class_weight[1]:.4f}")

# Định nghĩa base models
print("\n🔧 Cấu hình base models:")

svm = SVC(
    C=CONFIG['svm_C'],
    kernel='rbf',
    gamma=CONFIG['svm_gamma'],
    probability=True,       # Cần cho stacking
    class_weight=class_weight,
    random_state=CONFIG['random_state']
)
print(f"  [1] SVM (RBF): C={CONFIG['svm_C']}, gamma={CONFIG['svm_gamma']}")

rf = RandomForestClassifier(
    n_estimators=CONFIG['rf_n_estimators'],
    max_depth=None,
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=CONFIG['random_state']
)
print(f"  [2] Random Forest: {CONFIG['rf_n_estimators']} trees")

if HAS_XGBOOST:
    xgb = XGBClassifier(
        n_estimators=CONFIG['xgb_n_estimators'],
        max_depth=CONFIG['xgb_max_depth'],
        learning_rate=CONFIG['xgb_learning_rate'],
        scale_pos_weight=n_normal/n_pneumonia,  # Handle imbalance
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=CONFIG['random_state']
    )
    print(f"  [3] XGBoost: {CONFIG['xgb_n_estimators']} rounds, depth={CONFIG['xgb_max_depth']}")
else:
    xgb = GradientBoostingClassifier(
        n_estimators=CONFIG['xgb_n_estimators'],
        max_depth=CONFIG['xgb_max_depth'],
        learning_rate=CONFIG['xgb_learning_rate'],
        random_state=CONFIG['random_state']
    )
    print(f"  [3] GradientBoosting: {CONFIG['xgb_n_estimators']} rounds, depth={CONFIG['xgb_max_depth']}")

# Meta-model
meta_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=CONFIG['random_state']
)
print(f"  [Meta] Logistic Regression (class_weight=balanced)")

# Build Stacking Classifier
estimators = [
    ('svm', svm),
    ('rf', rf),
    ('xgb', xgb)
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5,                    # 5-fold cross-validation cho tầng 0
    stack_method='predict_proba',  # Dùng xác suất thay vì nhãn cứng
    n_jobs=-1,
    passthrough=False        # Chỉ dùng predictions, không dùng features gốc
)

# HUẤN LUYỆN
print(f"\n🚀 Bắt đầu huấn luyện Stacking Ensemble...")
print(f"   (5-fold CV trên {X_train.shape[0]} samples, {X_train.shape[1]} features)")
print(f"   Có thể mất 5-15 phút...")

start_time = time.time()
stacking_clf.fit(X_train, y_train)
train_time = time.time() - start_time

print(f"\n✅ Huấn luyện xong! Thời gian: {train_time/60:.1f} phút")

# ======================== ĐÁNH GIÁ TỪNG MODEL ========================
print("\n" + "=" * 80)
print("[7] ĐÁNH GIÁ MÔ HÌNH")
print("=" * 80)

# Đánh giá từng base model riêng
print("\n📊 Kết quả từng model riêng trên Test set:")
print(f"{'─'*60}")
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"{'─'*60}")

individual_results = {}

for name, model in stacking_clf.named_estimators_.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  {name.upper():<23} {acc*100:>9.2f}% {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}%")
    
    individual_results[name] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1)
    }

# Stacking Ensemble
print(f"{'─'*60}")
y_pred_stack = stacking_clf.predict(X_test)
y_proba_stack = stacking_clf.predict_proba(X_test)[:, 1]

acc_stack = accuracy_score(y_test, y_pred_stack)
prec_stack = precision_score(y_test, y_pred_stack)
rec_stack = recall_score(y_test, y_pred_stack)
f1_stack = f1_score(y_test, y_pred_stack)
auc_stack = roc_auc_score(y_test, y_proba_stack)

print(f"  {'⭐ STACKING':<23} {acc_stack*100:>9.2f}% {prec_stack*100:>9.2f}% "
      f"{rec_stack*100:>9.2f}% {f1_stack*100:>9.2f}%")
print(f"{'─'*60}")

# Val accuracy
y_pred_val = stacking_clf.predict(X_val)
val_acc = accuracy_score(y_val, y_pred_val)
overfitting = (val_acc - acc_stack) * 100

print(f"\n📊 Tổng quan Stacking Ensemble:")
print(f"  Val Accuracy:  {val_acc*100:.2f}%")
print(f"  Test Accuracy: {acc_stack*100:.2f}%")
print(f"  Overfitting:   {overfitting:+.2f}%")
print(f"  AUC-ROC:       {auc_stack:.4f}")

# ======================== CONFUSION MATRIX ========================
print(f"\n{'='*60}")
print("MA TRẬN NHẦM LẪN (Confusion Matrix)")
print(f"{'='*60}")

cm = confusion_matrix(y_test, y_pred_stack)
tn, fp, fn, tp = cm.ravel()

print(f"""
                    Dự đoán
                 NORMAL   PNEUMONIA
Thực tế  NORMAL   {tn:4d}      {fp:4d}     ← False Positive (Cảnh báo nhầm)
      PNEUMONIA   {fn:4d}      {tp:4d}     ← False Negative (⚠️ BỎ SÓT BỆNH!)

📊 Chi tiết:
  True Negative  (TN): {tn:4d} - Đúng NORMAL
  True Positive  (TP): {tp:4d} - Đúng PNEUMONIA
  False Positive (FP): {fp:4d} - Cảnh báo nhầm (người khỏe → bảo bệnh)
  False Negative (FN): {fn:4d} - ⚠️ BỎ SÓT (người bệnh → bảo khỏe)
""")

# ======================== PHÂN TÍCH Y KHOA ========================
print(f"{'='*60}")
print("NHẬN ĐỊNH Y KHOA")
print(f"{'='*60}")

sensitivity = rec_stack  # Recall = Sensitivity
specificity = tn / (tn + fp)

print(f"""
📋 Các chỉ số lâm sàng:

  Sensitivity (Recall): {sensitivity*100:.2f}%
    → Trong {tp+fn} bệnh nhân PNEUMONIA, phát hiện đúng {tp} người
    → Bỏ sót {fn} người bệnh (False Negative)
    {"✅ Tốt (>90%)" if sensitivity > 0.90 else "⚠️ Cần cải thiện (<90%)"}

  Specificity: {specificity*100:.2f}%
    → Trong {tn+fp} người NORMAL, xác định đúng {tn} người
    → Cảnh báo nhầm {fp} người khỏe (False Positive)
    {"✅ Tốt (>85%)" if specificity > 0.85 else "⚠️ Cần cải thiện (<85%)"}

  ⚕️ Nhận định:
    Trong y tế, Recall (Sensitivity) là chỉ số QUAN TRỌNG NHẤT vì:
    - False Negative (bỏ sót bệnh) → Bệnh nhân không được điều trị → NGUY HIỂM
    - False Positive (cảnh báo nhầm) → Chỉ cần chụp lại/khám thêm → CHẤP NHẬN ĐƯỢC
    
    → Mô hình này bỏ sót {fn}/{tp+fn} ca bệnh ({(1-sensitivity)*100:.1f}% FN rate)
""")

# ======================== CLASSIFICATION REPORT ========================
print(f"{'='*60}")
print("CLASSIFICATION REPORT ĐẦY ĐỦ")
print(f"{'='*60}")
print()
print(classification_report(y_test, y_pred_stack, 
                          target_names=['NORMAL', 'PNEUMONIA'],
                          digits=4))

# ======================== LƯU MÔ HÌNH & KẾT QUẢ ========================
print(f"\n{'='*80}")
print("[8] LƯU MÔ HÌNH & KẾT QUẢ")
print(f"{'='*80}")

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['models_dir'], exist_ok=True)

# Lưu models
joblib.dump(stacking_clf, os.path.join(CONFIG['models_dir'], 'stacking_classifier.joblib'))
joblib.dump(scaler, os.path.join(CONFIG['models_dir'], 'scaler.joblib'))
joblib.dump(pca, os.path.join(CONFIG['models_dir'], 'pca.joblib'))

print(f"✅ Models lưu tại: {CONFIG['models_dir']}/")

# Lưu kết quả
results = {
    'stacking_ensemble': {
        'test_accuracy': float(acc_stack),
        'test_precision': float(prec_stack),
        'test_recall': float(rec_stack),
        'test_f1': float(f1_stack),
        'test_auc_roc': float(auc_stack),
        'val_accuracy': float(val_acc),
        'overfitting_pct': float(overfitting),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    },
    'individual_models': individual_results,
    'confusion_matrix': {
        'TN': int(tn), 'FP': int(fp),
        'FN': int(fn), 'TP': int(tp)
    },
    'pca': {
        'original_dim': int(pca.n_features_in_),
        'reduced_dim': int(pca.n_components_),
        'variance_explained': float(pca.explained_variance_ratio_.sum())
    },
    'config': {k: str(v) for k, v in CONFIG.items()},
    'training_time_seconds': float(train_time)
}

with open(os.path.join(CONFIG['output_dir'], 'stacking_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Kết quả lưu tại: {CONFIG['output_dir']}/stacking_results.json")

# ======================== TỔNG KẾT ========================
print(f"\n{'='*80}")
print("🎉 HOÀN THÀNH TOÀN BỘ PIPELINE!")
print(f"{'='*80}")

print(f"""
📊 KẾT QUẢ CUỐI CÙNG:
  ┌────────────────────────────────────────┐
  │  Accuracy:    {acc_stack*100:>6.2f}%                │
  │  Precision:   {prec_stack*100:>6.2f}%                │
  │  Recall:      {rec_stack*100:>6.2f}%  ← Y TẾ         │
  │  F1-Score:    {f1_stack*100:>6.2f}%                │
  │  AUC-ROC:     {auc_stack:>6.4f}                 │
  │  Overfitting: {overfitting:>+6.2f}%                │
  └────────────────────────────────────────┘

💾 Files đã lưu:
  • {CONFIG['models_dir']}/stacking_classifier.joblib
  • {CONFIG['models_dir']}/scaler.joblib
  • {CONFIG['models_dir']}/pca.joblib
  • {CONFIG['output_dir']}/stacking_results.json
""")

print("=" * 80)
