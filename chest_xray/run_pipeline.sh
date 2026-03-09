#!/bin/bash
# ================================================================
# CHẠY TOÀN BỘ PIPELINE: RESNET50 + STACKING ENSEMBLE
# ================================================================
# Sử dụng: bash chest_xray/run_pipeline.sh
# Hoặc:    chmod +x chest_xray/run_pipeline.sh && ./chest_xray/run_pipeline.sh
# ================================================================

set -e  # Dừng nếu có lỗi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "========================================================"
echo "  PIPELINE: CHEST X-RAY PNEUMONIA DETECTION"
echo "  ResNet50 + Stacking Ensemble (SVM + RF + XGBoost)"
echo "========================================================"
echo ""
echo "📁 Script directory: $SCRIPT_DIR"
echo "🕐 Bắt đầu: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Kích hoạt conda
eval "$(conda shell.bash hook)"
conda activate ML
echo "✅ Conda environment: ML"
echo ""

# ======================== BƯỚC 1 ========================
echo "========================================================"
echo "BƯỚC 1/3: Tái cấu trúc dữ liệu (Data Re-splitting)"
echo "========================================================"
python "$SCRIPT_DIR/step1_resplit_data.py"
echo ""

# ======================== BƯỚC 2-4 ========================
echo "========================================================"
echo "BƯỚC 2-4/3: Tiền xử lý + Augmentation + Feature Extraction"
echo "  ⏱️ Có thể mất 15-30 phút"
echo "========================================================"
python "$SCRIPT_DIR/step2_preprocess_extract.py"
echo ""

# ======================== BƯỚC 5-7 ========================
echo "========================================================"
echo "BƯỚC 5-7/3: PCA + Stacking Ensemble + Đánh giá"
echo "  ⏱️ Có thể mất 5-15 phút"
echo "========================================================"
python "$SCRIPT_DIR/step3_train_stacking.py"
echo ""

# ======================== HOÀN THÀNH ========================
echo "========================================================"
echo "🎉 HOÀN THÀNH TOÀN BỘ PIPELINE!"
echo "🕐 Kết thúc: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""
echo "📁 Kết quả:"
echo "  • chest_xray/data_resplit/         - Dữ liệu đã chia lại"
echo "  • chest_xray/features_resnet50/    - Features đã trích xuất"
echo "  • chest_xray/models_stacking/      - Mô hình đã train"
echo "  • chest_xray/results_stacking/     - Kết quả đánh giá"
echo ""
echo "📊 Xem kết quả:"
echo "  cat chest_xray/results_stacking/stacking_results.json"
