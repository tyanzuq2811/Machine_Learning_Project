/**
 * AI X-Ray Diagnosis - Frontend Logic
 * =====================================
 * Xử lý: Upload ảnh, Scanning animation, Gọi API, Hiển thị kết quả
 */

// ======================== DOM ELEMENTS ========================
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadPrompt = document.getElementById('uploadPrompt');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const scanLine = document.getElementById('scanLine');
const btnPredict = document.getElementById('btnPredict');
const btnReset = document.getElementById('btnReset');
const btnText = document.getElementById('btnText');
const loadingState = document.getElementById('loadingState');
const emptyState = document.getElementById('emptyState');
const resultCard = document.getElementById('resultCard');
const loadingStep = document.getElementById('loadingStep');

let selectedFile = null;

// ======================== UPLOAD HANDLING ========================

// Click to upload
dropZone.addEventListener('click', () => fileInput.click());

// File selected
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    // Validate
    if (!file.type.startsWith('image/')) {
        showError('Vui lòng chọn file ảnh (JPEG, PNG)');
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        showError('File quá lớn (tối đa 20MB)');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadPrompt.classList.add('hidden');
        previewArea.classList.remove('hidden');
        fileName.textContent = file.name;
        btnPredict.disabled = false;
    };
    reader.readAsDataURL(file);

    // Reset results
    hideResults();
}

// ======================== PREDICTION ========================

btnPredict.addEventListener('click', async () => {
    if (!selectedFile) return;

    // UI: loading state
    setLoadingState(true);

    // Start scanning animation
    startScan();

    // Simulate pipeline steps in UI
    const steps = [
        'Đang tiền xử lý ảnh (CLAHE + Resize)...',
        'Đang trích xuất đặc trưng ResNet50...',
        'Đang tính GLCM texture features...',
        'Đang chuẩn hóa + giảm chiều PCA...',
        'Đang chạy Stacking Ensemble...',
    ];

    let stepIndex = 0;
    const stepInterval = setInterval(() => {
        stepIndex = (stepIndex + 1) % steps.length;
        loadingStep.textContent = steps[stepIndex];
    }, 1500);

    try {
        // Build FormData
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Call API
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData,
        });

        clearInterval(stepInterval);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Lỗi server');
        }

        const result = await response.json();

        // Stop scan, show results
        stopScan();
        setLoadingState(false);
        displayResult(result);

    } catch (error) {
        clearInterval(stepInterval);
        stopScan();
        setLoadingState(false);
        showError(error.message || 'Không thể kết nối đến server');
    }
});

// ======================== SCANNING ANIMATION ========================

function startScan() {
    scanLine.classList.remove('hidden');
    const container = previewArea.querySelector('.relative.inline-block');
    if (container) container.classList.add('scanning-active');
}

function stopScan() {
    scanLine.classList.add('hidden');
    const container = previewArea.querySelector('.relative.inline-block');
    if (container) container.classList.remove('scanning-active');
}

// ======================== DISPLAY RESULTS ========================

function displayResult(result) {
    emptyState.classList.add('hidden');
    resultCard.classList.remove('hidden');

    const badge = document.getElementById('resultBadge');
    const icon = document.getElementById('resultIcon');
    const label = document.getElementById('resultLabel');
    const conf = document.getElementById('resultConfidence');

    if (result.is_pneumonia) {
        badge.className = 'p-6 rounded-2xl border-2 text-center mb-4 result-pneumonia animate-slide-up';
        icon.textContent = '⚠️';
        label.textContent = 'VIÊM PHỔI (PNEUMONIA)';
        label.className = 'text-2xl font-bold text-red-400';
        conf.textContent = `Độ tin cậy: ${result.confidence}%`;
        conf.className = 'text-sm text-red-300/80';
    } else {
        badge.className = 'p-6 rounded-2xl border-2 text-center mb-4 result-normal animate-slide-up';
        icon.textContent = '✅';
        label.textContent = 'BÌNH THƯỜNG (NORMAL)';
        label.className = 'text-2xl font-bold text-green-400';
        conf.textContent = `Độ tin cậy: ${result.confidence}%`;
        conf.className = 'text-sm text-green-300/80';
    }

    // Probability bars (animate after a short delay)
    setTimeout(() => {
        document.getElementById('probNormal').textContent = result.probability_normal + '%';
        document.getElementById('barNormal').style.width = result.probability_normal + '%';

        document.getElementById('probPneumonia').textContent = result.probability_pneumonia + '%';
        document.getElementById('barPneumonia').style.width = result.probability_pneumonia + '%';
    }, 200);

    // Tech details
    const techDetails = document.getElementById('techDetails');
    techDetails.innerHTML = `
        <div class="grid grid-cols-2 gap-2 mt-2">
            <div class="p-2 rounded bg-dark-700/50">
                <span class="text-gray-400">File:</span>
                <span class="text-gray-300">${result.filename || 'N/A'}</span>
            </div>
            <div class="p-2 rounded bg-dark-700/50">
                <span class="text-gray-400">Feature gốc:</span>
                <span class="text-gray-300">${result.feature_dim_original}-d</span>
            </div>
            <div class="p-2 rounded bg-dark-700/50">
                <span class="text-gray-400">Sau PCA:</span>
                <span class="text-gray-300">${result.feature_dim_pca}-d</span>
            </div>
            <div class="p-2 rounded bg-dark-700/50">
                <span class="text-gray-400">Model:</span>
                <span class="text-gray-300">Stacking Ensemble</span>
            </div>
        </div>
        <div class="mt-2 p-2 rounded bg-dark-700/50">
            <span class="text-gray-400">Pipeline:</span>
            <span class="text-gray-300">CLAHE → Resize 224×224 → ResNet50 → GLCM → Scaler → PCA → SVM+RF+XGB → LogReg</span>
        </div>
    `;
}

function hideResults() {
    resultCard.classList.add('hidden');
    emptyState.classList.remove('hidden');
    // Reset bars
    document.getElementById('barNormal').style.width = '0%';
    document.getElementById('barPneumonia').style.width = '0%';
}

// ======================== UI HELPERS ========================

function setLoadingState(isLoading) {
    if (isLoading) {
        loadingState.classList.remove('hidden');
        emptyState.classList.add('hidden');
        resultCard.classList.add('hidden');
        btnPredict.disabled = true;
        btnPredict.classList.add('btn-loading');
        btnText.textContent = 'Đang phân tích...';
    } else {
        loadingState.classList.add('hidden');
        btnPredict.disabled = false;
        btnPredict.classList.remove('btn-loading');
        btnText.textContent = 'Chẩn đoán AI';
    }
}

function showError(message) {
    emptyState.classList.add('hidden');
    loadingState.classList.add('hidden');
    resultCard.classList.remove('hidden');

    const badge = document.getElementById('resultBadge');
    badge.className = 'p-6 rounded-2xl border-2 text-center mb-4 border-yellow-500/30 bg-yellow-900/10 animate-slide-up';
    document.getElementById('resultIcon').textContent = '❌';
    document.getElementById('resultLabel').textContent = 'Lỗi';
    document.getElementById('resultLabel').className = 'text-2xl font-bold text-yellow-400';
    document.getElementById('resultConfidence').textContent = message;
    document.getElementById('resultConfidence').className = 'text-sm text-yellow-300/80';

    // Hide probability section
    document.getElementById('probNormal').textContent = '-';
    document.getElementById('probPneumonia').textContent = '-';
    document.getElementById('barNormal').style.width = '0%';
    document.getElementById('barPneumonia').style.width = '0%';
    document.getElementById('techDetails').innerHTML = '';
}

// ======================== RESET ========================

btnReset.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadPrompt.classList.remove('hidden');
    previewArea.classList.add('hidden');
    btnPredict.disabled = true;
    stopScan();
    hideResults();
});

// ======================== THEME TOGGLE ========================

const themeToggle = document.getElementById('themeToggle');

themeToggle.addEventListener('click', () => {
    const html = document.documentElement;
    const isDark = html.classList.contains('dark');

    if (isDark) {
        html.classList.remove('dark');
        localStorage.setItem('theme', 'light');
    } else {
        html.classList.add('dark');
        localStorage.setItem('theme', 'dark');
    }
});
