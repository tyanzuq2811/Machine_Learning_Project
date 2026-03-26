from __future__ import annotations

import base64
import io
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = ROOT / "data_resplit" / "split_metadata.json"
FEATURE_META_PATH = ROOT / "features_resnet50" / "features_metadata.json"
RESULT_META_PATH = ROOT / "results_stacking" / "stacking_results.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_to_data_uri(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def find_sample_image() -> Path | None:
    candidates = [
        ROOT / "train" / "NORMAL",
        ROOT / "train" / "PNEUMONIA",
        ROOT / "test" / "NORMAL",
    ]
    for folder in candidates:
        if not folder.exists():
            continue
        files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if files:
            return files[0]
    return None


def build_preprocess_preview(sample_path: Path | None) -> dict:
    if sample_path is None:
        return {}

    gray = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {}

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    resized = cv2.resize(clahe_img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    return {
        "original": image_to_data_uri(gray),
        "clahe": image_to_data_uri(clahe_img),
        "resized": image_to_data_uri(resized),
        "sample_name": sample_path.name,
    }


def build_augmentation_preview(sample_path: Path | None) -> dict:
    if sample_path is None:
        return {}

    gray = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {}

    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    flipped = np.fliplr(resized).copy()

    h, w = resized.shape[:2]
    angle = 10.0
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(resized, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return {
        "original": image_to_data_uri(resized),
        "flipped": image_to_data_uri(flipped),
        "rotated": image_to_data_uri(rotated),
        "sample_name": sample_path.name,
    }


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - min_v) / (max_v - min_v)
    return (scaled * 255.0).clip(0, 255).astype(np.uint8)


def build_glcm_preview(sample_path: Path | None) -> dict:
    if sample_path is None:
        return {}

    gray = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {}

    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    # Quantize grayscale levels (0..31) for texture statistics.
    levels = 32
    q = (resized.astype(np.float32) / 256.0 * levels).astype(np.int32)
    q = np.clip(q, 0, levels - 1)

    diff = np.abs(q[:, 1:] - q[:, :-1]).astype(np.float32)
    contrast_map = diff ** 2
    homogeneity_map = 1.0 / (1.0 + diff)

    contrast_global = float(np.mean(contrast_map))
    homogeneity_global = float(np.mean(homogeneity_map))

    contrast_vis = _normalize_to_uint8(contrast_map)
    homogeneity_vis = _normalize_to_uint8(homogeneity_map)

    contrast_vis = cv2.applyColorMap(contrast_vis, cv2.COLORMAP_INFERNO)
    homogeneity_vis = cv2.applyColorMap(homogeneity_vis, cv2.COLORMAP_OCEAN)

    return {
        "sample_name": sample_path.name,
        "original": image_to_data_uri(resized),
        "contrast": image_to_data_uri(cv2.cvtColor(contrast_vis, cv2.COLOR_BGR2RGB)),
        "homogeneity": image_to_data_uri(cv2.cvtColor(homogeneity_vis, cv2.COLOR_BGR2RGB)),
        "contrast_value": contrast_global,
        "homogeneity_value": homogeneity_global,
    }


@lru_cache(maxsize=24)
def _list_sample_images_cached(split_filter: str, class_filter: str, limit: int) -> tuple[dict, ...]:
    split_values = ["train", "val", "test"] if split_filter == "all" else [split_filter]
    class_values = ["NORMAL", "PNEUMONIA"] if class_filter == "ALL" else [class_filter]

    images: list[dict] = []
    for split_name in split_values:
        for class_name in class_values:
            folder = ROOT / split_name / class_name
            if not folder.exists():
                continue

            files = sorted(
                [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            )
            for f in files:
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                thumb = cv2.resize(img, (280, 280), interpolation=cv2.INTER_AREA)
                images.append(
                    {
                        "split": split_name.upper(),
                        "class": class_name,
                        "name": f.name,
                        "uri": image_to_data_uri(thumb),
                    }
                )
                if len(images) >= limit:
                    return tuple(images)
    return tuple(images)


def list_sample_images(split_filter: str, class_filter: str, limit: int) -> list[dict]:
    return list(_list_sample_images_cached(split_filter, class_filter, limit))


def build_context() -> dict:
    split_meta = load_json(SPLIT_META_PATH)
    feature_meta = load_json(FEATURE_META_PATH)
    result_meta = load_json(RESULT_META_PATH)

    sample = find_sample_image()

    return {
        "split_meta": split_meta,
        "feature_meta": feature_meta,
        "result_meta": result_meta,
        "preprocess_preview": build_preprocess_preview(sample),
        "augmentation_preview": build_augmentation_preview(sample),
        "glcm_preview": build_glcm_preview(sample),
        "original_split": {
            "train": {"NORMAL": 1341, "PNEUMONIA": 3875, "TOTAL": 5216},
            "test": {"NORMAL": 234, "PNEUMONIA": 390, "TOTAL": 624},
            "val": {"NORMAL": 8, "PNEUMONIA": 8, "TOTAL": 16},
        },
    }
