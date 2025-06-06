# app/inference.py

import torch
from app.model.model import load_model  # 모델 로드 함수 (반드시 num_classes=3으로 되어 있어야 함)
from app.recommendation.utils import get_recommendations_by_disease  # 제품/병원 추천 함수 등

# ─────────────────────────────────────────────
# 1. 모델 캐시 (서버 기동 시 1회만 로드)
MODEL_CACHE = {}

def preload_models(model_paths, device):
    """
    model_paths 리스트와 device를 받아, 모델을 미리 모두 로딩해 딕셔너리에 저장.
    한 번만 실행되며, 이미 캐시에 있으면 재로딩하지 않음.
    """
    global MODEL_CACHE
    if not MODEL_CACHE:
        for idx, path in enumerate(model_paths):
            model = load_model(path, device)
            model.eval()
            MODEL_CACHE[idx] = model
    return MODEL_CACHE

# ─────────────────────────────────────────────
# 2. 메인 추론 함수
def disease_inference_sequential(image, model_paths, preprocess_funcs, disease_names, device):
    """
    입력 이미지를 각 질환별 모델에 순차적으로 추론하여 결과 반환.
    """
    severity_labels = ["정상", "경증", "중증"]   # ← 3개로 맞춤
    results = []
    raw_preds = []

    # 모델 캐싱 및 사전 로드
    models = preload_models(model_paths, device)

    for idx, (preprocess, disease) in enumerate(zip(preprocess_funcs, disease_names)):
        model = models[idx]
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_class = int(probs.argmax())
            confidence = float(probs[pred_class]) * 100

        raw_preds.append(pred_class)

        severity = severity_labels[pred_class] if 0 <= pred_class < len(severity_labels) else "분류불가"

        result = {
            "disease": disease,
            "severity": severity,
            "confidence": f"{confidence:.2f}%"
        }

        # 결과 해석 및 추가 정보 부여
        if pred_class == 0:
            result["comment"] = "정상 범위입니다. 두피 상태가 양호합니다."
        elif pred_class == 1:  # 경증
            result["recommendations"] = get_recommendations_by_disease(disease)
        elif pred_class == 2:  # 중증
            result["hospital_recommendation"] = "주변 피부과를 추천합니다. 위치 정보를 기반으로 제공합니다."

        results.append(result)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        "raw_predictions": raw_preds,
        "results": results
    }
