# app/inference.py

import torch
from app.model.model import load_model
from app.recommendation.utils import get_recommendations_by_disease

# 1. 서버 시작 시, 모든 모델을 미리 로드해서 딕셔너리에 저장 (전역)
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

def disease_inference_sequential(image, model_paths, preprocess_funcs, disease_names, device):
    """
    입력 이미지를 각 질환별 모델에 순차적으로 추론하여 결과 반환.
    - image: PIL Image
    - model_paths: 모델 파일 경로 리스트
    - preprocess_funcs: 전처리 함수 리스트
    - disease_names: 질환명 리스트
    - device: 'cpu' 또는 'cuda'
    결과: raw 예측값(클래스), 포맷팅된 결과 리스트 반환
    """
    severity_labels = ["정상", "경증", "중등도", "중증"]
    results = []
    raw_preds = []

    # 모델 사전 로드 (최초 1회)
    models = preload_models(model_paths, device)

    for idx, (preprocess, disease) in enumerate(zip(preprocess_funcs, disease_names)):
        model = models[idx]  # 미리 로드된 모델 사용
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
        elif pred_class in [1, 2]:  # 경증/중등도
            result["recommendations"] = get_recommendations_by_disease(disease)
        elif pred_class == 3:  # 중증
            result["hospital_recommendation"] = "주변 피부과를 추천합니다. 위치 정보를 기반으로 제공합니다."

        results.append(result)

    # (선택) GPU 메모리 관리
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        "raw_predictions": raw_preds,
        "results": results
    }
