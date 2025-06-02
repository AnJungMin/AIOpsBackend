import torch
from PIL import Image
from app.model import load_model  # 절대경로 import로 변경!
from app.preprocess import preprocess_funcs, disease_names, model_paths
from app.train.config import DEVICE
from app.recommendation.utils import get_recommendations_by_disease

def disease_inference_sequential(image, model_paths, preprocess_funcs, disease_names, device):
    severity_labels = ["정상", "경증", "중증"]
    results = []
    raw_preds = []

    for path, preprocess, disease in zip(model_paths, preprocess_funcs, disease_names):
        # 1. 모델 하나만 로드
        model = load_model(path, device)
        model.eval()
        # 2. 전처리
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_class = int(probs.argmax())
            confidence = float(probs[pred_class]) * 100

        raw_preds.append(pred_class)

        result = {
            "disease": disease,
            "severity": severity_labels[pred_class],
            "confidence": f"{confidence:.2f}%"
        }
        # 심각도별 응답 분기
        if pred_class == 0:
            result["comment"] = "정상 범위입니다. 두피 상태가 양호합니다."
        elif pred_class == 1:
            result["recommendations"] = get_recommendations_by_disease(disease)
        elif pred_class == 2:
            result["hospital_recommendation"] = "주변 피부과를 추천합니다. 위치 정보를 기반으로 제공합니다."

        results.append(result)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        "raw_predictions": raw_preds,
        "results": results
    }
