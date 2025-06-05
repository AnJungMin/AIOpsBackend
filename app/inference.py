import torch
from PIL import Image
from app.model import load_model
from app.preprocess import preprocess_funcs, disease_names, model_paths
from app.config import DEVICE
from app.recommendation.utils import get_recommendations_by_disease

def disease_inference_sequential(image, model_paths, preprocess_funcs, disease_names, device):
    # merge.ipynb 및 pt 기준 num_classes=4 → severity_labels 4개!
    severity_labels = ["정상", "경증", "중등도", "중증"]
    results = []
    raw_preds = []

    for path, preprocess, disease in zip(model_paths, preprocess_funcs, disease_names):
        model = load_model(path, device)
        model.eval()
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_class = int(probs.argmax())
            confidence = float(probs[pred_class]) * 100

        raw_preds.append(pred_class)

        # 예측값 체크
        if 0 <= pred_class < len(severity_labels):
            severity = severity_labels[pred_class]
        else:
            severity = "분류불가"

        result = {
            "disease": disease,
            "severity": severity,
            "confidence": f"{confidence:.2f}%"
        }

        # 심각도별 응답 분기 (num_classes=4 기준)
        if pred_class == 0:
            result["comment"] = "정상 범위입니다. 두피 상태가 양호합니다."
        elif pred_class == 1 or pred_class == 2:
            # 경증/중등도는 제품 추천
            result["recommendations"] = get_recommendations_by_disease(disease)
        elif pred_class == 3:
            # 중증은 병원 추천
            result["hospital_recommendation"] = "주변 피부과를 추천합니다. 위치 정보를 기반으로 제공합니다."
        # 기타(분류불가 등)는 필요시 elif 추가

        results.append(result)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        "raw_predictions": raw_preds,
        "results": results
    }
