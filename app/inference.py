import torch
from PIL import Image
from app.model import load_model
from app.preprocess import preprocess_funcs, disease_names, model_paths
from app.config import DEVICE
from app.recommendation.utils import get_recommendations_by_disease

def disease_inference_sequential(image, model_paths, preprocess_funcs, disease_names, device):
    severity_labels = ["정상", "경증", "중증"]  # num_classes에 맞게 수정!
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

        # 범위 밖 예측값 대비
        if 0 <= pred_class < len(severity_labels):
            severity = severity_labels[pred_class]
        else:
            severity = "분류불가"

        result = {
            "disease": disease,
            "severity": severity,
            "confidence": f"{confidence:.2f}%"
        }

        # 심각도별 응답 분기
        if pred_class == 0:
            result["comment"] = "정상 범위입니다. 두피 상태가 양호합니다."
        elif pred_class == 1:
            result["recommendations"] = get_recommendations_by_disease(disease)
        elif pred_class == 2:
            result["hospital_recommendation"] = "주변 피부과를 추천합니다. 위치 정보를 기반으로 제공합니다."
        # 혹시 pred_class==3 등 추가 필요시 elif로 대응

        results.append(result)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        "raw_predictions": raw_preds,
        "results": results
    }
