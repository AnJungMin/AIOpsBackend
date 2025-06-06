def disease_inference_sequential(image, model_paths, preprocess_funcs, disease_names, device):
    """
    입력 이미지를 각 질환별 모델에 순차적으로 추론하여 결과 반환.
    """
    severity_labels = ["정상", "경증", "중증"]   # ← 3개로 수정!
    results = []
    raw_preds = []

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
