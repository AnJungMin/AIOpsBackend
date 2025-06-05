def disease_inference_sequential(image, model_paths, preprocess_funcs, disease_names, device):
    severity_labels = ["정상", "경증", "중증"]  # ← 클래스 3개
    results = []
    raw_preds = []

    for path, preprocess, disease in zip(model_paths, preprocess_funcs, disease_names):
        model = load_model(path, device)  # 반드시 num_classes=3인 구조여야!
        model.eval()
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_class = int(probs.argmax())
            confidence = float(probs[pred_class]) * 100

        raw_preds.append(pred_class)

        if 0 <= pred_class < len(severity_labels):
            severity = severity_labels[pred_class]
        else:
            severity = "분류불가"

        result = {
            "disease": disease,
            "severity": severity,
            "confidence": f"{confidence:.2f}%"
        }

        # 분기 로직: 클래스 수에 따라 수정!
        if pred_class == 0:
            result["comment"] = "정상 범위입니다. 두피 상태가 양호합니다."
        elif pred_class == 1:
            # 경증 → 제품 추천
            result["recommendations"] = get_recommendations_by_disease(disease)
        elif pred_class == 2:
            # 중증 → 병원 추천
            result["hospital_recommendation"] = "주변 피부과를 추천합니다. 위치 정보를 기반으로 제공합니다."

        results.append(result)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        "raw_predictions": raw_preds,
        "results": results
    }
