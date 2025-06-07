from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import torch

from app.inference import disease_inference_sequential
from app.preprocess import preprocess_funcs, disease_names, model_paths
from app.config import DEVICE

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. 이미지 파일 여부 검사
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    # 2. 이미지 파일 읽기 및 검증
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=400, detail="올바른 이미지 파일이 아닙니다.")

    # 3. 추론 및 에러 처리
    try:
        with torch.no_grad():
            preds = disease_inference_sequential(
                image,
                model_paths,      # 모델 경로 리스트
                preprocess_funcs, # 전처리 리스트
                disease_names,    # 질환명 리스트
                DEVICE
            )
        del image
        torch.cuda.empty_cache()
    except Exception as e:
        print("추론 오류:", e)
        raise HTTPException(status_code=500, detail=f"추론 과정에서 오류 발생: {str(e)}")

    return {"predictions": preds}
