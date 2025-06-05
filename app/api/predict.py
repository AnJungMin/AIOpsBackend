# app/api/predict.py

from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
import io

from app.inference import disease_inference_sequential
from app.preprocess import preprocess_funcs, disease_names, model_paths
from app.config import DEVICE

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. 파일 유효성 검사: 이미지 타입 체크
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=400, detail="올바른 이미지 파일이 아닙니다.")

    # 2. 순차 추론 파이프라인 호출
    preds = disease_inference_sequential(
        image,
        model_paths,
        preprocess_funcs,
        disease_names,
        DEVICE
    )
    return {"predictions": preds}
