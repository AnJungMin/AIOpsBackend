from fastapi import APIRouter, File, UploadFile
from PIL import Image
import io

from app.inference import disease_inference_sequential  # 순차 추론 함수
from app.preprocess import preprocess_funcs, disease_names, model_paths
from app.config import DEVICE  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    # 순차적으로 모델 로딩/추론
    preds = disease_inference_sequential(
        image,
        model_paths,        # v2.0용 pt파일 이름과 경로가 맞아야 함!
        preprocess_funcs,   # 380x380, mean/std 확인!
        disease_names,      # 질환 순서 일치 확인!
        DEVICE
    )
    return {"predictions": preds}
