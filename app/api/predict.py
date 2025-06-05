from fastapi import APIRouter, File, UploadFile
from PIL import Image
import io

from app.inference import disease_inference_sequential
from app.preprocess import preprocess_funcs, disease_names, model_paths
from app.config import DEVICE

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    preds = disease_inference_sequential(
        image,
        model_paths,
        preprocess_funcs,
        disease_names,
        DEVICE
    )
    return {"predictions": preds}
