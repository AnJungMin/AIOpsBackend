# 1. Python 베이스 이미지
FROM python:3.10-slim

# 2. 작업 디렉토리 설정 (최상위 /app)
WORKDIR /app

# 3. requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 프로젝트 소스 복사 (app 폴더 전체)
COPY app ./app

# 5. 모델 weight 폴더 생성 (app/model_weight로 이동)
RUN mkdir -p app/model_weight

# 6. pt 파일 다운로드 (app/model_weight 하위에 저장)
RUN wget -O app/model_weight/EfficientNet_B3_BIDDEM_V3.1_2025_05_24_19_model.pt \
    "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_BIDDEM_V3.1_2025_05_24_19_model.pt"
RUN wget -O app/model_weight/EfficientNet_B3_MISE_V4_2025_05_26_18_model.pt \
    "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_MISE_V4_2025_05_26_18_model.pt"
RUN wget -O app/model_weight/EfficientNet_B3_MONO_V4_2025_05_27_16_model.pt \
    "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_MONO_V4_2025_05_27_16_model.pt"
RUN wget -O app/model_weight/EfficientNet_B3_MOSA_V5_TUNNING_2025_06_01_23_final_model.pt \
    "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_MOSA_V5_TUNNING_2025_06_01_23_final_model.pt"
RUN wget -O app/model_weight/EfficientNet_B3_PIZI_V4_2025_05_27_17_model.pt \
    "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_PIZI_V4_2025_05_27_17_model.pt"
RUN wget -O app/model_weight/EfficientNet_B3_TALMO_V5_TUNNING_2025_06_01_16_model.pt \
    "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_TALMO_V5_TUNNING_2025_06_01_16_model.pt"

# 7. uvicorn으로 FastAPI 실행 (app.api.main:app)
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
