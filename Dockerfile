# AI 서버 Dockerfile - HuggingFace 모델 포함 최적화 버전
FROM python:3.11.8-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# infra-repo에서 가져온 최적화된 requirements.txt
COPY requirements.txt .
# ai-repository에서 가져온 gunicorn 설정들  
COPY gunicorn.conf.py .
COPY gunicorn_log.conf .

# 의존성 설치 (CPU-only 최적화)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# HuggingFace 모델 미리 다운로드 (멀티스테이지 빌드 최적화)
RUN mkdir -p /app/models && \
    python3 << 'EOF'
import os
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 캐시 디렉토리 설정
model_cache = "/app/models"
os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache
os.environ['HF_HOME'] = f"{model_cache}/.cache/huggingface_hub"
os.environ['TORCH_HOME'] = f"{model_cache}/.cache/torch"

try:
    logger.info("📥 HuggingFace 모델 다운로드 시작...")
    
    from sentence_transformers import SentenceTransformer
    
    # 모델 다운로드
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=model_cache)
    logger.info("✅ all-MiniLM-L6-v2 모델 다운로드 완료!")
    
    # 모델 검증
    test_texts = ['Hello world', '안녕하세요']
    embeddings = model.encode(test_texts)
    logger.info(f"✅ 모델 검증 완료! 임베딩 차원: {embeddings.shape}")
    
    # 모델 파일 확인
    model_files = list(Path(model_cache).rglob("*"))
    logger.info(f"✅ 총 {len(model_files)}개 모델 파일 저장됨")
    
    logger.info("🎉 HuggingFace 모델 준비 완료!")
    
except Exception as e:
    logger.error(f"❌ 모델 다운로드 실패: {e}")
    raise e
EOF

# 모델 파일 확인 및 권한 설정
RUN echo "📋 모델 파일 구조:" && \
    find /app/models -type f | head -10 && \
    echo "총 파일 수: $(find /app/models -type f | wc -l)" && \
    chmod -R 755 /app/models

# 빌드 도구 정리 (모델 다운로드 후)
RUN apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && pip cache purge

# ai-repository에서 가져온 소스 코드
COPY src/ ./src/
COPY etc/ ./etc/

# 로그 디렉토리 생성
RUN mkdir -p /var/log/jypark/quiz-api /tmp/jypark/quiz-api

# 환경 변수 설정 (HuggingFace 로컬 모델 사용)
ENV PYTHONPATH=/app/src \
    STAGE=production \
    ERR_LOG_PATH=/var/log/jypark/quiz-api \
    TMP_FILE_PATH=/tmp/jypark/quiz-api \
    DEFAULT_LOGGING_PATH=/var/log/jypark/quiz-api \
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    HF_HOME=/app/models/.cache/huggingface_hub \
    TORCH_HOME=/app/models/.cache/torch \
    TRANSFORMERS_OFFLINE=0 \
    HF_HUB_OFFLINE=0 \
    USE_LOCAL_MODEL=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 원본 gunicorn 설정 사용
CMD ["gunicorn", "-c", "gunicorn.conf.py", "--bind", "0.0.0.0:8000", "src.app.main:app"]
