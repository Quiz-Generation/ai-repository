#!/usr/bin/env python3
"""
🚀 기본 FastAPI 애플리케이션
"""
import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lagnchain_fastapi_app.app.api import document_routes
import uvicorn

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="AI Repository API",
    description="FastAPI 애플리케이션",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_routes.router, prefix="/api/v1")

# 헬스체크
@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    logger.info("✅ 헬스체크 요청 받음")
    return {
        "status": "healthy",
        "message": "API가 정상적으로 작동 중입니다."
    }

# 개발 서버 실행
if __name__ == "__main__":
    logger.info("🚀 서버 시작")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )