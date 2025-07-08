#!/usr/bin/env python3
"""
🚀 FastAPI PDF Processing with Vector DB Integration
"""
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import uvicorn

from .api import document_routes, quiz_routes, test_routes
from .service.vector_db_service import VectorDBService

# 로깅 설정
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "app.log"), encoding="utf-8")
    ]
)

logger = logging.getLogger(__name__)

# 전역 서비스 인스턴스
global_vector_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행"""
    global global_vector_service

    logger.info("🚀 FastAPI PDF Processing with Vector DB 시작")

    # 전역 서비스 초기화
    try:
        logger.info("🔧 전역 벡터 DB 서비스 초기화 시작")
        global_vector_service = VectorDBService()
        await global_vector_service.initialize_embedding_model()
        await global_vector_service.initialize_vector_db("milvus")
        logger.info("✅ 전역 벡터 DB 서비스 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 전역 서비스 초기화 실패: {e}")
        raise

    yield

    logger.info("🛑 FastAPI PDF Processing with Vector DB 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="PDF Processing with Vector DB API",
    description="동적 PDF 로더 선택 및 벡터 데이터베이스 통합 시스템",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(document_routes.router, prefix="/api/v2/documents")
app.include_router(quiz_routes.router, prefix="/api/v2/quiz")
app.include_router(test_routes.router, prefix="/api/v2/test")


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "service": "PDF Processing with Vector DB",
        "version": "2.0.0",
        "global_services_initialized": global_vector_service is not None
    }

# 개발 서버 실행
if __name__ == "__main__":
    logger.info("🎯 서버 시작: http://localhost:7000")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7000,
        reload=True,
        log_level="info"
    )