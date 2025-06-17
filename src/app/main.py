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

from .api import document_routes, quiz_routes

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행"""
    logger.info("🚀 FastAPI PDF Processing with Vector DB 시작")
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
app.include_router(document_routes.router)
app.include_router(quiz_routes.router)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "PDF Processing with Vector DB & AI Quiz Generation API",
        "version": "2.0.0",
        "features": [
            "🔍 동적 PDF 로더 선택 (PyMuPDF, PDFPlumber, PyPDF2, PDFMiner)",
            "🗄️ 벡터 데이터베이스 통합 (Milvus, Weaviate, FAISS)",
            "🧠 임베딩 생성 및 유사도 검색",
            "🌐 다국어 지원 (한국어 특화)",
            "📊 복잡도 기반 자동 선택",
            "🔄 폴백 메커니즘",
            "🤖 AI 기반 문제 생성 (LangGraph)",
            "📚 다중 도메인 문제 생성",
            "🎯 난이도별 맞춤 문제"
        ],
        "endpoints": {
            "document_upload": "/documents/upload",
            "document_list": "/documents/all-documents",
            "vector_status": "/documents/vector-status",
            "vector_switch": "/documents/vector-switch",
            "quiz_files": "/quiz/available-files",
            "quiz_generate": "/quiz/generate",
            "quiz_generate_simple": "/quiz/generate-simple",
            "quiz_options": "/quiz/options"
        },
        "ai_features": {
            "quiz_generation": {
                "workflow": "문서요약 → 주제추출 → 키워드추출 → 문제생성 → 품질검증",
                "difficulties": ["easy", "medium", "hard"],
                "question_types": ["multiple_choice", "true_false", "short_answer", "essay", "fill_blank"],
                "multi_domain": "AWS, 심리학, 알고리즘, 기술 등 다양한 도메인 지원"
            }
        }
    }


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "service": "PDF Processing with Vector DB",
        "version": "2.0.0"
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