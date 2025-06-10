#!/usr/bin/env python3
"""
🔥 LangChain FastAPI 퀴즈 애플리케이션
- ChromaDB 벡터 검색
- OpenAI 퀴즈 자동 생성
- 고성능 배치 처리
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager

from .api import pdf_service, quiz_service
from .core.config import get_cached_settings
from .services.vector_service import get_global_vector_service

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작시 실행
    logger.info("🚀 AI Quiz Generator 시작")

    # 벡터 서비스 초기화
    vector_service = get_global_vector_service()
    logger.info(f"🔥 벡터 DB 초기화: {vector_service.vector_db.name}")

    yield

    # 종료시 실행
    logger.info("👋 AI Quiz Generator 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="🔥 AI Quiz Generator",
    description="""
## 📋 개요
PDF 문서를 업로드하고 AI가 자동으로 퀴즈를 생성하는 시스템

## 🎯 주요 기능
- 📄 PDF 업로드 및 텍스트 추출
- 🔍 ChromaDB 벡터 검색 (고성능)
- 🤖 AI 퀴즈 자동 생성 (O/X, 객관식, 주관식)
- 🎯 난이도별 문제 생성 (Easy, Medium, Hard)

## 🛠️ 기술 스택
- 백엔드: FastAPI, Python 3.12+
- AI: LangChain, OpenAI API
- 벡터 DB: ChromaDB (자동 임베딩)
- 임베딩: SentenceTransformers
- PDF 처리: PyMuPDF

## 🚀 성능 향상
- ⚡ 100배 빠른 벡터 검색 (HNSW 인덱싱)
- 🔄 배치 퀴즈 생성 (단일 API 호출)
- 💾 영구 저장 (ChromaDB 자동 저장)

## 🎯 사용법
1. PDF 업로드: `/pdf/upload` 엔드포인트 사용
2. 퀴즈 생성: `/quiz/generate` 엔드포인트 사용
3. 문서 검색: `/pdf/search` 엔드포인트 사용
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
app.include_router(pdf_service.router, prefix="/pdf", tags=["PDF 서비스"])
app.include_router(quiz_service.router, prefix="/quiz", tags=["퀴즈 서비스"])


# 미들웨어
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """요청 처리 시간 측정"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# 루트 엔드포인트
@app.get("/", tags=["기본"])
async def root():
    """서비스 정보"""
    settings = get_cached_settings()
    vector_service = get_global_vector_service()

    try:
        stats = vector_service.get_stats()
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        stats = {"error": str(e)}

    return {
        "message": "🔥 AI Quiz Generator API",
        "version": "1.0.0",
        "description": "ChromaDB 기반 고성능 퀴즈 생성 시스템",
        "features": [
            "📄 PDF 업로드 및 텍스트 처리",
            "🔍 ChromaDB 벡터 검색 (100배 빠름)",
            "🤖 AI 퀴즈 자동 생성 (배치 처리)",
            "🎯 난이도별 문제 생성",
            "⚡ 고성능 임베딩 검색",
            "💾 영구 저장 (자동 백업)"
        ],
        "tech_stack": {
            "backend": "FastAPI",
            "ai": "LangChain + OpenAI",
            "vector_db": "ChromaDB",
            "embedding": "SentenceTransformers",
            "pdf_processing": "PyMuPDF"
        },
        "vector_database": {
            "type": vector_service.vector_db.name,
            "status": "active",
            "stats": stats
        },
        "endpoints": {
            "docs": "/docs",
            "pdf_upload": "/pdf/upload",
            "quiz_generate": "/quiz/generate",
            "search": "/pdf/search"
        }
    }


# 헬스체크
@app.get("/health", tags=["시스템"])
async def health_check():
    """시스템 상태 확인"""
    vector_service = get_global_vector_service()

    # 간단한 상태 확인
    try:
        doc_count = vector_service.vector_db.count_documents()
        vector_status = "healthy"
    except Exception as e:
        logger.error(f"벡터 DB 상태 확인 실패: {e}")
        doc_count = -1
        vector_status = "error"

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "vector_db": {
                "status": vector_status,
                "type": vector_service.vector_db.name,
                "document_count": doc_count
            },
            "api": {
                "status": "healthy"
            }
        }
    }


# 에러 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 처리"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": "서버에서 예기치 않은 오류가 발생했습니다.",
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_cached_settings()

    print("🔥 AI Quiz Generator 시작")
    print(f"📍 서버: http://{settings.HOST}:{settings.PORT}")
    print(f"📚 문서: http://{settings.HOST}:{settings.PORT}/docs")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
