#!/usr/bin/env python3
"""
🚀 FastAPI PDF Processing with Vector DB Integration
"""
import logging
import os
import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from fastapi.responses import JSONResponse
import uvicorn

from src.common.utils.logger import set_logger
from src.common.error import JSendError, ErrorCode

from src.app.api import document, quiz, test_routes
from src.common.milvus.connect import milvus_db

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
logger = set_logger("exception")



@asynccontextmanager
async def lifespan(app: FastAPI):
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
app.include_router(document.router, prefix="/api/v2/documents")
app.include_router(quiz.router, prefix="/api/v2/quiz")
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


@app.exception_handler(JSendError)
async def jsend_error_exception_handler(request: Request, exc: JSendError):
    logger.error(f"[{request.url}] JSendError {exc.__dict__}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=400,
        content={
            "status": exc.status,
            "code": exc.code,
            "message": exc.message,
            "data": exc.data,
        },
    )

@app.exception_handler(Exception)
async def unknown_error_exception_handler(request, exc: Exception):
    # 전체 스택 트레이스를 가져옴
    full_traceback = traceback.format_exc()

    # 줄바꿈으로 분할하여 리스트로 만듦
    traceback_lines = full_traceback.splitlines()

    # 마지막 5줄 | 10줄만 추출
    # last_five_lines_of_traceback = "\n".join(traceback_lines[-5:])
    last_five_lines_of_traceback = "\n".join(traceback_lines[-10:])
    logger.error(
        f"""
            [{request.url}] InternalError
            {last_five_lines_of_traceback}
        """
    )


    # 슬랙과 로거에 에러 메세지
    # if setting.SLACK_WEBHOOK_ENABLE == "on":
    #         await SLACK_LOGGER.application_in_critical_status_alarm(
    #         str(exc) + "\n\n" + last_five_lines_of_traceback
    #     )
    logger.exception(f"[{request.url}] InternalError\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "code": ErrorCode.Common.DEFAULT_ERROR[0],
            "message": ErrorCode.Common.DEFAULT_ERROR[1]
        }
    )
