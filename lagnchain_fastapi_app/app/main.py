from fastapi import FastAPI, HTTPException,status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime

# API 라우터 임포트
from lagnchain_fastapi_app.app.api.pdf_service import router as pdf_router

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="최적화된 PDF 처리 API",
    description="PyMuPDF 기반 고성능 PDF 텍스트 추출 서비스",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(pdf_router)

# 루트 엔드포인트
@app.get("/")
async def root():
    """API 정보"""
    return {
        "message": "최적화된 PDF 처리 API",
        "version": "2.0.0",
        "features": [
            "🚀 PyMuPDF 기반 고속 처리",
            "🔧 팩토리 패턴으로 확장 가능",
            "📊 상세한 성능 메트릭",
            "🎯 실시간 처리 최적화"
        ],
        "endpoints": {
            "upload_and_extract": "/pdf/upload-and-extract",
            "extract_fast": "/pdf/extract-fast",
            "extractors": "/pdf/extractors",
            "health": "/pdf/health"
        }
    }

# 예외 처리 핸들러
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 처리"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 처리"""
    logger.error(f"예상치 못한 오류 발생: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "message": "내부 서버 오류가 발생했습니다.",
            "timestamp": datetime.now().isoformat()
        }
    )

# 개발 서버 실행
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
