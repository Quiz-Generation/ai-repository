from fastapi import FastAPI, HTTPException,status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime

# API 라우터 임포트 (상대 경로로 변경)
from lagnchain_fastapi_app.app.api.pdf_service import router as pdf_router

# 로깅 설정 (개선된 버전)
import logging
from datetime import datetime

# 동적 PDF 서비스용 상세 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
    ]
)

# 특정 모듈의 로그 레벨 조정
logger = logging.getLogger(__name__)
pdf_logger = logging.getLogger("app.services.dynamic_pdf")
pdf_logger.setLevel(logging.INFO)  # 동적 PDF 서비스 상세 로깅

# API 추출 관련 로깅
api_logger = logging.getLogger("app.api.pdf_service")
api_logger.setLevel(logging.INFO)

# FastAPI 앱 인스턴스 생성 (Swagger 문서 설정 개선)
app = FastAPI(
    title="🔥 PDF 벡터 검색 API",
    description="""
    PDF 파일을 업로드하여 벡터 데이터베이스에 저장하고 RAG 기반 검색을 제공하는 API

    🚀 주요 기능
    - 📤 PDF 파일 업로드 및 자동 텍스트 추출
    - 🔍 벡터 기반 유사도 검색 (Weaviate, ChromaDB 지원)
    - 📋 문서별 관리 및 메타데이터 저장
    - 🎯 특정 문서에서 컨텍스트 추출 (RAG 준비)
    - 🔄 실시간 데이터베이스 전환

    📊 지원 벡터 DB
    - **Weaviate** (권장): 고성능, 확장성 우수
    - **ChromaDB**: 가벼움, 메모리 효율적

    🛠️ 시작하기
    1. `/pdf/health` - 서비스 상태 확인
    2. `POST /pdf/upload` - PDF 업로드
    3. `GET /pdf/search` - 벡터 검색
    4. `GET /pdf/documents` - 문서 목록 조회
    """,
    version="3.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_tags=[
        {
            "name": "PDF Vector",
            "description": "PDF 벡터 검색 및 문서 관리 API",
        },
        {
            "name": "Health Check",
            "description": "서비스 상태 확인",
        }
    ]
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
@app.get("/", tags=["Health Check"])
async def root():
    """🏠 API 루트 - 서비스 정보 및 엔드포인트 가이드"""
    return {
        "message": "🔥 PDF 벡터 검색 API",
        "version": "3.0.0",
        "features": [
            "📤 PDF 업로드 및 벡터 저장",
            "🔍 고성능 유사도 검색",
            "🎯 문서별 컨텍스트 추출",
            "🔄 실시간 DB 전환 (Weaviate ↔ ChromaDB)",
            "📊 상세한 성능 메트릭"
        ],
        "quick_start": {
            "1_health_check": "GET /pdf/health",
            "2_upload_pdf": "POST /pdf/upload",
            "3_search": "GET /pdf/search?query=검색어",
            "4_documents": "GET /pdf/documents"
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        },
        "supported_databases": ["weaviate", "chroma"]
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

# 개발 서버 실행 (포트 7000으로 변경)
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7000,  # 사용자가 이미 7000번 포트 사용 중
        reload=True,
        log_level="info"
    )
