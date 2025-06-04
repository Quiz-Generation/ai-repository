from fastapi import FastAPI, HTTPException,status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime

# Config import 추가
from lagnchain_fastapi_app.app.core.config import get_settings

# API 라우터 임포트 (상대 경로로 변경)
from lagnchain_fastapi_app.app.api.pdf_service import router as pdf_router
from lagnchain_fastapi_app.app.api.quiz_service import router as quiz_router

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

# 설정 로드
settings = get_settings()

# FastAPI 앱 인스턴스 생성 (Swagger 문서 설정 개선)
app = FastAPI(
    title=settings.APP_NAME,
    description="""
🚀 **LangChain + FastAPI 기반 AI 문서 분석 및 퀴즈 생성 서비스**

이 서비스는 PDF 문서를 업로드하고 RAG(Retrieval-Augmented Generation) 기술을 통해
AI 기반 퀴즈를 자동 생성하는 포괄적인 플랫폼입니다.

## 🎯 주요 기능

### 📄 PDF 문서 처리
- **스마트 업로드**: PDF 파일 업로드 및 자동 텍스트 추출
- **벡터화**: 문서 내용을 벡터 데이터베이스에 저장하여 의미론적 검색 지원
- **청크 분할**: 긴 문서를 최적의 크기로 분할하여 효율적인 검색 성능 확보

### 🧠 AI 퀴즈 생성
- **다양한 문제 유형**: 객관식, 주관식, 빈칸 채우기, 참/거짓
- **난이도 조절**: 쉬움/보통/어려움 단계별 문제 생성
- **토픽 기반**: 문서에서 추출한 핵심 주제별 맞춤 문제
- **품질 검증**: AI가 생성한 문제의 품질을 자동으로 검증

### 🔄 LLM 모델 교체
- **유연한 아키텍처**: OpenAI, Anthropic, 한국어 모델 등 자유로운 교체
- **실시간 스위칭**: 서비스 중단 없이 모델 변경 가능
- **성능 최적화**: 각 모델별 최적화된 프롬프트 엔지니어링

## 🛠 기술 스택
- **Backend**: FastAPI, Python 3.12+
- **AI/ML**: LangChain, OpenAI GPT, RAG Pipeline
- **Vector DB**: ChromaDB, Weaviate 지원
- **Document Processing**: PyPDF2, python-docx
- **Quality**: TDD 방식 개발, 종합적인 테스트 커버리지

## 📊 워크플로우
1. **PDF 업로드** → 문서 분석 및 벡터화
2. **토픽 추출** → AI가 문서의 핵심 주제 자동 추출
3. **퀴즈 생성** → RAG 기반 맞춤형 문제 생성
4. **품질 검증** → 생성된 문제의 품질 자동 검증
5. **결과 제공** → 고품질의 학습용 퀴즈 완성

이 서비스는 교육, 연수, 시험 준비 등 다양한 학습 시나리오에서 활용할 수 있으며,
특히 대용량 문서에서 핵심 내용을 빠르게 학습하고 평가하는 데 최적화되어 있습니다.
    """,
    version=settings.APP_VERSION,
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
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(pdf_router)
app.include_router(quiz_router)

# 루트 엔드포인트
@app.get("/", tags=["Health Check"])
async def root():
    """🏠 API 루트 - 서비스 정보 및 엔드포인트 가이드"""
    return {
        "message": "🔥 PDF 벡터 검색 & RAG 퀴즈 생성 API",
        "version": "3.1.0",
        "features": [
            "📤 PDF 업로드 및 벡터 저장 (동적 추출기 지원)",
            "🔍 고성능 유사도 검색",
            "🧠 AI 기반 퀴즈 생성 (RAG + LLM)",
            "🎯 문서별 컨텍스트 추출",
            "🔄 실시간 DB 전환 (Weaviate ↔ ChromaDB)",
            "🤖 LLM 모델 교체 (OpenAI, 한국어 모델 등)",
            "📊 상세한 성능 메트릭"
        ],
        "quick_start": {
            "1_pdf_health": "GET /pdf/health",
            "2_quiz_health": "GET /quiz/health",
            "3_upload_pdf": "POST /pdf/upload",
            "4_generate_quiz": "POST /quiz/generate",
            "5_search": "GET /pdf/search?query=검색어",
            "6_documents": "GET /pdf/documents"
        },
        "workflows": {
            "basic_quiz_generation": [
                "1. POST /pdf/upload - PDF 업로드하여 document_id 획득",
                "2. GET /quiz/topics/{document_id} - 토픽 확인 (선택)",
                "3. POST /quiz/generate - 퀴즈 생성",
                "4. 생성된 퀴즈로 학습 진행"
            ],
            "advanced_customization": [
                "1. GET /quiz/models - 사용 가능한 LLM 모델 확인",
                "2. POST /quiz/switch-llm - 원하는 모델로 교체",
                "3. POST /quiz/generate - 커스텀 설정으로 퀴즈 생성"
            ]
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
            "examples": "/quiz/examples"
        },
        "supported_technologies": {
            "vector_databases": ["weaviate", "chroma"],
            "llm_providers": ["openai", "anthropic (준비중)", "korean_local (준비중)"],
            "pdf_extractors": ["pdfminer", "pdfplumber", "pymupdf"]
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

# 개발 서버 실행 (포트 7000으로 변경)
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7000,  # 사용자가 이미 7000번 포트 사용 중
        reload=True,
        log_level="info"
    )
