"""
퀴즈 생성 API 라우터
PDF 문서 기반 RAG 퀴즈 생성 시스템

주요 엔드포인트:
- POST /quiz/generate: 퀴즈 생성 (메인 기능)
- GET /quiz/topics/{document_id}: 문서 토픽 추출
- POST /quiz/switch-llm: LLM 모델 교체
- GET /quiz/health: 서비스 상태 확인
"""

from fastapi import APIRouter, HTTPException, Body, Path, Query
from fastapi.responses import JSONResponse
import logging
import time
from typing import List, Dict, Any, Optional

# 퀴즈 서비스 및 스키마 import
from ..services.quiz_service import get_default_quiz_service
from ..services.llm_factory import LLMFactory, LLMProvider, LLMConfig
from ..schemas.quiz_schema import (
    QuizRequest,  Difficulty, QuestionType,
    QuizRequestAPI, QuestionAPI
)

# Swagger 문서 설명 import
from ..docs.quiz_service import (
    desc_generate_quiz,
    desc_extract_topics,
    desc_switch_llm,
    desc_get_models,
    desc_health_check
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz", tags=["Quiz Generation"])

# 전역 퀴즈 서비스 인스턴스
quiz_service = get_default_quiz_service()


@router.get("/health", description=desc_health_check)
async def health_check() -> JSONResponse:
    """🔍 퀴즈 생성 서비스 상태 확인"""
    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "PDF RAG Quiz Generation Service",
                "llm_model": quiz_service.llm_service.model_name,
                "llm_provider": quiz_service.llm_service.provider.value,
                "vector_db": quiz_service.vector_service.db_type,
                "supported_features": [
                    "PDF 기반 퀴즈 생성",
                    "RAG 컨텍스트 검색",
                    "동적 토픽 추출",
                    "다양한 문제 유형",
                    "난이도별 문제 생성",
                    "LLM 모델 교체",
                    "문제 품질 검증"
                ],
                "available_difficulties": ["easy", "medium", "hard"],
                "available_question_types": [
                    "multiple_choice", "short_answer", "fill_blank", "true_false"
                ],
                "supported_llm_providers": LLMFactory.get_available_providers(),
                "endpoints": [
                    "POST /quiz/generate",
                    "GET /quiz/topics/{document_id}",
                    "POST /quiz/switch-llm",
                    "GET /quiz/health"
                ]
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@router.post("/generate", description=desc_generate_quiz)
async def generate_quiz(request: QuizRequestAPI) -> JSONResponse:
    generation_start = time.time()
    logger.info(f"퀴즈 생성 API 요청: {request.document_id} ({request.num_questions}문제)")
    try:
        # API 요청을 내부 모델로 변환
        quiz_request = QuizRequest(
            document_id=request.document_id,
            num_questions=request.num_questions,
            difficulty=Difficulty(request.difficulty),
            question_types=[QuestionType(qt) for qt in request.question_types] if request.question_types else None,
            language=request.language
        )

        # 퀴즈 생성
        response = await quiz_service.generate_quiz(quiz_request)

        if not response.success:
            raise HTTPException(status_code=400, detail=f"퀴즈 생성 실패: {response.error}")

        # API 응답 형식으로 변환
        api_questions = []
        for question in response.questions:
            api_question = QuestionAPI(
                question=question.question,
                question_type=question.question_type.value,
                correct_answer=question.correct_answer,
                options=question.options,
                explanation=question.explanation,
                difficulty=question.difficulty.value,
                topic=question.topic
            )
            api_questions.append(api_question)

        total_time = time.time() - generation_start

        return JSONResponse(
            status_code=200,
            content={
                "message": "퀴즈 생성 성공",
                "quiz_id": response.quiz_id,
                "document_id": response.document_id,
                "questions": [q.__dict__ for q in api_questions],
                "total_questions": response.total_questions,
                "difficulty": response.difficulty.value,
                "generation_time": response.generation_time,
                "api_processing_time": round(total_time, 3),
                "created_at": response.created_at,

                # 📊 생성 통계 및 품질 정보
                "generation_info": {
                    "llm_model_used": response.metadata.get("llm_model"),
                    "extracted_topics": response.metadata.get("extracted_topics", []),
                    "contexts_used": response.metadata.get("contexts_used", 0),
                    "avg_context_similarity": round(response.metadata.get("avg_context_similarity", 0), 3),
                    "question_types_generated": response.metadata.get("generation_stats", {}).get("question_types_used", [])
                },

                # 🔍 품질 검증 결과
                "quality_assessment": response.metadata.get("validation_result", {}),

                # 💡 사용 팁
                "usage_tips": {
                    "quiz_id": "이 quiz_id로 퀴즈 결과를 추적할 수 있습니다",
                    "question_navigation": "questions 배열의 각 문제는 topic과 source_context를 포함합니다",
                    "quality_improvement": "더 나은 품질을 위해 specific topics를 지정하거나 difficulty를 조정해보세요"
                }
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_time = time.time() - generation_start
        logger.error(f"퀴즈 생성 API 오류: {str(e)} ({error_time:.2f}초)")
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 오류: {str(e)}")


@router.get("/topics/{document_id}", description=desc_extract_topics)
async def extract_document_topics(
    document_id: str = Path(..., description="문서 ID"),
    max_topics: int = Query(10, ge=1, le=20, description="최대 토픽 수")
) -> JSONResponse:
    """📚 문서에서 퀴즈 생성용 토픽 자동 추출"""

    extraction_start = time.time()

    logger.info(f"토픽 추출 API 요청: {document_id} (최대 {max_topics}개)")

    try:
        # 토픽 추출
        extracted_topics = await quiz_service.extract_topics(document_id)

        if not extracted_topics:
            raise HTTPException(status_code=404, detail="문서에서 토픽을 추출할 수 없습니다")

        # 최대 개수 제한
        limited_topics = extracted_topics[:max_topics]

        extraction_time = time.time() - extraction_start

        return JSONResponse(
            status_code=200,
            content={
                "message": "토픽 추출 완료",
                "document_id": document_id,
                "extracted_topics": limited_topics,
                "total_topics_found": len(extracted_topics),
                "max_topics_requested": max_topics,
                "extraction_info": {
                    "document_analysis_time": round(extraction_time, 3),
                    "content_quality": "high" if len(extracted_topics) >= 5 else "medium",
                    "llm_model_used": quiz_service.llm_service.model_name
                },
                "usage_tip": "이 토픽들을 힌트로 사용하여 더 정확한 퀴즈를 생성할 수 있습니다"
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        error_time = time.time() - extraction_start
        logger.error(f"토픽 추출 API 오류: {str(e)} ({error_time:.2f}초)")
        raise HTTPException(status_code=500, detail=f"토픽 추출 오류: {str(e)}")


@router.post("/switch-llm", description=desc_switch_llm)
async def switch_llm_model(
    provider: str = Body(..., description="LLM 제공업체"),
    model_name: str = Body(..., description="모델 이름"),
    api_key: Optional[str] = Body(None, description="API 키 (선택사항)")
) -> JSONResponse:
    """🔄 퀴즈 생성용 LLM 모델을 동적으로 교체"""

    logger.info(f"LLM 모델 교체 요청: {provider}/{model_name}")

    try:
        # 이전 모델 정보 저장
        previous_model = {
            "provider": quiz_service.llm_service.provider.value,
            "model_name": quiz_service.llm_service.model_name
        }

        # 새로운 LLM 서비스 생성
        try:
            llm_provider = LLMProvider(provider.lower())
        except ValueError:
            available_providers = LLMFactory.get_available_providers()
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 LLM 제공업체: {provider}. 사용 가능: {available_providers}"
            )

        config = LLMConfig(
            provider=llm_provider,
            model_name=model_name,
            api_key=api_key
        )

        new_llm_service = LLMFactory.create_llm(config)

        # 퀴즈 서비스에서 LLM 모델 교체
        quiz_service.switch_llm_model(new_llm_service)

        return JSONResponse(
            status_code=200,
            content={
                "message": "LLM 모델 교체 완료",
                "previous_model": previous_model,
                "current_model": {
                    "provider": provider,
                    "model_name": model_name
                },
                "switch_time": time.time(),
                "status": "success",
                "note": "새로운 모델로 퀴즈 생성 시 특성이 달라질 수 있습니다"
            }
        )

    except Exception as e:
        logger.error(f"LLM 모델 교체 실패: {e}")
        raise HTTPException(status_code=500, detail=f"LLM 모델 교체 오류: {str(e)}")


@router.get("/models", description=desc_get_models)
async def get_available_models() -> JSONResponse:
    """📋 사용 가능한 LLM 모델 목록 조회"""

    try:
        current_model = {
            "provider": quiz_service.llm_service.provider.value,
            "model_name": quiz_service.llm_service.model_name
        }

        available_providers = [
            {
                "provider": "openai",
                "models": ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
                "status": "available",
                "description": "OpenAI GPT 시리즈 - 한국어 지원 우수"
            },
            {
                "provider": "anthropic",
                "models": ["claude-3-sonnet", "claude-3-haiku"],
                "status": "coming_soon",
                "description": "Anthropic Claude 시리즈 - 추후 지원 예정"
            },
            {
                "provider": "korean_local",
                "models": ["kullm-polyglot-12.8b-v2"],
                "status": "development",
                "description": "한국어 특화 로컬 모델 - 개발 중"
            }
        ]

        return JSONResponse(
            status_code=200,
            content={
                "message": "사용 가능한 LLM 모델 목록",
                "current_model": current_model,
                "available_providers": available_providers,
                "recommendations": {
                    "korean_quiz": "OpenAI gpt-4o-mini (한국어 최적화)",
                    "high_quality": "OpenAI gpt-4 (최고 품질)",
                    "fast_generation": "OpenAI gpt-3.5-turbo (빠른 생성)"
                },
                "switch_endpoint": "POST /quiz/switch-llm"
            }
        )

    except Exception as e:
        logger.error(f"모델 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail="모델 목록 조회 실패")


def create_error_response(exc: HTTPException) -> JSONResponse:
    """통합 에러 응답 생성"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )