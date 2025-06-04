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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz", tags=["Quiz Generation"])

# 전역 퀴즈 서비스 인스턴스
quiz_service = get_default_quiz_service()


@router.get("/health")
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


@router.post("/generate")
async def generate_quiz(request: QuizRequestAPI) -> JSONResponse:
    """🧠 PDF 문서 기반 퀴즈 자동 생성 (메인 기능)

    **🤖 AI가 PDF를 분석하여 자동으로 최적의 퀴즈를 생성합니다**

    **핵심 특징:**
    - ✨ **토픽 자동 추출**: PDF 내용을 분석하여 핵심 주제들을 자동 추출
    - 🎯 **RAG 최적화**: 관련성 높은 컨텍스트만 선별하여 고품질 문제 생성
    - 🔄 **지능형 난이도 조절**: 요청한 난이도에 맞는 문제 유형과 복잡도 자동 선택
    - 📊 **품질 보장**: AI가 생성한 문제를 자동으로 검증

    **처리 과정:**
    1. 📄 **문서 확인**: 업로드된 PDF 문서 존재 및 상태 확인
    2. 🤖 **토픽 자동 추출**: AI가 PDF 내용을 분석하여 핵심 주제 추출
    3. 🔍 **RAG 컨텍스트 검색**: 추출된 토픽 기반으로 최적 컨텍스트 검색
    4. ⚡ **LLM 퀴즈 생성**: 컨텍스트와 토픽을 바탕으로 문제 생성
    5. ✅ **품질 검증**: 생성된 문제의 품질 자동 검증 및 최적화

    **📝 요청 예시 (간단):**
    ```json
    {
        "document_id": "f7dbd017-426e-4919-8a88-feda68949615",
        "num_questions": 5,
        "difficulty": "medium"
    }
    ```
    → AI가 자동으로 토픽을 추출하고 적절한 문제 유형을 선택합니다

    **📝 요청 예시 (커스텀):**
    ```json
    {
        "document_id": "f7dbd017-426e-4919-8a88-feda68949615",
        "num_questions": 8,
        "difficulty": "hard",
        "question_types": ["multiple_choice", "short_answer"],
        "topics": ["알고리즘", "복잡도"]
    }
    ```
    → 자동 추출된 토픽 + 힌트 토픽을 조합하여 더 정확한 문제 생성

    **💡 사용 팁:**
    - `topics`는 선택사항입니다. AI가 자동으로 최적의 토픽을 찾아줍니다
    - 특정 주제에 집중하고 싶다면 `topics`에 힌트를 제공하세요
    - 난이도에 따라 문제 유형이 자동으로 최적화됩니다
    """

    generation_start = time.time()

    logger.info(f"퀴즈 생성 API 요청: {request.document_id} ({request.num_questions}문제)")

    try:
        # API 요청을 내부 스키마로 변환
        internal_request = QuizRequest(
            document_id=request.document_id,
            num_questions=request.num_questions,
            difficulty=Difficulty(request.difficulty),
            question_types=[QuestionType(qt) for qt in request.question_types] if request.question_types else None,
            topics=request.topics,
            language=request.language
        )

        # 퀴즈 생성
        response = quiz_service.generate_quiz(internal_request)

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


@router.get("/topics/{document_id}")
async def extract_document_topics(
    document_id: str = Path(..., description="문서 ID"),
    max_topics: int = Query(10, ge=1, le=20, description="최대 토픽 수")
) -> JSONResponse:
    """📝 문서에서 주요 토픽 추출

    퀴즈 생성 전에 문서의 주요 토픽들을 미리 확인할 수 있습니다.
    추출된 토픽을 /quiz/generate의 topics 파라미터에 활용하세요.
    """

    logger.info(f"토픽 추출 요청: {document_id}")

    try:
        extraction_start = time.time()

        # 토픽 추출
        topics = quiz_service.extract_topics(document_id)

        if not topics:
            raise HTTPException(
                status_code=404,
                detail=f"문서를 찾을 수 없거나 토픽을 추출할 수 없습니다: {document_id}"
            )

        extraction_time = time.time() - extraction_start

        # 토픽 수 제한
        limited_topics = topics[:max_topics]

        return JSONResponse(
            status_code=200,
            content={
                "message": "토픽 추출 성공",
                "document_id": document_id,
                "total_topics_found": len(topics),
                "returned_topics": len(limited_topics),
                "extraction_time": round(extraction_time, 3),
                "topics": limited_topics,
                "llm_model_used": quiz_service.llm_service.model_name,
                "recommendations": {
                    "quiz_generation": "이 토픽들을 /quiz/generate API의 topics 파라미터에 사용하세요",
                    "topic_selection": "관심 있는 토픽 2-5개를 선택하면 더 집중된 퀴즈를 생성할 수 있습니다",
                    "difficulty_matching": "토픽의 복잡도에 따라 difficulty 파라미터를 조정하세요"
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"토픽 추출 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"토픽 추출 오류: {str(e)}")


@router.post("/switch-llm")
async def switch_llm_model(
    provider: str = Body(..., description="LLM 제공업체"),
    model_name: str = Body(..., description="모델 이름"),
    api_key: Optional[str] = Body(None, description="API 키 (선택사항)")
) -> JSONResponse:
    """🔄 LLM 모델 교체

    다른 LLM 모델로 전환하여 퀴즈 생성 스타일을 변경할 수 있습니다.

    **지원 모델:**
    - OpenAI: gpt-4o-mini, gpt-4, gpt-3.5-turbo
    - 추후: 국내 한국어 모델들

    **예시 요청:**
    ```json
    {
        "provider": "openai",
        "model_name": "gpt-4",
        "api_key": "sk-..."
    }
    ```
    """

    logger.info(f"LLM 모델 교체 요청: {provider}/{model_name}")

    try:
        global quiz_service

        # 현재 모델 정보 저장
        previous_model = quiz_service.llm_service.model_name
        previous_provider = quiz_service.llm_service.provider.value

        # 제공업체 검증
        if provider not in LLMFactory.get_available_providers():
            raise ValueError(f"지원하지 않는 제공업체: {provider}")

        # 새 LLM 서비스 생성
        config = LLMConfig(
            provider=LLMProvider(provider),
            model_name=model_name,
            api_key=api_key
        )

        new_llm_service = LLMFactory.create_llm(config)

        # 퀴즈 서비스의 LLM 교체
        quiz_service.switch_llm_model(new_llm_service)

        return JSONResponse(
            status_code=200,
            content={
                "message": "LLM 모델 교체 성공",
                "previous_model": {
                    "provider": previous_provider,
                    "model_name": previous_model
                },
                "current_model": {
                    "provider": provider,
                    "model_name": model_name
                },
                "switch_timestamp": time.time(),
                "note": "이제 새로운 모델로 퀴즈가 생성됩니다"
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"LLM 모델 교체 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM 모델 교체 실패: {str(e)}")


@router.get("/models")
async def get_available_models() -> JSONResponse:
    """📋 사용 가능한 LLM 모델 목록 조회"""

    try:
        return JSONResponse(
            status_code=200,
            content={
                "message": "사용 가능한 LLM 모델 목록",
                "current_model": {
                    "provider": quiz_service.llm_service.provider.value,
                    "model_name": quiz_service.llm_service.model_name
                },
                "available_providers": LLMFactory.get_available_providers(),
                "provider_details": {
                    "openai": {
                        "models": ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
                        "status": "available",
                        "note": "API 키 필요"
                    },
                    "anthropic": {
                        "models": ["claude-3-sonnet", "claude-3-haiku"],
                        "status": "coming_soon",
                        "note": "준비 중"
                    },
                    "korean_local": {
                        "models": ["kullm-polyglot-12.8b-v2", "ko-alpaca"],
                        "status": "planned",
                        "note": "한국어 최적화 모델 (계획 중)"
                    }
                },
                "recommendations": {
                    "korean_documents": "한국어 문서에는 한국어 최적화 모델을 권장합니다 (준비 중)",
                    "technical_content": "기술 문서에는 gpt-4를 권장합니다",
                    "general_usage": "일반적인 사용에는 gpt-4o-mini가 적합합니다"
                }
            }
        )

    except Exception as e:
        logger.error(f"모델 목록 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"모델 목록 조회 오류: {str(e)}")


@router.get("/examples")
async def get_usage_examples() -> JSONResponse:
    """💡 퀴즈 생성 API 사용 예시"""

    return JSONResponse(
        status_code=200,
        content={
            "message": "퀴즈 생성 API 사용 예시",
            "examples": {
                "basic_quiz": {
                    "description": "기본 퀴즈 생성",
                    "request": {
                        "document_id": "doc_12345",
                        "num_questions": 5,
                        "difficulty": "medium"
                    },
                    "note": "시스템이 자동으로 토픽을 추출하고 문제 유형을 결정합니다"
                },
                "specific_topics": {
                    "description": "특정 토픽 집중 퀴즈",
                    "request": {
                        "document_id": "doc_12345",
                        "num_questions": 10,
                        "difficulty": "hard",
                        "topics": ["알고리즘", "자료구조", "복잡도"]
                    },
                    "note": "지정된 토픽에 집중된 문제가 생성됩니다"
                },
                "custom_question_types": {
                    "description": "문제 유형 지정",
                    "request": {
                        "document_id": "doc_12345",
                        "num_questions": 8,
                        "difficulty": "easy",
                        "question_types": ["multiple_choice", "true_false"]
                    },
                    "note": "객관식과 참/거짓 문제만 생성됩니다"
                },
                "comprehensive_quiz": {
                    "description": "종합 퀴즈",
                    "request": {
                        "document_id": "doc_12345",
                        "num_questions": 15,
                        "difficulty": "medium",
                        "topics": ["핵심개념", "응용"],
                        "question_types": ["multiple_choice", "short_answer", "fill_blank"],
                        "language": "ko"
                    },
                    "note": "다양한 문제 유형과 토픽을 포함한 종합 퀴즈"
                }
            },
            "workflow": {
                "step1": "POST /pdf/upload - PDF 문서 업로드하여 document_id 획득",
                "step2": "GET /quiz/topics/{document_id} - 문서 토픽 확인 (선택사항)",
                "step3": "POST /quiz/generate - 퀴즈 생성",
                "step4": "생성된 퀴즈를 활용하여 학습 진행"
            },
            "tips": {
                "quality": "더 나은 품질을 위해 구체적인 토픽을 지정하세요",
                "performance": "대용량 문서의 경우 토픽을 미리 추출하여 활용하세요",
                "customization": "학습 목표에 맞게 난이도와 문제 유형을 조정하세요"
            }
        }
    )


# 에러 핸들러 제거 - 대신 일반 함수로 변경
def create_error_response(exc: HTTPException) -> JSONResponse:
    """퀴즈 API 에러 응답 생성 함수"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "퀴즈 생성 오류",
            "detail": exc.detail,
            "suggestions": {
                "document_not_found": "document_id가 올바른지 확인하고, PDF가 먼저 업로드되었는지 확인하세요",
                "generation_failed": "다른 난이도나 더 적은 문제 수로 다시 시도해보세요",
                "invalid_parameters": "API 문서를 참조하여 파라미터 형식을 확인하세요"
            },
            "helpful_endpoints": [
                "GET /quiz/health - 서비스 상태 확인",
                "GET /quiz/examples - 사용 예시 확인",
                "GET /pdf/documents - 업로드된 문서 목록 확인"
            ]
        }
    )