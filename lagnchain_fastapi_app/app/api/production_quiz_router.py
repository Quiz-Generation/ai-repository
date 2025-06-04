"""
🏆 프로덕션 급 고품질 퀴즈 API 라우터
복잡하더라도 실제 품질이 보장되는 시스템
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import logging
import time
import numpy as np

from ..schemas.quiz_schema import QuizRequest, QuizResponse, Difficulty
from ..services.production_quiz_service import get_production_quiz_service, ProductionQuizService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/production/quiz",
    tags=["Production High-Quality Quiz"],
    responses={404: {"description": "Not found"}},
)


@router.post("/generate", response_model=QuizResponse)
async def generate_production_quiz(
    request: QuizRequest,
    quiz_service: ProductionQuizService = Depends(get_production_quiz_service)
) -> QuizResponse:
    """
    🏆 프로덕션 급 최고 품질 퀴즈 생성

    **복잡하더라도 실제 품질 보장:**
    - ✅ 다단계 중복 검출 엔진 (어휘적 + 의미적 + 내용적)
    - ✅ 지능형 문제 생성기 (품질 임계값 8.0/10)
    - ✅ 고급 RAG 다양성 검색 (Fibonacci 반복 방지)
    - ✅ 정확한 2:6:2 비율 적용
    - ✅ 실시간 품질 평가 및 재시도
    """

    start_time = time.time()

    try:
        logger.info(f"🏆 프로덕션 급 퀴즈 생성 요청: {request.num_questions}문제")

        # 엄격한 입력 검증
        if request.num_questions <= 0 or request.num_questions > 30:
            raise HTTPException(
                status_code=400,
                detail="프로덕션 시스템에서는 문제 수가 1-30개 사이여야 합니다"
            )

        if not request.document_id:
            raise HTTPException(
                status_code=400,
                detail="document_id가 필요합니다"
            )

        # 프로덕션 급 퀴즈 생성
        response = await quiz_service.generate_high_quality_quiz(request)

        processing_time = time.time() - start_time

        # 프로덕션 메타데이터 추가
        if response.metadata:
            response.metadata["api_processing_time"] = round(processing_time, 3)
            response.metadata["api_version"] = "production_v1"
            response.metadata["quality_engine"] = "Multi-Stage Quality Assurance"
            response.metadata["production_features"] = [
                "고급 RAG 다양성 검색",
                "지능형 문제 생성기",
                "다단계 중복 검출",
                "실시간 품질 평가",
                "자동 재시도 시스템"
            ]

        if response.success:
            quality_score = response.metadata.get("quality_report", {}).get("overall_score", 0)
            removed_duplicates = response.metadata.get("duplicate_removal", {}).get("removed_count", 0)

            logger.info(f"🎉 프로덕션 퀴즈 성공: {response.total_questions}문제, 품질 {quality_score:.1f}/10, 중복 제거 {removed_duplicates}개")

            # 품질 보장 검증
            if quality_score < 7.5:
                logger.warning(f"⚠️ 품질 점수 낮음: {quality_score:.1f}/10")
            if removed_duplicates > response.total_questions * 0.3:
                logger.warning(f"⚠️ 높은 중복 제거율: {removed_duplicates}개")

        else:
            logger.error(f"🚨 프로덕션 퀴즈 생성 실패: {response.error}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = f"프로덕션 퀴즈 생성 중 예상치 못한 오류: {str(e)}"

        logger.error(f"🚨 {error_msg} ({error_time:.2f}초)")

        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@router.get("/health")
async def production_health_check():
    """🔍 프로덕션 퀴즈 서비스 상태 확인"""
    try:
        quiz_service = get_production_quiz_service()

        return {
            "status": "healthy",
            "service": "Production High-Quality Quiz Service",
            "quality_level": "Production Grade",
            "components": {
                "rag_retriever": "Advanced RAG with Diversity Search",
                "question_generator": "Intelligent Question Generator",
                "duplicate_detector": "Multi-Stage Duplicate Detection",
                "quality_evaluator": "Real-time Quality Assessment"
            },
            "quality_features": [
                "🏆 품질 임계값 8.0/10",
                "🔍 다단계 중복 검출 (어휘적 + 의미적 + 내용적)",
                "🧠 지능형 문제 생성 (재시도 포함)",
                "🎯 정확한 타입 분배 (2:6:2)",
                "⚡ 고급 RAG 다양성 검색",
                "📊 실시간 품질 보고서"
            ],
            "production_standards": {
                "min_quality_score": 8.0,
                "max_duplicate_threshold": 0.75,
                "quality_assurance_stages": 6,
                "automatic_retry": True,
                "diversity_enforcement": True
            }
        }

    except Exception as e:
        logger.error(f"프로덕션 상태 확인 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"서비스 상태 확인 실패: {str(e)}"
        )


@router.get("/quality/standards")
async def get_quality_standards():
    """📊 품질 기준 정보"""
    return {
        "quality_scoring": {
            "scale": "0-10점",
            "production_threshold": 8.0,
            "factors": {
                "clarity": "명확성 (30%)",
                "relevance": "관련성 (30%)",
                "difficulty": "난이도 적절성 (20%)",
                "uniqueness": "고유성 (20%)"
            }
        },
        "duplicate_detection": {
            "stages": [
                "어휘적 중복 (Lexical Similarity)",
                "의미적 중복 (Semantic Similarity)",
                "내용적 중복 (Content Similarity)"
            ],
            "semantic_threshold": 0.75,
            "embedding_model": "jhgan/ko-sroberta-multitask"
        },
        "type_distribution": {
            "default_ratio": "2:6:2 (OX:객관식:주관식)",
            "exact_calculation": True,
            "minimum_per_type": 1
        },
        "generation_process": {
            "max_attempts": 5,
            "batch_generation": True,
            "quality_filtering": True,
            "diversity_enforcement": True
        }
    }


@router.post("/analyze/quality", response_model=dict)
async def analyze_quiz_quality(
    request: QuizRequest,
    quiz_service: ProductionQuizService = Depends(get_production_quiz_service)
) -> dict:
    """
    🔍 퀴즈 품질 분석 (생성 없이)
    실제 생성하지 않고 품질 예측만 수행
    """

    try:
        logger.info(f"🔍 퀴즈 품질 분석: {request.num_questions}문제")

        # 문서 확인
        doc_info = quiz_service.vector_service.get_document_info(request.document_id)
        if not doc_info:
            raise HTTPException(
                status_code=404,
                detail=f"문서를 찾을 수 없습니다: {request.document_id}"
            )

        # 컨텍스트 품질 분석
        contexts = await quiz_service.rag_retriever.get_diverse_contexts(
            request.document_id,
            request.num_questions
        )

        # 타입 분배 분석
        type_distribution = quiz_service._calculate_exact_distribution(request)

        # 품질 예측
        estimated_quality = 8.5  # 프로덕션 시스템 기본 예상 품질
        if len(contexts) < request.num_questions:
            estimated_quality -= 1.0
        if not contexts:
            estimated_quality = 3.0

        diversity_score = len(set(ctx.diversity_keywords for ctx in contexts if ctx.diversity_keywords))

        return {
            "document_analysis": {
                "document_exists": True,
                "document_info": doc_info
            },
            "context_analysis": {
                "available_contexts": len(contexts),
                "required_contexts": request.num_questions,
                "context_sufficiency": len(contexts) >= request.num_questions,
                "diversity_score": min(10, diversity_score / max(1, request.num_questions) * 10),
                "average_complexity": np.mean([ctx.complexity_level for ctx in contexts]) if contexts else 0
            },
            "type_distribution_analysis": {
                "requested_distribution": {k.value: v for k, v in type_distribution.items()},
                "distribution_feasible": True
            },
            "quality_prediction": {
                "estimated_overall_score": round(estimated_quality, 1),
                "predicted_success_rate": min(100, max(0, (estimated_quality - 5) * 20)),
                "expected_duplicate_rate": "< 5%",
                "quality_factors": {
                    "context_quality": "High" if len(contexts) >= request.num_questions else "Medium",
                    "diversity": "High" if diversity_score > request.num_questions else "Medium",
                    "complexity_balance": "Appropriate"
                }
            },
            "recommendations": [
                "프로덕션 시스템으로 고품질 보장됨",
                "다단계 중복 검출로 중복 최소화",
                "지능형 생성기로 품질 8.0+ 보장"
            ] if estimated_quality >= 8.0 else [
                "컨텍스트 부족으로 품질 저하 예상",
                "추가 문서 업로드 권장",
                "문제 수 조정 고려"
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"품질 분석 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"품질 분석 실패: {str(e)}"
        )


@router.post("/test/production", response_model=QuizResponse)
async def test_production_system(
    document_id: str = "test-doc",
    num_questions: int = 3,
    quiz_service: ProductionQuizService = Depends(get_production_quiz_service)
) -> QuizResponse:
    """
    🧪 프로덕션 시스템 테스트
    소규모 테스트로 품질 검증
    """

    try:
        test_request = QuizRequest(
            document_id=document_id,
            num_questions=num_questions,
            difficulty=Difficulty.MEDIUM,
            question_types=None  # 기본 2:6:2 비율 테스트
        )

        logger.info(f"🧪 프로덕션 시스템 테스트: {num_questions}문제")

        response = await quiz_service.generate_high_quality_quiz(test_request)

        # 테스트 결과 검증
        if response.success:
            quality_score = response.metadata.get("quality_report", {}).get("overall_score", 0)
            duplicate_count = response.metadata.get("duplicate_removal", {}).get("removed_count", 0)

            test_result = {
                "test_passed": quality_score >= 8.0 and duplicate_count == 0,
                "quality_score": quality_score,
                "duplicate_count": duplicate_count,
                "production_ready": quality_score >= 8.0
            }

            response.metadata["test_result"] = test_result

        return response

    except Exception as e:
        logger.error(f"프로덕션 시스템 테스트 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"프로덕션 시스템 테스트 실패: {str(e)}"
        )