"""
🚀 LangGraph 기반 퀴즈 API 라우터
- 진짜 작동하는 중복 제거
- 실제 2:6:2 비율 적용
- Agent 워크플로우 기반
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import logging
import time

from ..schemas.quiz_schema import QuizRequest, QuizResponse, Difficulty
from ..services.langgraph_quiz_service import get_langgraph_quiz_service, LangGraphQuizService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/quiz",
    tags=["LangGraph Quiz (v2)"],
    responses={404: {"description": "Not found"}},
)


@router.post("/generate", response_model=QuizResponse)
async def generate_langgraph_quiz(
    request: QuizRequest,
    quiz_service: LangGraphQuizService = Depends(get_langgraph_quiz_service)
) -> QuizResponse:
    """
    🚀 LangGraph 기반 고품질 퀴즈 생성

    **진짜 작동하는 개선사항:**
    - ✅ Agent 워크플로우로 품질 보장
    - ✅ 실제 중복 제거 (0.7 임계값)
    - ✅ 진짜 2:6:2 비율 (OX:객관식:주관식)
    - ✅ 구린 문제 자동 재생성 루프
    - ✅ 다양성 있는 컨텍스트 검색
    """

    start_time = time.time()

    try:
        logger.info(f"🚀 LangGraph 퀴즈 생성 요청: {request.num_questions}문제")

        # 입력 검증
        if request.num_questions <= 0 or request.num_questions > 50:
            raise HTTPException(
                status_code=400,
                detail="문제 수는 1-50개 사이여야 합니다"
            )

        if not request.document_id:
            raise HTTPException(
                status_code=400,
                detail="document_id가 필요합니다"
            )

        # LangGraph 워크플로우로 퀴즈 생성
        response = await quiz_service.generate_quiz(request)

        processing_time = time.time() - start_time

        # 메타데이터에 API 처리 시간 추가
        if response.metadata:
            response.metadata["api_processing_time"] = round(processing_time, 3)
            response.metadata["api_version"] = "v2_langgraph"
            response.metadata["workflow_engine"] = "LangGraph Agent"

        if response.success:
            logger.info(f"🎉 LangGraph 퀴즈 생성 성공: {response.total_questions}문제 ({processing_time:.2f}초)")

            # 성공 통계 로깅
            quality_score = response.metadata.get("quality_score", 0)
            duplicate_count = response.metadata.get("duplicate_count", 0)

            logger.info(f"📊 품질 통계: 점수 {quality_score:.1f}/10, 중복 {duplicate_count}개")

        else:
            logger.error(f"🚨 LangGraph 퀴즈 생성 실패: {response.error}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = f"LangGraph 퀴즈 생성 중 예상치 못한 오류: {str(e)}"

        logger.error(f"🚨 {error_msg} ({error_time:.2f}초)")

        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@router.get("/health")
async def langgraph_health_check():
    """🔍 LangGraph 퀴즈 서비스 상태 확인"""
    try:
        quiz_service = get_langgraph_quiz_service()

        return {
            "status": "healthy",
            "service": "LangGraph Quiz Service v2",
            "features": [
                "🚀 Agent 워크플로우",
                "🔥 실제 중복 제거",
                "🎯 진짜 2:6:2 비율",
                "⚡ 자동 재생성",
                "🧠 품질 보장"
            ],
            "workflow_nodes": [
                "initialize",
                "retrieve_contexts",
                "generate_questions",
                "validate_quality",
                "check_duplicates",
                "regenerate_bad_questions",
                "finalize"
            ],
            "improvements_over_v1": [
                "실제 작동하는 중복 제거",
                "정확한 문제 유형 비율",
                "Agent 기반 품질 보장",
                "구린 문제 자동 재생성",
                "다양성 있는 컨텍스트"
            ]
        }

    except Exception as e:
        logger.error(f"LangGraph 상태 확인 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"서비스 상태 확인 실패: {str(e)}"
        )


@router.get("/workflow/status")
async def get_workflow_status():
    """🔄 워크플로우 상태 정보"""
    return {
        "workflow_engine": "LangGraph",
        "workflow_type": "StateGraph",
        "nodes": {
            "initialize": "워크플로우 초기화 및 문서 확인",
            "retrieve_contexts": "다양성 있는 컨텍스트 검색",
            "generate_questions": "타입별 문제 생성",
            "validate_quality": "품질 검증 (7점 이상)",
            "check_duplicates": "중복 검사 (0.7 임계값)",
            "regenerate_bad_questions": "구린 문제 재생성",
            "finalize": "최종 결과 정리"
        },
        "conditional_edges": {
            "should_regenerate": "품질/중복/개수 기준으로 재생성 여부 결정"
        },
        "max_attempts": 3,
        "quality_threshold": 7.0,
        "duplicate_threshold": 0.7,
        "success_criteria": [
            "요청한 문제 수의 80% 이상 생성",
            "품질 점수 7.5 이상",
            "중복 문제 2개 이하"
        ]
    }


@router.post("/test/workflow", response_model=QuizResponse)
async def test_langgraph_workflow(
    document_id: str = "test-doc",
    num_questions: int = 5,
    quiz_service: LangGraphQuizService = Depends(get_langgraph_quiz_service)
) -> QuizResponse:
    """
    🧪 LangGraph 워크플로우 테스트
    간단한 테스트용 엔드포인트
    """

    try:
        test_request = QuizRequest(
            document_id=document_id,
            num_questions=num_questions,
            difficulty=Difficulty.MEDIUM,
            question_types=None  # 기본 2:6:2 비율 테스트
        )

        logger.info(f"🧪 LangGraph 워크플로우 테스트: {num_questions}문제")

        response = await quiz_service.generate_quiz(test_request)

        return response

    except Exception as e:
        logger.error(f"워크플로우 테스트 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"워크플로우 테스트 실패: {str(e)}"
        )