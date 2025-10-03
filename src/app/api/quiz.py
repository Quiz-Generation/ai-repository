"""
🎯 Quiz Generation API Routes
"""
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.app.models import quiz as quiz_models
from src.app.service import quiz as quiz_service
from src.common.utils.logger import set_logger

from ..docs import quiz_docs

logger = set_logger("api.quiz")

router = APIRouter(tags=["quiz"])


# 📋 1. 문제 생성 가능한 파일 목록 조회
@router.get("/available-files",
    summary="문제 생성 가능한 파일 목록 조회",
)
async def get_available_files(
    request: Request
) -> JSONResponse:
    """
    📋 문제 생성 가능한 파일 목록 조회
    - 벡터 DB에 저장된 파일들 중 문제 생성에 적합한 파일들만 반환
    - 각 파일의 도메인, 언어, 청크 수 등 메타데이터 포함
    """
    result = await quiz_service.get_available_files(
        logger=logger,
        vector_db=request.app.state.vector_db
    )
    return JSONResponse(content=result)


# 2. AI 문제 생성 (POST 방식) - 스트리밍 방식으로 변경
@router.post("/generate",
    summary="AI 문제 생성 (스트리밍)",
    description=quiz_docs.generate_quiz_description,
)
async def generate_quiz(
    request: Request,
    quiz_request: quiz_models.QuizGenerationRequest,
) -> JSONResponse:
    """
    AI를 사용하여 PDF 문서에서 문제를 생성합니다.
    문제 생성은 백그라운드에서 진행되며, 생성된 문제는 Redis 스트림을 통해 실시간으로 전송됩니다.
    """
    # Pydantic 모델에서 자동으로 검증이 수행되므로 추가 검증 불필요

    # 고유 요청 ID 생성
    request_id = str(uuid.uuid4())

    # 백그라운드에서 문제 생성 시작
    import asyncio
    asyncio.create_task(
        quiz_service.generate_quiz_from_file_streaming(
            request_id=request_id,
            quizset_idx=quiz_request.quizset_idx,
            user_idx=quiz_request.user_idx,
            logger=logger,
            vector_db=request.app.state.vector_db,
            file_id=quiz_request.file_id,
            num_questions=quiz_request.num_questions,
            difficulty=quiz_request.difficulty,
            question_type=quiz_request.question_type,
            custom_topic=quiz_request.custom_topic,
            category=getattr(quiz_request, 'category', None),
            sub_category=getattr(quiz_request, 'sub_category', None)
        )
    )

    return JSONResponse(content={
        "success": True,
        "message": "문제 생성이 시작되었습니다. Redis 스트림을 통해 실시간으로 진행 상황을 확인할 수 있습니다.",
        "request_id": request_id,
        "stream_key": "quiz-stream",
        "status": "started"
    })




# 문제 생성 스트림 구독 (스프링 서버용)
@router.get("/stream",
    summary="문제 생성 스트림 구독",
    description="문제 생성 진행 상황을 Redis 스트림에서 조회합니다."
)
async def get_quiz_stream(
    count: int = 10
) -> JSONResponse:
    """
    문제 생성 스트림에서 메시지를 조회합니다.
    스프링 서버에서 이 엔드포인트를 주기적으로 호출하여 진행 상황을 확인할 수 있습니다.
    """
    from src.common.redis.connect import get_quiz_stream_messages

    try:
        messages = await get_quiz_stream_messages(count)

        return JSONResponse(content={
            "success": True,
            "stream_key": "quiz-stream",
            "messages": messages,
            "message_count": len(messages),
            "last_updated": messages[0].get("data", {}).get("timestamp") if messages else None
        })

    except Exception as e:
        logger.error(f"스트림 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 🗑️ 3. 문제 생성 스트림 삭제
@router.delete("/stream",
    summary="문제 생성 스트림 삭제",
    description="quiz-stream의 모든 메시지를 삭제합니다."
)
async def clear_quiz_stream() -> JSONResponse:
    """
    quiz-stream의 모든 메시지를 삭제합니다.
    오래된 에러 메시지나 테스트 데이터를 정리할 때 사용합니다.
    """
    from src.common.redis.connect import clear_quiz_stream as clear_stream

    try:
        result = await clear_stream()

        return JSONResponse(content={
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "stream_key": "quiz-stream",
            "deleted_count": result.get("deleted_count", 0)
        })

    except Exception as e:
        logger.error(f"스트림 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 🗑️ 4. 문제 생성 스트림 메시지 일부 삭제 (오래된 것만)
@router.delete("/stream/old",
    summary="오래된 스트림 메시지 삭제",
    description="지정된 시간보다 오래된 메시지만 삭제합니다."
)
async def clear_old_quiz_stream_messages(
    max_age_seconds: int = 3600
) -> JSONResponse:
    """
    quiz-stream의 오래된 메시지만 삭제합니다.

    Args:
        max_age_seconds: 이 시간(초)보다 오래된 메시지를 삭제 (기본값: 3600초 = 1시간)
    """
    from src.common.redis.connect import clear_quiz_stream_messages

    try:
        result = await clear_quiz_stream_messages(max_age_seconds)

        return JSONResponse(content={
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "stream_key": "quiz-stream",
            "deleted_count": result.get("deleted_count", 0),
            "max_age_seconds": max_age_seconds
        })

    except Exception as e:
        logger.error(f"오래된 메시지 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 📊 5. 문제 생성 옵션 조회
@router.get("/options")
async def get_quiz_options() -> JSONResponse:
    """
    📊 문제 생성 시 사용 가능한 옵션들 조회
    - 지원되는 난이도 레벨
    - 지원되는 문제 유형
    - 각 옵션별 설명
    """
    try:
        options = {
            "success": True,
            "message": "문제 생성 옵션 조회 완료",
            "options": {
                "difficulties": [
                    {
                        "value": "easy",
                        "name": "쉬움",
                        "description": "기본 개념 암기, 단순 사실 확인",
                        "cognitive_level": "기억, 이해"
                    },
                    {
                        "value": "medium",
                        "name": "보통",
                        "description": "개념 이해와 적용, 관계 파악",
                        "cognitive_level": "적용, 분석"
                    },
                    {
                        "value": "hard",
                        "name": "어려움",
                        "description": "종합적 사고, 응용력, 창의적 해결",
                        "cognitive_level": "종합, 평가"
                    }
                ],
                "question_types": [
                    {
                        "value": "multiple_choice",
                        "name": "객관식 (4지선다)",
                        "description": "4개 선택지 중 정답 선택"
                    },
                    {
                        "value": "true_false",
                        "name": "참/거짓 (OX)",
                        "description": "진술이 참인지 거짓인지 판단"
                    },
                    {
                        "value": "short_answer",
                        "name": "단답형",
                        "description": "짧은 답안 작성"
                    },
                    {
                        "value": "essay",
                        "name": "서술형",
                        "description": "상세한 설명이나 논리적 답안 작성"
                    },
                    {
                        "value": "fill_blank",
                        "name": "빈칸 채우기",
                        "description": "문장의 빈칸에 적절한 단어나 구문 입력"
                    }
                ],
                "constraints": {
                    "min_questions": 1,
                    "max_questions": 50,
                    "min_files": 1,
                    "max_files": 10
                }
            }
        }

        return JSONResponse(content=options)

    except Exception as e:
        logger.error(f"ERROR 옵션 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))