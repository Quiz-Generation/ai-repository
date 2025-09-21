"""
🎯 Quiz Generation API Routes
"""
import logging
import uuid
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Request
from ..docs import quiz_docs
from fastapi.responses import JSONResponse
from src.common.utils.logger import set_logger
from src.app.models import quiz as quiz_models
from src.app.service import quiz as quiz_service

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


# 2. AI 문제 생성 (POST 방식) - 스트리밍 방식으로 변경 (테스트용)
@router.post("/generate/test",
    summary="AI 문제 생성 (스트리밍) - 테스트용",
    description=quiz_docs.generate_quiz_description,
)
async def generate_quiz_test(
    request: Request,
    quiz_request: quiz_models.QuizGenerationRequest,
) -> JSONResponse:
    """
    AI를 사용하여 PDF 문서에서 문제를 생성합니다. (테스트용)
    문제 생성은 백그라운드에서 진행되며, 생성된 문제는 Redis 스트림을 통해 실시간으로 전송됩니다.
    테스트용 스트림 키: quiz-stream-test
    """
    # Pydantic 모델에서 자동으로 검증이 수행되므로 추가 검증 불필요
    
    # 고유 요청 ID 생성
    request_id = str(uuid.uuid4())
    
    # 백그라운드에서 문제 생성 시작 (테스트용)
    import asyncio
    asyncio.create_task(
        quiz_service.generate_quiz_from_file_streaming_test(
            request_id=request_id,
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
        "message": "문제 생성이 시작되었습니다. Redis 스트림을 통해 실시간으로 진행 상황을 확인할 수 있습니다. (테스트용)",
        "request_id": request_id,
        "stream_key": "quiz-stream-test",
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


# 문제 생성 스트림 구독 (테스트용)
@router.get("/stream/test",
    summary="문제 생성 스트림 구독 (테스트용)",
    description="문제 생성 진행 상황을 Redis 스트림에서 조회합니다. (테스트용)"
)
async def get_quiz_stream_test(
    count: int = 10
) -> JSONResponse:
    """
    문제 생성 스트림에서 메시지를 조회합니다. (테스트용)
    스프링 서버에서 이 엔드포인트를 주기적으로 호출하여 진행 상황을 확인할 수 있습니다.
    테스트용 스트림 키: quiz-stream-test
    """
    from src.common.redis.connect import get_quiz_stream_messages_test
    
    try:
        messages = await get_quiz_stream_messages_test(count)
        
        return JSONResponse(content={
            "success": True,
            "stream_key": "quiz-stream-test",
            "messages": messages,
            "message_count": len(messages),
            "last_updated": messages[0].get("data", {}).get("timestamp") if messages else None
        })
        
    except Exception as e:
        logger.error(f"테스트 스트림 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 📊 4. 문제 생성 옵션 조회
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