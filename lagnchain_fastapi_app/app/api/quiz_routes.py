"""
🎯 Quiz Generation API Routes
"""
import logging
import os
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..service.quiz_service import QuizService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz", tags=["quiz"])


# 🔧 Request Models
class QuizGenerationRequest(BaseModel):
    """문제 생성 요청 모델"""
    file_id: str  # 🔥 단일 파일 ID만 받음
    num_questions: int = 5
    difficulty: str = "medium"  # easy, medium, hard
    question_type: str = "multiple_choice"  # multiple_choice, true_false, short_answer, essay, fill_blank
    custom_topic: Optional[str] = None


# 🔧 서비스 의존성 주입
async def get_quiz_service() -> QuizService:
    """퀴즈 서비스 의존성 주입"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API 키가 설정되지 않았습니다. OPENAI_API_KEY 환경변수를 설정해주세요."
        )
    return QuizService(openai_api_key)


# 📋 1. 문제 생성 가능한 파일 목록 조회
@router.get("/available-files")
async def get_available_files(
    quiz_service: QuizService = Depends(get_quiz_service)
) -> JSONResponse:
    """
    📋 문제 생성 가능한 파일 목록 조회
    - 벡터 DB에 저장된 파일들 중 문제 생성에 적합한 파일들만 반환
    - 각 파일의 도메인, 언어, 청크 수 등 메타데이터 포함
    """
    try:
        logger.info("STEP_FILES 문제 생성 가능한 파일 목록 조회 시작")

        result = await quiz_service.get_available_files()

        if result["success"]:
            logger.info(f"SUCCESS 파일 목록 조회 완료: {result['total_files']}개")
        else:
            logger.error(f"ERROR 파일 목록 조회 실패: {result.get('error')}")

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"ERROR 파일 목록 조회 API 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 🤖 2. AI 문제 생성 (POST 방식)
@router.post("/generate")
async def generate_quiz(
    request: QuizGenerationRequest,
    quiz_service: QuizService = Depends(get_quiz_service)
) -> JSONResponse:
    """
    🤖 AI 기반 문제 생성 (단일 PDF 파일)

    **요청 파라미터:**
    - file_id: 대상 파일 ID (단일 파일)
    - num_questions: 생성할 문제 수 (1-10개)
    - difficulty: 난이도 (easy/medium/hard)
    - question_type: 문제 유형 (multiple_choice/true_false/short_answer/essay/fill_blank)
    - custom_topic: 특정 주제 지정 (선택사항)

    **AI 워크플로우:**
    1. 📄 문서 분석 → 2. 🎯 핵심 개념 추출 → 3. 🔑 키워드 매핑 → 4. ❓ 응용 문제 생성 → 5. ✅ 품질 검증
    """
    try:
        logger.info("🚀 AI 문제 생성 API 시작 (단일 파일)")

        # 기본 검증
        if not request.file_id:
            raise HTTPException(status_code=400, detail="file_id는 필수입니다")

        if not (1 <= request.num_questions <= 10):
            raise HTTPException(status_code=400, detail="문제 수는 1-10개 사이여야 합니다")

        if request.difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(status_code=400, detail="difficulty는 easy/medium/hard 중 하나여야 합니다")

        valid_types = ["multiple_choice", "true_false", "short_answer", "essay", "fill_blank"]
        if request.question_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"question_type은 {valid_types} 중 하나여야 합니다")

        logger.info(f"STEP_REQUEST 문제 생성 요청: {request.file_id}, {request.num_questions}개 문제, {request.difficulty} 난이도")

        # 문제 생성 실행
        result = await quiz_service.generate_quiz_from_file(
            file_id=request.file_id,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            question_type=request.question_type,
            custom_topic=request.custom_topic
        )

        if result["success"]:
            logger.info(f"🎉 SUCCESS AI 문제 생성 완료: {result['meta']['generated_count']}개 문제")
        else:
            logger.error(f"ERROR AI 문제 생성 실패: {result.get('error')}")

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERROR AI 문제 생성 API 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 🤖 3. AI 문제 생성 (간단한 GET 방식)
@router.get("/generate-simple")
async def generate_quiz_simple(
    file_id: str = Query(..., description="파일 ID (단일 파일)"),
    num_questions: int = Query(5, description="생성할 문제 수 (1-10개)"),
    difficulty: str = Query("medium", description="난이도 (easy/medium/hard)"),
    question_type: str = Query("multiple_choice", description="문제 유형"),
    custom_topic: Optional[str] = Query(None, description="특정 주제 (선택사항)"),
    quiz_service: QuizService = Depends(get_quiz_service)
) -> JSONResponse:
    """
    🤖 AI 기반 문제 생성 (간단한 GET 방식)
    - 단일 파일 ID로 간단하게 문제 생성
    """
    try:
        logger.info("🚀 AI 문제 생성 API (간단 버전) 시작")

        # QuizGenerationRequest 객체 생성
        request = QuizGenerationRequest(
            file_id=file_id,
            num_questions=num_questions,
            difficulty=difficulty,
            question_type=question_type,
            custom_topic=custom_topic
        )

        # 기존 generate_quiz 함수 재사용
        return await generate_quiz(request, quiz_service)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERROR AI 문제 생성 간단 API 실패: {e}")
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
                    "max_questions": 10,
                    "min_files": 1,
                    "max_files": 10
                }
            }
        }

        return JSONResponse(content=options)

    except Exception as e:
        logger.error(f"ERROR 옵션 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))