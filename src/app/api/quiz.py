"""
🎯 Quiz Generation API Routes
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from src.common.utils.logger import set_logger
from src.app.models import quiz as quiz_models
from src.app.service import quiz as quiz_service
from src.app.docs import quiz as quiz_docs

logger = set_logger("api.quiz")

router = APIRouter(tags=["quiz"])


# 📋 1. 문제 생성 가능한 파일 목록 조회
@router.get("/available-files",
    summary="문제 생성 가능한 파일 목록 조회",
    description=quiz_docs.available_files_description,
)
async def get_available_files(
    request: Request,
) -> JSONResponse:
    """
    📋 문제 생성 가능한 파일 목록 조회
    - 벡터 DB에 저장된 파일들 중 문제 생성에 적합한 파일들만 반환
    - 각 파일의 도메인, 언어, 청크 수 등 메타데이터 포함
    """
    return await quiz_service.get_available_files(
        logger=logger,
        vector_db=request.app.state.vector_db
    )



# 🤖 2. AI 문제 생성 (POST 방식)
@router.post("/generate",
    summary="AI 문제 생성",
    description=quiz_docs.generate_quiz_description,
)
async def generate_quiz(
    request: Request,
    quiz_request: quiz_models.QuizGenerationRequest,
) -> JSONResponse:
    # 문제 생성 실행
    return await quiz_service.generate_quiz_from_file(
        logger=logger,
        vector_db=request.app.state.vector_db,
        file_id=quiz_request.file_id,
        num_questions=quiz_request.num_questions,
        difficulty=quiz_request.difficulty,
        question_type=quiz_request.question_type,
        category=getattr(quiz_request, 'category', None) or "",
        sub_category=getattr(quiz_request, 'sub_category', None) or ""
    )



# 📊 3. 문제 생성 옵션 조회
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