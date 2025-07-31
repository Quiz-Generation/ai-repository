
# 🔧 Request Models
from typing import Optional
from pydantic import BaseModel


class QuizGenerationRequest(BaseModel):
    """문제 생성 요청 모델"""
    file_id: str  # 🔥 단일 파일 ID만 받음
    num_questions: int = 10
    difficulty: str = "medium"  # easy, medium, hard
    question_type: str = "multiple_choice"  # multiple_choice, true_false, short_answer, essay, fill_blank
    custom_topic: Optional[str] = None  # 특정 주제 (선택사항)
    category: Optional[str] = None  # 대분류(예: 컴퓨터 공학)
    sub_category: Optional[str] = None  # 소분류(예: 데이터베이스)
