
# 🔧 Request Models
from typing import Optional
from pydantic import BaseModel, field_validator


class QuizGenerationRequest(BaseModel):
    """문제 생성 요청 모델"""
    user_idx: int 
    file_id: str  # 단일 파일 ID만 받음
    num_questions: int = 10
    difficulty: str = "medium"  # easy, medium, hard
    question_type: str = "multiple_choice"  # multiple_choice, true_false, short_answer, essay, fill_blank
    custom_topic: Optional[str] = None  # 특정 주제 (선택사항)
    category: Optional[str] = None  # 대분류(예: 컴퓨터 공학)
    sub_category: Optional[str] = None  # 소분류(예: 데이터베이스)

    @field_validator('user_idx')
    @classmethod
    def validate_user_idx(cls, v):
        if v is None:
            raise ValueError("user_idx is required")
        if not isinstance(v, int) or v <= 0:
            raise ValueError("user_idx must be a positive integer")
        return v

    @field_validator('file_id')
    @classmethod
    def validate_file_id(cls, v):
        if not v or not isinstance(v, str) or v.strip() == "":
            raise ValueError("file_id is required and cannot be empty")
        
        # 잘못된 파일 ID 패턴 검증
        invalid_patterns = ['string', 'test', 'example', 'sample', 'dummy', 'placeholder']
        if v.lower() in invalid_patterns:
            raise ValueError(f"Invalid file_id pattern: '{v}'. Please use a real uploaded file ID.")
        
        # 파일 ID 형식 검증 (실제 파일 ID는 'file_'로 시작해야 함)
        if not v.startswith('file_'):
            raise ValueError(f"Invalid file_id format: '{v}'. File ID must start with 'file_'.")
        
        return v.strip()

    @field_validator('num_questions')
    @classmethod
    def validate_num_questions(cls, v):
        if not isinstance(v, int) or v < 1 or v > 50:
            raise ValueError("num_questions must be an integer between 1 and 50")
        return v

    @field_validator('difficulty')
    @classmethod
    def validate_difficulty(cls, v):
        valid_difficulties = ['easy', 'medium', 'hard']
        if v not in valid_difficulties:
            raise ValueError(f"difficulty must be one of: {', '.join(valid_difficulties)}")
        return v

    @field_validator('question_type')
    @classmethod
    def validate_question_type(cls, v):
        valid_types = ['multiple_choice', 'true_false', 'short_answer', 'essay', 'fill_blank']
        if v not in valid_types:
            raise ValueError(f"question_type must be one of: {', '.join(valid_types)}")
        return v