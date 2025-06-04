"""
퀴즈 생성 시스템 스키마 정의
PDF 기반 RAG 퀴즈 생성을 위한 데이터 모델
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid
from datetime import datetime


class LLMProvider(Enum):
    """지원하는 LLM 제공업체"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    KOREAN_LOCAL = "korean_local"  # 추후 국내 모델용
    HUGGINGFACE = "huggingface"


@dataclass
class LLMConfig:
    """LLM 설정"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    language: str = "ko"
    custom_params: Optional[Dict[str, Any]] = None


class Difficulty(Enum):
    """문제 난이도"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionType(Enum):
    """문제 유형"""
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    FILL_BLANK = "fill_blank"
    TRUE_FALSE = "true_false"


@dataclass
class Question:
    """문제 데이터 클래스"""
    question: str
    question_type: QuestionType
    correct_answer: str
    options: Optional[List[str]] = None
    explanation: str = ""
    difficulty: Difficulty = Difficulty.MEDIUM
    source_context: str = ""
    topic: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """문제 유형별 검증"""
        if self.question_type == QuestionType.MULTIPLE_CHOICE and not self.options:
            raise ValueError("객관식 문제는 선택지가 필요합니다")
        if self.question_type == QuestionType.TRUE_FALSE and self.correct_answer not in ["True", "False"]:
            raise ValueError("참/거짓 문제의 정답은 True 또는 False여야 합니다")


@dataclass
class QuizRequest:
    """퀴즈 생성 요청"""
    document_id: str
    num_questions: int = 5
    difficulty: Difficulty = Difficulty.MEDIUM
    question_types: Optional[List[QuestionType]] = None
    topics: Optional[List[str]] = None
    language: str = "ko"  # 언어 설정 (나중에 모델 스위칭용)
    custom_prompt: Optional[str] = None  # 커스텀 프롬프트

    def __post_init__(self):
        """요청 검증"""
        if self.num_questions < 1 or self.num_questions > 50:
            raise ValueError("문제 수는 1-50개 사이여야 합니다")
        if not self.document_id.strip():
            raise ValueError("문서 ID는 필수입니다")


@dataclass
class QuizResponse:
    """퀴즈 생성 응답"""
    quiz_id: str
    document_id: str
    questions: List[Question]
    total_questions: int
    difficulty: Difficulty
    generation_time: float
    success: bool = True
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """응답 데이터 정리"""
        self.total_questions = len(self.questions)


@dataclass
class RAGContext:
    """RAG 검색 컨텍스트"""
    text: str
    similarity: float
    source: str
    chunk_index: int
    topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicAnalysis:
    """문서 토픽 분석 결과"""
    topic: str
    confidence: float
    keywords: List[str]
    context_chunks: List[str]
    question_potential: int  # 1-10 점수


@dataclass
class QuizGenerationStats:
    """퀴즈 생성 통계"""
    total_contexts_retrieved: int
    avg_context_similarity: float
    topics_analyzed: List[str]
    generation_attempts: int
    success_rate: float
    llm_model_used: str
    processing_time_breakdown: Dict[str, float]


# API 응답용 Pydantic 모델들 (FastAPI용)
try:
    from pydantic import BaseModel, Field
    from typing import Union

    class QuestionAPI(BaseModel):
        """API용 문제 모델"""
        question: str
        question_type: str
        correct_answer: str
        options: Optional[List[str]] = None
        explanation: str = ""
        difficulty: str = "medium"
        topic: str = ""


    class QuizRequestAPI(BaseModel):
        """퀴즈 생성 요청 API 모델"""
        document_id: str = Field(..., description="업로드된 PDF 문서 ID")
        num_questions: int = Field(5, ge=1, le=20, description="생성할 문제 수 (1-20개)")
        difficulty: str = Field("medium", description="난이도 (easy/medium/hard)")
        question_types: Optional[List[str]] = Field(
            None,
            description="문제 유형 (생략 시 자동 선택): multiple_choice, short_answer, fill_blank, true_false"
        )
        topics: Optional[List[str]] = Field(
            None,
            description="🤖 토픽 힌트 (선택사항): PDF에서 자동 추출된 토픽이 우선 사용되며, 여기 지정한 토픽은 추가 힌트로 활용됩니다"
        )
        language: str = Field("ko", description="언어 (ko/en)")

        class Config:
            schema_extra = {
                "example": {
                    "document_id": "f7dbd017-426e-4919-8a88-feda68949615",
                    "num_questions": 5,
                    "difficulty": "medium",
                    "question_types": ["multiple_choice", "short_answer"],
                    "topics": ["알고리즘", "자료구조"],  # 힌트로만 사용
                    "language": "ko"
                }
            }


    class QuizResponseAPI(BaseModel):
        """API용 퀴즈 생성 응답"""
        quiz_id: str
        document_id: str
        questions: List[QuestionAPI]
        total_questions: int
        difficulty: str
        generation_time: float
        success: bool = True
        error: str = ""
        created_at: str
        generation_stats: Optional[Dict[str, Any]] = None

except ImportError:
    # Pydantic이 없어도 동작하도록
    pass