"""
🎯 Quiz Generation Prompt Manager
프롬프트 템플릿과 가이드를 체계적으로 관리하는 모듈
"""

from typing import Dict, Any
from enum import Enum


class DifficultyLevel(Enum):
    """문제 난이도 레벨"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionType(Enum):
    """문제 타입"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    FILL_BLANK = "fill_blank"


class QuizPromptManager:
    """퀴즈 생성을 위한 프롬프트 관리자"""

    @staticmethod
    def get_base_prompt_template() -> str:
        """기본 프롬프트 템플릿"""
        return """
당신은 명문대학교의 베테랑 교수입니다.
학생들의 수준에 맞는 적절한 난이도의 문제를 출제하는 것이 목표입니다.

📚 **강의 내용**: {summary}

🎯 **출제 기준**:
- 문제 수: {num_questions}개
- 난이도: {difficulty}
- 문제 유형: {question_type}

{difficulty_guide}

{question_type_guide}

{distribution_guide}

**출력 형식**:
```json
{{
  "questions": [
    {{
      "id": 1,
      "question": "문제 내용",
      "type": "{question_type}",
      "difficulty": "{difficulty}",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "correct_answer": "정답",
      "explanation": "정답 해설",
      "learning_objective": "학습 목표",
      "problem_level": "basic 또는 application",
      "keywords": ["키워드1", "키워드2"]
    }}
  ]
}}
```

정확히 {num_questions}개의 문제를 생성해주세요.
"""

    @staticmethod
    def get_difficulty_guide(difficulty: DifficultyLevel) -> str:
        """난이도별 가이드 (자연스러운 수준)"""
        guides = {
            DifficultyLevel.EASY: """
🟢 **EASY 난이도 가이드**:
- **핵심**: 기본 개념과 정의를 정확히 이해하고 있는지 확인
- **문제 스타일**:
  * 용어의 정확한 의미 (예: "동적계획법이란?")
  * 기본 특징 구별 (예: "다음 중 DP의 특징은?")
  * 간단한 예시 적용 (예: "피보나치 수열에서 DP를 쓰는 이유는?")
- **학생 반응**: "아, 이 정도는 기본이니까 맞춰야지"
- **난이도 느낌**: 공부했으면 확실히 맞출 수 있는 수준
""",
            DifficultyLevel.MEDIUM: """
🟡 **MEDIUM 난이도 가이드**:
- **핵심**: 개념을 이해하고 상황에 맞게 적용할 수 있는지 평가
- **문제 스타일**:
  * 개념 간 비교 분석 (예: "DP vs 그리디 알고리즘 언제 사용?")
  * 조건부 적용 (예: "이 상황에서 어떤 방법이 더 효율적?")
  * 단계적 해결 과정 (예: "이 문제를 DP로 해결할 때 첫 번째 단계는?")
- **학생 반응**: "음... 좀 생각해봐야겠네, 하지만 풀 만해"
- **난이도 느낌**: 조금 고민하면 답을 찾을 수 있는 수준
""",
            DifficultyLevel.HARD: """
🔴 **HARD 난이도 가이드**:
- **핵심**: 여러 개념을 종합하고 창의적으로 문제를 해결할 수 있는지 측정
- **문제 스타일**:
  * 복합 개념 융합 (예: "DP + 그래프 + 최적화를 모두 고려하면?")
  * 제약 조건 하 최적화 (예: "메모리와 시간을 모두 고려한 최선책은?")
  * 심화 원리 이해 (예: "이 알고리즘이 실패하는 경우와 대안은?")
- **학생 반응**: "어렵네... 하지만 차근차근 분석하면 풀 수 있을 것 같아"
- **난이도 느낌**: 상위권 학생들도 고민이 필요한 수준
"""
        }
        return guides.get(difficulty, "")

    @staticmethod
    def get_question_type_guide(question_type: QuestionType) -> str:
        """문제 유형별 가이드"""
        guides = {
            QuestionType.MULTIPLE_CHOICE: """
📝 **객관식 4지선다 가이드**:
- 정답: 명확하고 완전한 답
- 오답1: 부분적으로 맞지만 완전하지 않은 답
- 오답2: 흔히 헷갈리는 유사 개념
- 오답3: 명백히 틀렸지만 공부 안 한 학생이 선택할 만한 답
""",
            QuestionType.TRUE_FALSE: """
⭕ **참/거짓 문제 가이드**:
- 명확한 이론적 근거가 있는 진술
- "항상", "절대" 등의 절대적 표현 신중 사용
- 핵심 개념의 정확한 이해 여부 확인
""",
            QuestionType.SHORT_ANSWER: """
✏️ **단답형 문제 가이드**:
- 핵심 용어의 정확한 표현
- 간결하고 명확한 답변
- 주관적 해석 여지가 적은 객관적 답안
""",
            QuestionType.ESSAY: """
📝 **서술형 문제 가이드**:
- 논리적 사고 과정을 평가할 수 있는 문제
- 여러 개념을 종합하여 설명하는 능력 측정
- 명확한 채점 기준이 있는 구조화된 답안 요구
""",
            QuestionType.FILL_BLANK: """
✏️ **빈칸 채우기 가이드**:
- 문맥상 핵심이 되는 용어나 개념
- 논리적 흐름을 완성하는 중요한 단어
- 전문 용어의 정확한 사용법 확인
"""
        }
        return guides.get(question_type, "")

    @staticmethod
    def get_distribution_guide(num_questions: int) -> str:
        """문제 분포 가이드 (90% 일반 + 10% 응용)"""
        application_count = max(1, int(num_questions * 0.1))  # 최소 1개
        basic_count = num_questions - application_count

        return f"""
📊 **문제 구성 분포**:
- **일반 문제 {basic_count}개 (90%)**: 해당 난이도에 맞는 표준적인 문제
  * 교과서/강의 내용 기반
  * 학생들이 "아, 이 정도면 적당하네, 공부한 보람이 있어"라고 느낄 수준
  * 개념 이해 → 기본 적용 중심

- **응용 문제 {application_count}개 (10%)**: 실무/심화 응용 문제
  * 실제 상황 적용
  * 여러 개념 융합
  * 창의적 사고 요구

**중요**: 각 문제마다 "problem_level"을 "basic" 또는 "application"으로 명시해주세요.
"""

    @classmethod
    def generate_final_prompt(
        cls,
        summary: str,
        num_questions: int,
        difficulty: DifficultyLevel,
        question_type: QuestionType
    ) -> str:
        """최종 프롬프트 생성"""
        base_template = cls.get_base_prompt_template()
        difficulty_guide = cls.get_difficulty_guide(difficulty)
        question_type_guide = cls.get_question_type_guide(question_type)
        distribution_guide = cls.get_distribution_guide(num_questions)

        return base_template.format(
            summary=summary[:1000],
            num_questions=num_questions,
            difficulty=difficulty.value,
            question_type=question_type.value,
            difficulty_guide=difficulty_guide,
            question_type_guide=question_type_guide,
            distribution_guide=distribution_guide
        )

    @staticmethod
    def get_system_message() -> str:
        """시스템 메시지"""
        return "당신은 학생들의 수준에 맞는 적절한 문제를 출제하는 베테랑 교수입니다."