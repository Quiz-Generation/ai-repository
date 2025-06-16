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
당신은 명문대학교의 베테랑 교수이자 교육 전문가입니다.
학생들의 심층적 이해와 창의적 사고를 돕는 고품질 문제를 출제하는 것이 목표입니다.

📚 **강의 내용**: {summary}

🎯 **출제 기준**:
- 문제 수: {num_questions}개
- 난이도: {difficulty}
- 문제 유형: {question_type}

{difficulty_guide}

{question_type_guide}

{distribution_guide}

**문제 품질 가이드**:
1. **다양성 확보**
   - 기본 개념 (20%)
   - 원리 이해 (30%)
   - 실제 응용 (30%)
   - 심화 분석 (20%)

2. **선택지 구성**
   - 정답: 깊은 이해와 분석이 필요한 답
   - 오답1: 부분적으로 맞지만 불완전한 답
   - 오답2: 흔한 오개념이나 실수
   - 오답3: 관련은 있지만 명백히 틀린 답

3. **해설 품질**
   - 정답의 근거 설명
   - 오답이 틀린 이유
   - 관련 개념 연결
   - 실제 적용 예시
   - 추가 학습 포인트

4. **문제 구성**
   - 명확한 문제 상황
   - 구체적인 조건
   - 실무/실생활 연계
   - 심층적 사고 유도
   - 창의적 해결방안 요구

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
      "problem_level": "basic/application/analysis/creation",
      "keywords": ["키워드1", "키워드2"],
      "concept_depth": "basic/understanding/application/analysis/creation",
      "real_world_connection": "실무/실생활 연계 설명",
      "critical_thinking": "비판적 사고 요구사항",
      "creative_thinking": "창의적 사고 요구사항"
    }}
  ]
}}
```

정확히 {num_questions}개의 문제를 생성해주세요.
각 문제는 위의 품질 가이드를 철저히 준수해야 합니다.
"""

    @staticmethod
    def get_difficulty_guide(difficulty: DifficultyLevel) -> str:
        """난이도별 가이드 (자연스러운 수준)"""
        guides = {
            DifficultyLevel.EASY: """
🟢 **EASY 난이도 가이드**:
- **핵심**: 기본 개념과 정의를 정확히 이해하고 있는지 확인
- **문제 스타일**:
  * 핵심 용어의 정확한 의미
  * 기본 특징 구별
  * 간단한 예시 적용
  * 실생활 연계
- **학생 반응**: "아, 이 정도는 기본이니까 맞춰야지"
- **난이도 느낌**: 공부했으면 확실히 맞출 수 있는 수준
- **문제 구성**:
  * 명확한 상황 설명
  * 직관적인 선택지
  * 구체적인 예시 포함
  * 실생활 연계
""",
            DifficultyLevel.MEDIUM: """
🟡 **MEDIUM 난이도 가이드**:
- **핵심**: 개념을 이해하고 상황에 맞게 적용할 수 있는지 평가
- **문제 스타일**:
  * 개념 간 비교 분석
  * 조건부 적용
  * 단계적 해결 과정
  * 실무 사례 적용
- **학생 반응**: "음... 좀 생각해봐야겠네, 하지만 풀 만해"
- **난이도 느낌**: 조금 고민하면 답을 찾을 수 있는 수준
- **문제 구성**:
  * 실제 상황 기반
  * 여러 개념 연계
  * 선택지 간 미묘한 차이
  * 구체적인 적용 사례
  * 실무적 고려사항
""",
            DifficultyLevel.HARD: """
🔴 **HARD 난이도 가이드**:
- **핵심**: 여러 개념을 종합하고 창의적으로 문제를 해결할 수 있는지 측정
- **문제 스타일**:
  * 복합 개념 융합
  * 제약 조건 하 최적화
  * 심화 원리 이해
  * 창의적 해결방안
  * 실무 프로젝트 적용
- **학생 반응**: "어렵네... 하지만 차근차근 분석하면 풀 수 있을 것 같아"
- **난이도 느낌**: 상위권 학생들도 고민이 필요한 수준
- **문제 구성**:
  * 복잡한 상황 설정
  * 여러 제약 조건
  * 최적화 요구
  * 실무적 고려사항
  * 창의적 사고 요구
"""
        }
        return guides.get(difficulty, "")

    @staticmethod
    def get_question_type_guide(question_type: QuestionType) -> str:
        """문제 유형별 가이드"""
        guides = {
            QuestionType.MULTIPLE_CHOICE: """
📝 **객관식 4지선다 가이드**:
- **정답**: 깊은 이해와 분석이 필요한 답
- **오답1**: 부분적으로 맞지만 완전하지 않은 답
- **오답2**: 흔히 헷갈리는 유사 개념
- **오답3**: 명백히 틀렸지만 공부 안 한 학생이 선택할 만한 답

**선택지 구성 원칙**:
1. 모든 선택지가 문법적으로 일관성 있게
2. 선택지 길이를 비슷하게
3. "모두 다르다", "모두 같다" 같은 함정 지양
4. 각 선택지가 독립적이고 명확하게
5. 실무/실생활 연계된 선택지 포함
6. 창의적 사고를 요구하는 선택지 포함
""",
            QuestionType.TRUE_FALSE: """
⭕ **참/거짓 문제 가이드**:
- **명확한 진술**:
  * 이론적 근거가 있는 진술
  * "항상", "절대" 등의 절대적 표현 신중 사용
  * 핵심 개념의 정확한 이해 여부 확인
  * 실무 적용 가능성 검토
  * 창의적 해결방안 고려

**문제 구성 원칙**:
1. 각 진술이 독립적이고 명확하게
2. 모호한 표현 지양
3. 부분적 진실 회피
4. 명확한 판단 기준 제시
5. 실무/실생활 연계
6. 창의적 사고 유도
""",
            QuestionType.SHORT_ANSWER: """
✏️ **단답형 문제 가이드**:
- **핵심 용어**:
  * 정확한 표현
  * 간결하고 명확한 답변
  * 주관적 해석 여지가 적은 객관적 답안
  * 실무 적용 가능성
  * 창의적 해결방안

**문제 구성 원칙**:
1. 명확한 답변 범위 제시
2. 모호한 표현 지양
3. 구체적인 답변 요구
4. 채점 기준 명시
5. 실무/실생활 연계
6. 창의적 사고 유도
""",
            QuestionType.ESSAY: """
📝 **서술형 문제 가이드**:
- **논리적 사고**:
  * 사고 과정 평가
  * 여러 개념 종합
  * 명확한 채점 기준
  * 실무 적용 가능성
  * 창의적 해결방안

**문제 구성 원칙**:
1. 명확한 문제 상황
2. 구체적인 답변 요구사항
3. 평가 기준 제시
4. 충분한 답변 시간 고려
5. 실무/실생활 연계
6. 창의적 사고 유도
""",
            QuestionType.FILL_BLANK: """
✏️ **빈칸 채우기 가이드**:
- **핵심 개념**:
  * 문맥상 핵심 용어
  * 논리적 흐름 완성
  * 전문 용어 사용
  * 실무 적용 가능성
  * 창의적 해결방안

**문제 구성 원칙**:
1. 명확한 문맥 제공
2. 빈칸의 역할 명시
3. 답변 범위 제시
4. 채점 기준 명시
5. 실무/실생활 연계
6. 창의적 사고 유도
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
  * 학생들이 "적당하네, 공부한 보람이 있어"라고 느낄 수준
  * 개념 이해 → 기본 적용 중심
  * 실제 교육 현장에서 사용 가능한 수준
  * 실무/실생활 연계
  * 창의적 사고 유도

- **응용 문제 {application_count}개 (10%)**: 실무/심화 응용 문제
  * 실제 상황 적용
  * 여러 개념 융합
  * 창의적 사고 요구
  * 실무/실생활 연계
  * 복잡한 문제 해결
  * 최적화 요구

**문제 품질 기준**:
1. **개념 깊이**
   - Basic: 기본 개념 이해
   - Understanding: 원리 이해
   - Application: 실제 적용
   - Analysis: 심층 분석
   - Creation: 창의적 해결

2. **실무 연계**
   - 실제 사례 기반
   - 실무적 고려사항
   - 현실적 제약조건
   - 실용적 해결방안
   - 창의적 접근

3. **교육적 가치**
   - 명확한 학습 목표
   - 단계적 이해 촉진
   - 실력 향상 기여
   - 자기주도학습 유도
   - 창의적 사고 개발

**중요**: 각 문제마다 다음을 명시해주세요:
- problem_level: "basic/application/analysis/creation"
- concept_depth: "basic/understanding/application/analysis/creation"
- real_world_connection: 실무/실생활 연계 설명
- critical_thinking: 비판적 사고 요구사항
- creative_thinking: 창의적 사고 요구사항
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
        return "당신은 학생들의 심층적 이해와 창의적 사고를 돕는 베테랑 교수입니다."