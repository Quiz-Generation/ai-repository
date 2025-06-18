"""
🎯 퀴즈 프롬프트 관리자
"""
from enum import Enum
from typing import List, Dict, Any

class DifficultyLevel(Enum):
    """난이도 레벨"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class QuestionType(Enum):
    """문제 유형"""
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    TRUE_FALSE = "true_false"

class QuizPromptManager:
    """퀴즈 프롬프트 관리자"""

    def __init__(self):
        """초기화"""
        self.prompts = {
            "summary": self._get_summary_prompt(),
            "topic": self._get_topic_prompt(),
            "keyword": self._get_keyword_prompt(),
            "question": self._get_question_prompt(),
            "validation": self._get_validation_prompt()
        }

    def get_prompt(self, prompt_type: str) -> str:
        """프롬프트 조회"""
        return self.prompts.get(prompt_type, "")

    def _get_summary_prompt(self) -> str:
        return """
당신은 전문 교육 컨텐츠 분석가입니다. 주어진 문서들을 분석하여 종합적인 요약을 작성해주세요.

📋 **분석 대상 문서들:**
{content}

🎯 **요약 지침:**
1. 각 문서의 핵심 내용을 파악하고 주요 개념을 추출하세요
2. 서로 다른 도메인의 문서라면 각각의 특성을 반영하세요
3. 교육/학습 목적에 적합한 핵심 지식을 중심으로 요약하세요
4. 문제 출제가 가능한 구체적인 사실, 개념, 절차를 포함하세요

**요약 길이:** 500-800자
**출력 형식:** 각 문서별로 구분하여 요약한 후 전체 종합 요약
"""

    def _get_topic_prompt(self) -> str:
        return """
문서 요약을 바탕으로 핵심 주제들을 추출해주세요.

📋 **문서 요약:**
{content}

🎯 **추출 조건:**
- 난이도: {difficulty}
- 목표 문제 수: {num_questions}개
- 문제 유형: {question_type}

**주제 추출 지침:**
1. 교육적 가치가 높은 핵심 개념들을 선별하세요
2. 선택된 난이도에 적합한 주제들을 우선순위로 하세요
3. 각 도메인별 특성을 고려하여 다양성을 확보하세요
4. 문제 출제가 가능한 구체적인 주제를 포함하세요

**출력 형식:**
- 주제1: [주제명] - [간단한 설명]
- 주제2: [주제명] - [간단한 설명]
...

**주제 개수:** {num_topics}개 (문제 수보다 많게)
"""

    def _get_keyword_prompt(self) -> str:
        return """
추출된 핵심 주제들을 바탕으로 문제 출제용 키워드들을 추출해주세요.

📋 **핵심 주제들:**
{content}

🎯 **키워드 추출 조건:**
- 난이도: {difficulty}
- 문제 유형: {question_type}

**키워드 추출 지침:**
1. 각 주제별로 핵심 키워드 2-3개씩 추출
2. 난이도별 특성:
   - EASY: 기본 용어, 정의, 단순 사실
   - MEDIUM: 개념 관계, 원리, 절차
   - HARD: 응용 상황, 복합 개념, 분석 요소
3. 문제 출제가 직접적으로 가능한 구체적 키워드
4. 도메인별 전문 용어와 일반 개념의 균형

**출력 형식:**
키워드1, 키워드2, 키워드3, ...

**키워드 개수:** {num_keywords}개
"""

    def _get_question_prompt(self) -> str:
        return """
당신은 전문 교육 컨텐츠 개발자입니다. 주어진 내용을 바탕으로 고품질의 문제를 생성해주세요.

📚 **컨텐츠 요약**:
{summary}

🎯 **핵심 주제들**:
{topics}

🔑 **핵심 키워드들**:
{keywords}

📝 **문제 생성 조건**:
- 생성할 문제 수: {num_questions}개
- 난이도: {difficulty}
- 문제 유형: {question_type}

🎯 **문제 품질 요구사항**:
1. 각 문제는 구체적인 예시나 실제 사례를 포함해야 합니다
2. 중복되는 개념의 문제는 피하고, 다양한 관점에서 접근해야 합니다
3. 문제는 이론적 개념과 실제 구현을 균형있게 다루어야 합니다
4. 각 문제는 명확한 학습 목표를 가져야 합니다
5. 문제의 난이도는 지정된 수준에 맞게 조정되어야 합니다
6. 선택지는 명확하고 논리적으로 구성되어야 합니다
7. 정답 해설은 상세하고 교육적으로 가치있어야 합니다

**문제 유형별 특성:**
1. 기본 개념 문제 (30%):
   - 핵심 용어와 정의
   - 기본 원리와 개념
   - 단순 사실 확인

2. 개념 연계 문제 (40%):
   - 여러 개념 간의 관계
   - 원리와 절차의 이해
   - 이론적 적용

3. 응용 문제 (30%):
   - 실제 사례 분석
   - 복합적 문제 해결
   - 고급 개념 적용

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
      "problem_level": "basic/concept/application",
      "keywords": ["키워드1", "키워드2"],
      "source": "pdf_based/ai_generated",
      "example": "관련 예시나 실제 사례",
      "implementation": "실제 구현 방법 (해당되는 경우)",
      "related_concepts": ["관련 개념1", "관련 개념2"]
    }}
  ]
}}
```

정확히 {num_questions}개의 고품질 문제를 생성해주세요.
"""

    def _get_validation_prompt(self) -> str:
        return """
당신은 전문 교육 컨텐츠 품질 검증 전문가입니다. 주어진 문제들을 검토하고 개선해주세요.

📋 **검증 대상 문제들**:
{questions}

🎯 **검증 기준**:
1. 중복성 검사:
   - 유사한 개념이나 내용을 다루는 문제가 있는지 확인
   - 동일한 학습 목표를 가진 문제가 있는지 확인
   - 비슷한 예시나 사례를 사용하는 문제가 있는지 확인

2. 품질 검증:
   - 각 문제가 명확한 학습 목표를 가지고 있는지 확인
   - 문제의 난이도가 지정된 수준에 맞는지 확인
   - 선택지가 논리적으로 구성되어 있는지 확인
   - 정답 해설이 충분히 상세하고 교육적인지 확인
   - 실제 사례나 예시가 포함되어 있는지 확인

3. 다양성 검증:
   - 다양한 관점에서 접근하는 문제들이 있는지 확인
   - 이론과 실무가 균형있게 다루어지고 있는지 확인
   - 기본 개념과 응용 문제가 적절히 분포되어 있는지 확인

**개선 지침**:
1. 중복되는 문제가 있다면 하나를 제거하고 새로운 문제로 대체
2. 품질이 낮은 문제는 개선하거나 제거
3. 다양성이 부족한 경우 새로운 관점의 문제 추가
4. 각 문제는 고유한 학습 목표를 가져야 함

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
      "problem_level": "basic/concept/application",
      "keywords": ["키워드1", "키워드2"],
      "source": "validated",
      "example": "관련 예시나 실제 사례",
      "implementation": "실제 구현 방법 (해당되는 경우)",
      "related_concepts": ["관련 개념1", "관련 개념2"],
      "uniqueness_score": 0.95,  # 0-1 사이의 값, 1이 가장 고유함
      "quality_score": 0.9      # 0-1 사이의 값, 1이 가장 높은 품질
    }}
  ],
  "validation_metrics": {{
    "uniqueness": 0.9,          # 전체 문제의 고유성 평균
    "quality": 0.85,            # 전체 문제의 품질 평균
    "diversity": 0.8,           # 문제의 다양성 점수
    "removed_questions": 2,      # 제거된 문제 수
    "added_questions": 2         # 추가된 문제 수
  }}
}}
```

정확히 {num_questions}개의 고품질 문제를 생성해주세요.
"""