# 🎯 Quiz Generation Prompt Management System

프롬프트 템플릿과 가이드를 체계적으로 관리하는 모듈입니다.

## 📁 폴더 구조

```
app/agent/prompt/
├── __init__.py              # 모듈 초기화
├── quiz_prompt_manager.py   # 메인 프롬프트 관리자
└── README.md               # 사용 가이드 (이 파일)
```

## 🔧 주요 클래스

### `QuizPromptManager`
문제 생성을 위한 모든 프롬프트를 관리하는 핵심 클래스

#### 주요 메서드:

```python
# 1. 기본 프롬프트 템플릿
get_base_prompt_template() -> str

# 2. 난이도별 가이드 (EASY/MEDIUM/HARD)
get_difficulty_guide(difficulty: DifficultyLevel) -> str

# 3. 문제 유형별 가이드 (객관식/OX/단답형/서술형/빈칸)
get_question_type_guide(question_type: QuestionType) -> str

# 4. 문제 분포 가이드 (90% 일반 + 10% 응용)
get_distribution_guide(num_questions: int) -> str

# 5. 최종 프롬프트 생성 (모든 템플릿 조합)
generate_final_prompt(summary, num_questions, difficulty, question_type) -> str

# 6. 시스템 메시지
get_system_message() -> str
```

## 🎯 사용 예시

```python
from app.agent.prompt import QuizPromptManager
from app.agent.prompt.quiz_prompt_manager import DifficultyLevel, QuestionType

# 프롬프트 관리자 초기화
prompt_manager = QuizPromptManager()

# 최종 프롬프트 생성
final_prompt = prompt_manager.generate_final_prompt(
    summary="동적계획법의 핵심 개념...",
    num_questions=10,
    difficulty=DifficultyLevel.MEDIUM,
    question_type=QuestionType.MULTIPLE_CHOICE
)

# 시스템 메시지
system_msg = prompt_manager.get_system_message()
```

## ⚖️ 문제 분포 시스템

### 90% 일반 문제 + 10% 응용 문제

- **일반 문제 (90%)**: 해당 난이도에 맞는 표준적인 문제
  - 교과서/강의 내용 기반
  - 학생들이 "적당하네, 공부한 보람이 있어"라고 느낄 수준
  - 개념 이해 → 기본 적용 중심

- **응용 문제 (10%)**: 실무/심화 응용 문제
  - 실제 상황 적용
  - 여러 개념 융합
  - 창의적 사고 요구

## 📝 난이도별 특성

### 🟢 EASY
- **핵심**: 기본 개념과 정의 이해 확인
- **학생 반응**: "아, 이 정도는 기본이니까 맞춰야지"
- **예시**: "동적계획법이란?", "DP의 특징은?"

### 🟡 MEDIUM
- **핵심**: 개념 이해하고 상황별 적용
- **학생 반응**: "음... 좀 생각해봐야겠네, 하지만 풀 만해"
- **예시**: "DP vs 그리디 언제 사용?", "첫 번째 단계는?"

### 🔴 HARD
- **핵심**: 여러 개념 종합하고 창의적 문제해결
- **학생 반응**: "어렵네... 하지만 차근차근 분석하면 풀 수 있을 것 같아"
- **예시**: "DP + 그래프 + 최적화", "실패 경우와 대안은?"

## 🔄 프롬프트 수정 가이드

### 1. 난이도 조정
`get_difficulty_guide()` 메서드에서 각 난이도별 설명 수정

### 2. 새로운 문제 유형 추가
`get_question_type_guide()` 메서드에 새로운 QuestionType 추가

### 3. 분포 비율 변경
`get_distribution_guide()` 메서드에서 application_count 계산식 수정

### 4. 기본 템플릿 변경
`get_base_prompt_template()` 메서드에서 전체 구조 수정

## 🚀 확장 가능성

- **도메인별 특화 프롬프트**: AWS, 알고리즘, 심리학 등
- **언어별 프롬프트**: 한국어, 영어 지원
- **시험 유형별 프롬프트**: 수능형, 토익형, 자격증형
- **개인화 프롬프트**: 학습자 수준별 맞춤형

---

💡 **수정 시 주의사항**:
프롬프트 수정 후 반드시 테스트를 통해 문제 생성 품질과 수량을 확인하세요!