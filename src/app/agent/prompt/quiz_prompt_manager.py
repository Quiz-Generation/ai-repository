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
            "validation": self._get_validation_prompt(),
            "combined_preprocessing": self._get_combined_preprocessing_prompt()
        }

        # 카테고리별 특화 프롬프트
        self.category_prompts = {
            "it": self._get_it_category_prompt(),
            "certification": self._get_certification_category_prompt(),
            "general": self._get_general_category_prompt()
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
- 전체 시험 난이도: {difficulty} (각 문제는 이 난이도를 중심으로 분산)
- 문제 유형: {question_type}

🎯 **개별 문제 난이도 분산 지침**:
전체 시험 난이도가 {difficulty}인 경우, 각 문제의 난이도는 다음과 같이 분산되어야 합니다:

- **EASY 시험**: 40% 쉬운 문제, 40% 보통 문제, 20% 어려운 문제
- **MEDIUM 시험**: 20% 쉬운 문제, 50% 보통 문제, 30% 어려운 문제
- **HARD 시험**: 10% 쉬운 문제, 30% 보통 문제, 60% 어려운 문제

각 문제는 다음 중 하나의 난이도를 가져야 합니다:
- **쉬운 문제**: 기본 개념, 정의, 단순 사실 확인
- **보통 문제**: 개념 간 관계, 원리 이해, 절차적 사고
- **어려운 문제**: 복합적 분석, 응용, 고급 개념 활용

🎯 **문제 품질 요구사항**:
1. 각 문제는 구체적인 예시나 실제 사례를 포함해야 합니다
2. 중복되는 개념의 문제는 피하고, 다양한 관점에서 접근해야 합니다
3. 문제는 이론적 개념과 실제 구현을 균형있게 다루어야 합니다
4. 각 문제는 명확한 학습 목표를 가져야 합니다
5. 문제의 난이도는 지정된 수준에 맞게 조정되어야 합니다
6. 선택지는 명확하고 논리적으로 구성되어야 합니다
7. 정답 해설은 친절하고 교육적으로 가치있어야 합니다

**정답 해설 작성 지침(아주 중요!)**

🎯 **해설 품질 기준:**
1. **개념적 이해**: 왜 이 답이 정답인지 개념적으로 명확히 설명
2. **오답 분석**: 다른 선택지가 왜 틀렸는지 구체적으로 분석
3. **실무 연관**: 실제 상황에서 어떻게 적용되는지 설명
4. **학습 포인트**: 이 문제를 통해 배울 수 있는 핵심 개념 강조

📝 **해설 작성 방법:**
- **시작**: "정답은 ~입니다. 이는 ~하기 때문입니다."
- **개념 설명**: "~의 핵심 개념은 ~이며, 이 문제에서는 ~한 상황을 다루고 있습니다."
- **오답 분석**: "다른 선택지들은 ~한 이유로 틀렸습니다. 특히 ~는 ~와 혼동하기 쉬운 개념입니다."
- **실무 적용**: "실제로는 ~할 때 이 개념이 중요하며, ~한 상황에서 활용됩니다."
- **학습 포인트**: "이 문제를 통해 ~에 대한 이해를 높일 수 있으며, ~와의 차이점도 명확히 구분할 수 있습니다."

🔍 **구체적 설명 요소:**
- 핵심 개념의 정의와 특징
- 다른 개념과의 차이점
- 실제 적용 사례나 예시
- 자주 실수하는 부분이나 함정
- 관련된 추가 학습 포인트
- 실무에서의 중요성이나 활용법

💡 **해설 길이**: 3-5문장으로 구성하여 충분히 상세하고 교육적 가치가 있게 작성

**중복 방지 지침(아주 중요!)**:
- 이미 등장한 개념, 주제, 또는 문제와 유사한 문제는 절대 생성하지 마세요.
- 표현만 다르고 본질이 같은 문제(예: 질문의 단어만 바꾼 경우)도 중복으로 간주합니다.
- 각 문제는 반드시 고유한 학습 목표, 예시, 설명을 가져야 하며, 다른 문제와 명확히 구분되어야 합니다.
- 동일한 키워드, 개념, 사례, 정답을 반복하지 마세요.
- 문제의 질문, 보기, 해설, 예시, 학습 목표가 모두 중복되지 않도록 하세요.
- 중복이 의심되는 경우, 완전히 새로운 주제/관점/예시로 문제를 생성하세요.

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
      "difficulty": "easy/medium/hard",
      "exam_difficulty": "{difficulty}",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "correct_answer": "정답",
      "correct_answer_number": 1,
      "explanation": "정답 해설",
      "learning_objective": "학습 목표",
      "problem_level": "basic/concept/application",
      "keywords": ["키워드1", "키워드2"],
    }}
  ]
}}
```

**🚨 절대 금지 사항 (이 규칙을 위반하면 안 됩니다):**

1. 선택지 작성 시 절대 금지:
   - ❌ "1. 1. 내용" (번호 중복 금지)
   - ❌ "1번. 내용" (번외 형식 금지)
   - ❌ "1) 내용" (괄호 형식 금지)
   - ❌ "A. 내용" (알파벳 형식 금지)

   ✅ 올바른 형식만 사용:
   - "1. 내용"
   - "2. 내용"
   - "3. 내용"
   - "4. 내용"

2. 정답 작성 시 절대 금지:
   - ❌ correct_answer_number: null
   - ❌ correct_answer_number: "1"
   - ❌ correct_answer_number: 0

   ✅ 올바른 형식만 사용:
   - correct_answer_number: 1
   - correct_answer_number: 2
   - correct_answer_number: 3
   - correct_answer_number: 4

**🔍 생성 후 필수 검증:**
모든 문제를 생성한 후, 다음 사항을 반드시 확인하세요:

1. 선택지 검증:
   - 모든 선택지가 "번호. 내용" 형식인가?
   - 선택지에 번호가 중복되지 않았는가?
   - 선택지 내용이 명확한가?

2. 정답 검증:
   - correct_answer_number가 null이 아닌가?
   - correct_answer와 correct_answer_number가 일치하는가?
   - 정답이 올바른 형식인가?

**⚠️ 최종 경고:**
위의 규칙을 위반하면 문제가 무효화됩니다. 반드시 지켜주세요.

정확히 {num_questions}개의 고품질 문제를 생성해주세요.
"""

    def get_category_prompt(self, category: str, sub_category: str = None) -> str:
        """카테고리별 특화 프롬프트 조회"""
        category = category.lower()

        # 카테고리 매핑
        if category in ["it", "computer", "software", "programming"]:
            base_prompt = self.category_prompts["it"]
        elif category in ["certification", "license", "exam"]:
            base_prompt = self.category_prompts["certification"]
        else:
            base_prompt = self.category_prompts["general"]

        # 서브카테고리 특화
        if sub_category:
            sub_category = sub_category.lower()
            if category in ["it", "computer", "software", "programming"]:
                if sub_category in ["database", "db", "sql"]:
                    base_prompt += "\n\n🎯 **데이터베이스 특화 지침**:\n- SQL 쿼리, 정규화, 인덱싱 등 실무 중심\n- 실제 데이터베이스 설계 사례 포함\n- 성능 최적화 관련 문제 포함\n\n📝 **해설 특화**:\n- SQL 문법의 핵심 포인트와 자주 실수하는 부분 설명\n- 실제 데이터베이스 성능 문제 해결 사례 포함\n- 정규화 단계별 장단점과 적용 시기 설명\n- 인덱스 설계 원리와 성능 영향 구체적 분석"
                elif sub_category in ["algorithm", "data_structure"]:
                    base_prompt += "\n\n🎯 **알고리즘 특화 지침**:\n- 시간복잡도, 공간복잡도 분석\n- 실제 코딩 문제 형태\n- 최적화 알고리즘 사례 포함\n\n📝 **해설 특화**:\n- 알고리즘의 핵심 아이디어와 동작 원리 단계별 설명\n- 시간/공간복잡도 분석 과정과 최적화 방법 설명\n- 실제 코딩에서 자주 실수하는 부분과 디버깅 팁\n- 비슷한 알고리즘과의 차이점과 선택 기준 설명"
                elif sub_category in ["network", "security"]:
                    base_prompt += "\n\n🎯 **네트워크/보안 특화 지침**:\n- 프로토콜, 암호화, 인증 방식\n- 실제 보안 취약점 사례\n- 네트워크 설계 문제 포함\n\n📝 **해설 특화**:\n- 프로토콜의 동작 원리와 실제 네트워크에서의 역할 설명\n- 보안 위협의 구체적 사례와 대응 방안 설명\n- 암호화 알고리즘의 원리와 적용 시나리오 분석\n- 네트워크 설계 시 고려해야 할 보안 요소들 설명"
            elif category in ["certification", "license", "exam"]:
                if sub_category in ["information_processing", "computer_utilization"]:
                    base_prompt += "\n\n🎯 **정보처리기사 특화 지침**:\n- 국가기술자격증 출제 경향 반영\n- 실무 중심의 문제 구성\n- 최신 기술 트렌드 반영\n\n📝 **해설 특화**:\n- 실제 시험에서 자주 나오는 함정과 오답 패턴 분석\n- 국가기술자격증의 출제 기준과 채점 포인트 설명\n- 실무에서 자격증 지식을 어떻게 활용하는지 구체적 사례\n- 최신 기술 트렌드와 자격증 커리큘럼의 연관성 설명"

        return base_prompt

    def _get_it_category_prompt(self) -> str:
        """IT 전공 카테고리 특화 프롬프트"""
        return """
🎯 **IT 전공 특화 지침**:

**문제 구성 원칙:**
1. **실무 중심**: 이론과 실무를 균형있게 다루되, 실무 적용 가능한 내용 우선
2. **최신 기술 반영**: 최신 기술 트렌드와 실무에서 사용하는 도구/언어 반영
3. **코딩 실습**: 실제 코드 예시나 알고리즘 문제 포함
4. **시스템 설계**: 아키텍처, 설계 패턴, 성능 최적화 문제 포함
5. **문제 해결**: 실제 개발 과정에서 마주치는 문제 상황 제시

**IT 전공별 특성:**
- **컴퓨터공학**: 하드웨어, 운영체제, 컴퓨터 구조
- **소프트웨어공학**: 개발 방법론, 프로젝트 관리, 품질 보증
- **정보보안**: 암호화, 네트워크 보안, 보안 정책
- **데이터사이언스**: 통계, 머신러닝, 빅데이터 처리
- **AI/ML**: 알고리즘, 모델링, 데이터 전처리

**출제 포인트:**
- 실제 개발 환경에서 사용하는 도구와 기술
- 최신 프레임워크와 라이브러리
- 성능 최적화와 확장성 고려
- 보안과 품질 관리
- 협업과 프로젝트 관리

**해설 특화 지침:**
- **개념 설명**: 기술적 용어를 쉽게 풀어서 설명
- **실무 연관**: 실제 개발에서 어떻게 활용되는지 구체적 사례 포함
- **오답 분석**: 비슷한 기술이나 개념과의 차이점 명확히 구분
- **학습 포인트**: 이 기술을 배우면 어떤 문제를 해결할 수 있는지 설명
- **최신 동향**: 최신 기술 트렌드와 연관지어 설명
"""

    def _get_certification_category_prompt(self) -> str:
        """자격증 카테고리 특화 프롬프트"""
        return """
🎯 **자격증 특화 지침**:

**문제 구성 원칙:**
1. **시험 경향 반영**: 실제 자격증 시험의 출제 패턴과 난이도 반영
2. **실무 중심**: 자격증 취득 후 실제 업무에서 활용 가능한 내용
3. **최신 동향**: 자격증 커리큘럼의 최신 변경사항 반영
4. **표준 준수**: 각 자격증의 표준과 가이드라인 준수
5. **실습 문제**: 실제 시험에서 나올 수 있는 실습 문제 포함

**자격증별 특성:**
- **정보처리기사**: 프로그래밍, 데이터베이스, 네트워크, 운영체제
- **컴퓨터활용능력**: 엑셀, 파워포인트, 워드, 데이터베이스
- **SQLD**: 데이터베이스 설계, SQL 활용, 성능 최적화
- **AWS/Azure**: 클라우드 서비스, 아키텍처, 보안
- **네트워크 관리사**: 네트워크 구성, 보안, 트러블슈팅

**출제 포인트:**
- 자격증별 핵심 개념과 용어
- 실제 시험에서 자주 출제되는 문제 유형
- 실무 적용 가능한 실습 문제
- 최신 기술 트렌드 반영
- 문제 해결 능력 평가

**해설 특화 지침:**
- **시험 포인트**: 실제 시험에서 자주 나오는 함정이나 오답 패턴 설명
- **실무 적용**: 자격증 취득 후 실제 업무에서 어떻게 활용되는지 구체적 사례
- **개념 정리**: 관련된 여러 개념을 체계적으로 정리하여 설명
- **학습 전략**: 이 문제를 통해 어떤 부분을 더 공부해야 하는지 안내
- **최신 동향**: 자격증 커리큘럼의 최신 변경사항 반영
"""

    def _get_general_category_prompt(self) -> str:
        """일반 카테고리 특화 프롬프트"""
        return """
🎯 **일반 교육 특화 지침**:

**문제 구성 원칙:**
1. **교육적 가치**: 학습 목표 달성에 도움이 되는 문제 구성
2. **이해도 중심**: 개념 이해와 적용 능력 평가
3. **다양성**: 다양한 관점과 접근 방법 포함
4. **실용성**: 실제 상황에서 활용 가능한 지식 중심
5. **흥미 유발**: 학습자의 관심을 끌 수 있는 문제 구성

**일반 과목별 특성:**
- **언어/문학**: 문법, 작문, 문학 작품 이해
- **수학**: 개념 이해, 문제 해결, 논리적 사고
- **과학**: 실험, 관찰, 과학적 사고
- **사회**: 역사, 지리, 사회 현상 이해
- **예술**: 창작, 감상, 문화 이해

**출제 포인트:**
- 기본 개념과 원리 이해
- 실제 상황 적용 능력
- 비판적 사고와 분석 능력
- 창의적 문제 해결 능력
- 다양한 관점에서의 접근

**해설 특화 지침:**
- **개념 이해**: 복잡한 개념을 쉽고 친근한 예시로 설명
- **실생활 연관**: 일상생활에서 어떻게 적용되는지 구체적 사례 포함
- **사고 과정**: 문제 해결을 위한 논리적 사고 과정 단계별 설명
- **확장 학습**: 이 개념과 관련된 추가 학습 포인트 안내
- **다양한 관점**: 여러 가지 해석이나 접근 방법 제시
"""

    def _get_combined_preprocessing_prompt(self) -> str:
        """통합 전처리 프롬프트"""
        return """
당신은 전문 교육 컨텐츠 분석가입니다. 주어진 문서를 분석하여 요약, 핵심 주제, 키워드를 한 번에 추출해주세요.

📋 **분석 대상 문서:**
{content}

🎯 **분석 조건:**
- 난이도: {difficulty}
- 문제 유형: {question_type}
- 목표 문제 수: {num_questions}개

**분석 지침:**
1. **요약**: 문서의 핵심 내용을 300자 이내로 명확하게 요약
2. **핵심 주제**: 교육적 가치가 높은 5개의 핵심 주제 추출
3. **키워드**: 문제 출제에 직접 활용 가능한 10개의 키워드 추출

**출력 형식 (정확히 지켜주세요):**

요약:
[문서의 핵심 내용을 300자 이내로 요약]

핵심 주제:
- [주제1]
- [주제2]
- [주제3]
- [주제4]
- [주제5]

키워드:
[키워드1], [키워드2], [키워드3], [키워드4], [키워드5], [키워드6], [키워드7], [키워드8], [키워드9], [키워드10]

**중요**: 위 형식을 정확히 지켜주세요. 각 섹션은 반드시 "요약:", "핵심 주제:", "키워드:"로 시작해야 합니다.
"""

    def parse_combined_response(self, response_text: str) -> Dict[str, str]:
        """통합 응답 파싱"""
        try:
            # 요약 추출
            summary_start = response_text.find("요약:") + 3
            summary_end = response_text.find("핵심 주제:")
            summary = response_text[summary_start:summary_end].strip()

            # 주제 추출
            topics_start = response_text.find("핵심 주제:") + 6
            topics_end = response_text.find("키워드:")
            topics_text = response_text[topics_start:topics_end].strip()
            topics = [line.strip()[2:] for line in topics_text.split('\n') if line.strip().startswith('-')]

            # 키워드 추출
            keywords_start = response_text.find("키워드:") + 4
            keywords_text = response_text[keywords_start:].strip()
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]

            return {
                "summary": summary,
                "topics": "\n".join([f"- {topic}" for topic in topics]),
                "keywords": ", ".join(keywords)
            }
        except Exception as e:
            # 파싱 실패 시 기본값 반환
            return {
                "summary": "문서 분석 중 오류가 발생했습니다.",
                "topics": "- 문서 분석",
                "keywords": "분석, 문서, 내용"
            }

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