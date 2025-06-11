"""
🤖 Quiz Generation AI Agent using LangGraph
"""
import logging
import os
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """문제 난이도 레벨"""
    EASY = "easy"      # 기본 개념, 암기 위주
    MEDIUM = "medium"  # 이해 + 적용
    HARD = "hard"      # 응용 + 분석 + 종합


class QuestionType(Enum):
    """문제 타입"""
    MULTIPLE_CHOICE = "multiple_choice"  # 객관식 (4지선다)
    TRUE_FALSE = "true_false"           # OX 문제
    SHORT_ANSWER = "short_answer"       # 단답형
    ESSAY = "essay"                     # 서술형
    FILL_BLANK = "fill_blank"          # 빈칸 채우기


@dataclass
class QuizRequest:
    """문제 생성 요청"""
    file_ids: List[str]                 # 대상 파일 ID들
    num_questions: int = 5              # 생성할 문제 수
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    question_type: QuestionType = QuestionType.MULTIPLE_CHOICE
    custom_topic: Optional[str] = None  # 특정 주제 지정


class QuizState(TypedDict):
    """LangGraph 상태 관리"""
    # 입력
    request: QuizRequest
    documents: List[Dict[str, Any]]

    # 워크플로우 상태
    summary: str
    core_topics: List[str]
    keywords: List[str]
    generated_questions: List[Dict[str, Any]]

    # 메타데이터
    current_step: str
    errors: List[str]
    domain_context: Dict[str, Any]


class QuizGeneratorAgent:
    """문제 생성 AI 에이전트"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        초기화
        Args:
            openai_api_key: OpenAI API 키 (환경변수에서 자동 로드 가능)
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 직접 전달하세요.")

        # LLM 모델 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # 비용 효율적인 모델
            temperature=0.7,      # 창의성과 일관성의 균형
            api_key=self.openai_api_key
        )

        # LangGraph 워크플로우 구성
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 생성"""

        # 워크플로우 그래프 생성
        workflow = StateGraph(QuizState)

        # 노드 추가
        workflow.add_node("document_summarizer", self._summarize_documents)
        workflow.add_node("topic_extractor", self._extract_core_topics)
        workflow.add_node("keyword_extractor", self._extract_keywords)
        workflow.add_node("question_generator", self._generate_questions)
        workflow.add_node("quality_validator", self._validate_questions)

        # 워크플로우 순서 정의
        workflow.set_entry_point("document_summarizer")
        workflow.add_edge("document_summarizer", "topic_extractor")
        workflow.add_edge("topic_extractor", "keyword_extractor")
        workflow.add_edge("keyword_extractor", "question_generator")
        workflow.add_edge("question_generator", "quality_validator")
        workflow.add_edge("quality_validator", END)

        return workflow.compile()

    async def _summarize_documents(self, state: QuizState) -> QuizState:
        """📄 1단계: 문서 요약"""
        try:
            logger.info("STEP1 문서 요약 시작")

            # 문서 내용 결합
            combined_content = ""
            domain_info = {}

            for doc in state["documents"]:
                filename = doc.get("filename", "Unknown")
                content = doc.get("content", "")

                combined_content += f"\n\n=== {filename} ===\n{content[:2000]}"  # 첫 2000자만

                # 도메인 정보 수집
                domain_info[filename] = {
                    "language": doc.get("language", "unknown"),
                    "file_size": doc.get("file_size", 0),
                    "chunk_count": doc.get("total_chunks", 0)
                }

            # 🔥 다중 도메인 대응 요약 프롬프트
            summary_prompt = f"""
당신은 전문 교육 컨텐츠 분석가입니다. 주어진 문서들을 분석하여 종합적인 요약을 작성해주세요.

📋 **분석 대상 문서들:**
{combined_content}

🎯 **요약 지침:**
1. 각 문서의 핵심 내용을 파악하고 주요 개념을 추출하세요
2. 서로 다른 도메인(기술, 학문, 실무 등)의 문서라면 각각의 특성을 반영하세요
3. 교육/학습 목적에 적합한 핵심 지식을 중심으로 요약하세요
4. 문제 출제가 가능한 구체적인 사실, 개념, 절차를 포함하세요

**요약 길이:** 500-800자
**출력 형식:** 각 문서별로 구분하여 요약한 후 전체 종합 요약
"""

            # LLM 호출
            messages = [
                SystemMessage(content="당신은 전문 교육 컨텐츠 분석가입니다."),
                HumanMessage(content=summary_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            summary = response.content

            # 상태 업데이트
            state["summary"] = summary
            state["domain_context"] = domain_info
            state["current_step"] = "document_summarizer"

            logger.info("SUCCESS 문서 요약 완료")
            return state

        except Exception as e:
            logger.error(f"ERROR 문서 요약 실패: {e}")
            state["errors"].append(f"문서 요약 실패: {str(e)}")
            return state

    async def _extract_core_topics(self, state: QuizState) -> QuizState:
        """🎯 2단계: 핵심 주제 추출"""
        try:
            logger.info("STEP2 핵심 주제 추출 시작")

            summary = state["summary"]
            request = state["request"]

            # 🔥 주제 추출 프롬프트 (일반화된)
            topic_prompt = f"""
문서 요약을 바탕으로 핵심 주제들을 추출해주세요.

📋 **문서 요약:**
{summary}

🎯 **추출 조건:**
- 난이도: {request.difficulty.value}
- 목표 문제 수: {request.num_questions}개
- 문제 유형: {request.question_type.value}

**주제 추출 지침:**
1. 교육적 가치가 높은 핵심 개념들을 선별하세요
2. 선택된 난이도에 적합한 주제들을 우선순위로 하세요
3. 각 도메인별 특성을 고려하여 다양성을 확보하세요
4. 문제 출제가 가능한 구체적인 주제를 포함하세요

**출력 형식:**
- 주제1: [주제명] - [간단한 설명]
- 주제2: [주제명] - [간단한 설명]
...

**주제 개수:** 5-8개 (문제 수보다 많게)
"""

            messages = [
                SystemMessage(content="당신은 전문 교육과정 설계자입니다."),
                HumanMessage(content=topic_prompt)
            ]

            response = await self.llm.ainvoke(messages)

            # 주제 파싱 (간단한 파싱)
            topics_text = response.content
            topics = []
            for line in topics_text.split('\n'):
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    topic = line.strip().lstrip('- •').strip()
                    if topic:
                        topics.append(topic)

            state["core_topics"] = topics
            state["current_step"] = "topic_extractor"

            logger.info(f"SUCCESS 핵심 주제 추출 완료: {len(topics)}개")
            return state

        except Exception as e:
            logger.error(f"ERROR 핵심 주제 추출 실패: {e}")
            state["errors"].append(f"주제 추출 실패: {str(e)}")
            return state

    async def _extract_keywords(self, state: QuizState) -> QuizState:
        """🔑 3단계: 핵심 키워드 추출"""
        try:
            logger.info("STEP3 키워드 추출 시작")

            topics = state["core_topics"]
            request = state["request"]

            # 🔥 키워드 추출 프롬프트
            keyword_prompt = f"""
추출된 핵심 주제들을 바탕으로 문제 출제용 키워드들을 추출해주세요.

📋 **핵심 주제들:**
{chr(10).join(f"{i+1}. {topic}" for i, topic in enumerate(topics))}

🎯 **키워드 추출 조건:**
- 난이도: {request.difficulty.value}
- 문제 유형: {request.question_type.value}

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

**키워드 개수:** 15-25개
"""

            messages = [
                SystemMessage(content="당신은 전문 시험 출제 전문가입니다."),
                HumanMessage(content=keyword_prompt)
            ]

            response = await self.llm.ainvoke(messages)

            # 키워드 파싱
            keywords_text = response.content
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]

            state["keywords"] = keywords
            state["current_step"] = "keyword_extractor"

            logger.info(f"SUCCESS 키워드 추출 완료: {len(keywords)}개")
            return state

        except Exception as e:
            logger.error(f"ERROR 키워드 추출 실패: {e}")
            state["errors"].append(f"키워드 추출 실패: {str(e)}")
            return state

    async def _generate_questions(self, state: QuizState) -> QuizState:
        """❓ 4단계: 진짜 대학 수준 응용 문제 생성 (복합적 사고 + 실무 연결)"""
        try:
            logger.info("STEP4 고급 응용 문제 생성 시작")

            keywords = state["keywords"]
            topics = state["core_topics"]
            request = state["request"]
            domain_context = state["domain_context"]
            summary = state["summary"]

            # 🎯 난이도별 실제 대학 수준 정의
            if request.difficulty == DifficultyLevel.EASY:
                cognitive_approach = "개념 이해 + 기본 적용"
                complexity = "단일 개념, 명확한 답"
                scenario = "교과서 예제 수준"
            elif request.difficulty == DifficultyLevel.MEDIUM:
                cognitive_approach = "개념 연결 + 실무 적용"
                complexity = "2-3개 개념 조합, 상황 분석"
                scenario = "실제 프로젝트 상황"
            else:  # HARD
                cognitive_approach = "복합적 사고 + 창의적 문제해결"
                complexity = "다중 개념 융합, 트레이드오프 분석, 최적화"
                scenario = "실무 전문가 수준 의사결정"

            # 🔥 수량 보장을 위한 강화된 프롬프트
            question_prompt = f"""
당신은 국내 최고 대학의 컴퓨터과학과 교수이며, 삼성전자/네이버 등에서 10년간 실무 경험을 쌓은 전문가입니다.
학생들이 졸업 후 바로 현업에서 활용할 수 있는 실질적이고 응용력 있는 문제를 출제해야 합니다.

📚 **강의 핵심 내용**:
{summary[:1200]}

🎯 **출제 기준 (난이도: {request.difficulty.value})**:
**인지적 요구사항**: {cognitive_approach}
**문제 복잡도**: {complexity}
**시나리오 수준**: {scenario}

🔢 **중요: 반드시 정확히 {request.num_questions}개의 서로 다른 문제를 생성해주세요**

🏆 **진짜 대학 수준 문제 출제 전략**:

### EASY 난이도 (기본 + 적용):
- 개념을 간단한 실무 상황에 적용
- "이 개념을 사용하면 어떤 이점이 있는가?"
- 기본 원리의 실제 활용 예시 제시

### MEDIUM 난이도 (연결 + 분석):
- 여러 개념을 연결한 문제 해결
- "A 방식과 B 방식을 비교했을 때 어떤 상황에서 어떤 것이 더 적합한가?"
- 실제 시스템 설계 시 고려사항들

### HARD 난이도 (복합 + 최적화):
- 실무 전문가가 직면하는 복잡한 문제들
- "제약 조건 A, B, C를 모두 만족하면서 성능을 최적화하려면?"
- 다양한 솔루션의 트레이드오프 분석
- 창의적이고 혁신적인 접근법 요구

🎨 **응용 문제 설계 원칙**:

1. **실무 시나리오 중심**: 교과서가 아닌 실제 회사/프로젝트 상황
2. **복합적 사고**: 단일 개념이 아닌 여러 개념의 융합
3. **의사결정 요구**: "무엇을 선택하고 왜?"
4. **트레이드오프 분석**: 장단점 비교, 최적화 고려
5. **창의적 해결**: 정해진 답이 아닌 합리적 근거 기반 답안

🔄 **다양성 확보 전략**:
- 게임 개발, 금융 시스템, 의료 정보, 물류 최적화, SNS 플랫폼, 전자상거래, IoT 시스템 등 다양한 분야
- 각 문제마다 완전히 다른 실무 상황과 제약 조건
- 서로 다른 관점에서 핵심 개념 접근

💡 **{request.question_type.value} 특화 전략**:
{self._get_advanced_question_strategy(request.question_type, request.difficulty)}

출력은 반드시 다음 JSON 형식으로만 해주세요:

```json
{{
  "questions": [
    {{
      "id": 1,
      "question": "구체적인 실무 상황을 포함한 응용 문제",
      "type": "{request.question_type.value}",
      "difficulty": "{request.difficulty.value}",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "correct_answer": "정답",
      "explanation": "왜 이 답이 실무적으로 가장 타당한지에 대한 전문가 수준 해설",
      "learning_objective": "이 문제로 평가하고자 하는 실무 역량",
      "scenario_type": "적용된 실무 시나리오 유형",
      "keywords": ["핵심키워드1", "핵심키워드2"]
    }}
  ]
}}
```

**🎯 절대 준수사항**:
1. 정확히 {request.num_questions}개의 문제를 생성해야 합니다
2. 각 문제는 서로 다른 실무 시나리오를 사용해야 합니다
3. 모든 문제에 question, correct_answer, options를 반드시 포함해야 합니다

{request.num_questions}개의 고품질 응용 문제를 빠짐없이 생성해주세요.
"""

            messages = [
                SystemMessage(content="당신은 국내 최고 대학의 교수이자 실무 경험 10년의 전문가입니다. 요청된 수량의 문제를 정확히 생성하는 것이 핵심입니다."),
                HumanMessage(content=question_prompt)
            ]

            response = await self.llm.ainvoke(messages)

            # JSON 파싱 (기존 로직 유지하되 더 강화된 처리)
            try:
                import json

                content = response.content
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                else:
                    json_content = content.strip()
                    if not json_content.startswith("{"):
                        lines = json_content.split('\n')
                        json_lines = []
                        in_json = False
                        for line in lines:
                            if line.strip().startswith('{') or in_json:
                                in_json = True
                                json_lines.append(line)
                                if line.strip().endswith('}') and json_lines:
                                    break
                        json_content = '\n'.join(json_lines)

                questions_data = json.loads(json_content)
                questions = questions_data.get("questions", [])

                # 🔄 개선된 검증 및 필터링
                validated_questions = self._validate_and_filter_questions(questions, request)

                # 🚨 수량 부족 시 추가 처리
                if len(validated_questions) < request.num_questions * 0.9:  # 90% 미달 시
                    logger.warning(f"TARGET_SHORTAGE 목표 수량 부족 감지: {len(validated_questions)}/{request.num_questions}")

                    # 원본 questions에서 추가 복구 시도
                    for q in questions:
                        if len(validated_questions) >= request.num_questions:
                            break
                        if q not in validated_questions and isinstance(q, dict) and q.get("question"):
                            validated_questions.append(q)
                            logger.info(f"RECOVERY 추가 문제 복구: {q.get('id', 'unknown')}")

                state["generated_questions"] = validated_questions
                state["current_step"] = "question_generator"

                logger.info(f"SUCCESS 고급 응용 문제 생성 완료: {len(validated_questions)}개 (목표: {request.num_questions}개)")
                return state

            except json.JSONDecodeError as e:
                logger.error(f"ERROR JSON 파싱 실패: {e}")
                logger.error(f"LLM 응답 내용: {response.content[:500]}...")
                state["generated_questions"] = [{"raw_content": response.content, "parsing_error": str(e)}]
                state["errors"].append(f"JSON 파싱 실패: {str(e)}")
                return state

        except Exception as e:
            logger.error(f"ERROR 문제 생성 실패: {e}")
            state["errors"].append(f"문제 생성 실패: {str(e)}")
            return state

    def _get_advanced_question_strategy(self, question_type: QuestionType, difficulty: DifficultyLevel) -> str:
        """고급 응용 문제 전략"""
        base_strategies = {
            QuestionType.MULTIPLE_CHOICE: {
                DifficultyLevel.EASY: "실무 기본 상황에서의 개념 적용, 명확한 정답",
                DifficultyLevel.MEDIUM: "여러 선택지의 실무적 타당성 비교, 상황별 최적해",
                DifficultyLevel.HARD: "복잡한 제약조건 하에서의 최적 솔루션 선택, 트레이드오프 분석"
            },
            QuestionType.TRUE_FALSE: {
                DifficultyLevel.EASY: "실무에서 자주 접하는 개념의 참/거짓",
                DifficultyLevel.MEDIUM: "특정 상황에서의 원리 적용 가능성",
                DifficultyLevel.HARD: "복합적 상황에서의 이론적 원칙 적용 타당성"
            },
            QuestionType.SHORT_ANSWER: {
                DifficultyLevel.EASY: "핵심 개념의 실무적 정의",
                DifficultyLevel.MEDIUM: "문제 해결을 위한 핵심 접근법",
                DifficultyLevel.HARD: "최적화를 위한 창의적 솔루션"
            }
        }

        strategy = base_strategies.get(question_type, {}).get(difficulty, "고급 응용 문제")
        return f"**{question_type.value} {difficulty.value} 전략**: {strategy}"

    def _validate_and_filter_questions(self, questions: List[Dict], request: QuizRequest) -> List[Dict]:
        """문제 중복 검증 및 품질 필터링 (수량 보장 우선)"""
        validated_questions = []
        used_keywords = set()
        used_core_concepts = set()

        for i, q in enumerate(questions):
            if not isinstance(q, dict) or not q.get("question"):
                continue

            question_text = q.get("question", "").lower()
            keywords = q.get("keywords", [])
            scenario = q.get("scenario_type", "")

            # 🔥 완화된 중복 검증 (너무 엄격하지 않게)
            keyword_overlap = sum(1 for kw in keywords if kw.lower() in used_keywords)
            core_concept_used = any(concept in question_text for concept in used_core_concepts)

            # 🔄 완화된 품질 기준 (수량 보장 우선)
            basic_quality = (
                len(question_text) > 20 and  # 최소 길이만 체크
                q.get("correct_answer") and  # 정답 존재
                len(q.get("options", [])) >= 2  # 최소 선택지 존재
            )

            # 🎯 실무 연결성 체크 (선택적)
            has_practical_context = any(word in question_text for word in [
                "실무", "회사", "시스템", "프로젝트", "기업", "고객", "서비스",
                "비즈니스", "솔루션", "최적화", "효율", "성능", "관리"
            ])

            # 🚀 수량 우선 정책: 기본 품질만 만족하면 통과
            if basic_quality:
                # 중복도가 너무 높지 않으면 포함
                if keyword_overlap < 3 and not core_concept_used:
                    validated_questions.append(q)
                    used_keywords.update(kw.lower() for kw in keywords[:2])  # 처음 2개만 저장
                    if scenario:
                        used_core_concepts.add(scenario.lower()[:10])  # 핵심 개념만 저장
                elif len(validated_questions) < request.num_questions * 0.7:  # 70% 미달 시 완화
                    validated_questions.append(q)
                    logger.info(f"RELAXED 품질 기준 완화로 문제 {i+1} 포함")

            # ✅ 목표 수량 달성 시 조기 종료
            if len(validated_questions) >= request.num_questions:
                break

        # 🔄 수량 부족 시 최소 기준만으로 재시도
        if len(validated_questions) < request.num_questions * 0.8:  # 80% 미달 시
            logger.warning(f"WARNING 목표 수량 부족: {len(validated_questions)}/{request.num_questions}")

            # 최소 기준만으로 재검토
            for i, q in enumerate(questions):
                if len(validated_questions) >= request.num_questions:
                    break

                if (isinstance(q, dict) and
                    q.get("question") and
                    q.get("correct_answer") and
                    q not in validated_questions):
                    validated_questions.append(q)
                    logger.info(f"MINIMAL 최소 기준으로 문제 {i+1} 추가")

        return validated_questions

    async def _validate_questions(self, state: QuizState) -> QuizState:
        """✅ 5단계: 문제 품질 검증"""
        try:
            logger.info("STEP5 문제 검증 시작")

            questions = state["generated_questions"]
            request = state["request"]

            # 기본 검증
            validated_questions = []

            for i, q in enumerate(questions):
                if isinstance(q, dict) and "question" in q:
                    # 필수 필드 검증
                    if q.get("question") and q.get("correct_answer"):
                        validated_questions.append(q)
                    else:
                        logger.warning(f"WARNING 문제 {i+1} 필수 필드 누락")
                else:
                    logger.warning(f"WARNING 문제 {i+1} 형식 오류")

            # 최종 상태 업데이트
            state["generated_questions"] = validated_questions
            state["current_step"] = "quality_validator"

            logger.info(f"SUCCESS 문제 검증 완료: {len(validated_questions)}개 문제 확정")
            return state

        except Exception as e:
            logger.error(f"ERROR 문제 검증 실패: {e}")
            state["errors"].append(f"문제 검증 실패: {str(e)}")
            return state

    async def generate_quiz(self, request: QuizRequest, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        문제 생성 메인 메서드

        Args:
            request: 문제 생성 요청
            documents: 대상 문서들

        Returns:
            생성된 문제 데이터
        """
        try:
            logger.info("🚀 문제 생성 AI 에이전트 시작")

            # 초기 상태 설정
            initial_state: QuizState = {
                "request": request,
                "documents": documents,
                "summary": "",
                "core_topics": [],
                "keywords": [],
                "generated_questions": [],
                "current_step": "init",
                "errors": [],
                "domain_context": {}
            }

            # 워크플로우 실행
            final_state = await self.workflow.ainvoke(initial_state)

            # 결과 정리
            result = {
                "success": True,
                "request": {
                    "file_ids": request.file_ids,
                    "num_questions": request.num_questions,
                    "difficulty": request.difficulty.value,
                    "question_type": request.question_type.value
                },
                "process_info": {
                    "summary": final_state["summary"],
                    "core_topics": final_state["core_topics"],
                    "keywords": final_state["keywords"],
                    "domain_context": final_state["domain_context"]
                },
                "questions": final_state["generated_questions"],
                "meta": {
                    "generated_count": len(final_state["generated_questions"]),
                    "errors": final_state["errors"],
                    "final_step": final_state["current_step"]
                }
            }

            logger.info("🎉 SUCCESS 문제 생성 완료")
            return result

        except Exception as e:
            logger.error(f"ERROR 문제 생성 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "request": {
                    "file_ids": request.file_ids,
                    "num_questions": request.num_questions,
                    "difficulty": request.difficulty.value,
                    "question_type": request.question_type.value
                }
            }