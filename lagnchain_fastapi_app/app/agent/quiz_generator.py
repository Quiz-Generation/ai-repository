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
        """❓ 4단계: 교수급 고품질 문제 생성 (대학 시험 + 자격증 대응)"""
        try:
            logger.info("STEP4 교수급 문제 생성 시작")

            keywords = state["keywords"]
            topics = state["core_topics"]
            request = state["request"]
            domain_context = state["domain_context"]
            summary = state["summary"]

            # 🎯 시험 유형별 맞춤 가이드
            exam_style = self._get_exam_style_guidance(domain_context, request.difficulty)

            # 🎯 난이도별 교수 관점 전략
            if request.difficulty == DifficultyLevel.EASY:
                professor_approach = "기본 개념 확실히 이해했는지 확인하는 문제"
                cognitive_focus = "암기와 이해 검증"
            elif request.difficulty == DifficultyLevel.MEDIUM:
                professor_approach = "개념을 실제 상황에 적용할 수 있는지 평가하는 문제"
                cognitive_focus = "적용과 분석 능력 평가"
            else:  # HARD
                professor_approach = "여러 개념을 종합하여 창의적으로 사고할 수 있는지 측정하는 문제"
                cognitive_focus = "종합적 사고와 문제해결 능력 측정"

            # 🔥 뛰어난 대학교 교수 관점의 프롬프트
            question_prompt = f"""
당신은 명문대학교의 베테랑 교수이자 수년간 우수한 시험 문제를 출제해온 전문가입니다.
학생들이 진정으로 성장할 수 있는 고품질 문제를 만드는 것이 당신의 철학입니다.

📚 **강의 내용**:
{summary[:1000]}

🎯 **출제 조건**:
- 문제 수: {request.num_questions}개
- 난이도: {request.difficulty.value} ({professor_approach})
- 문제 유형: {request.question_type.value}

🏫 **시험 환경 가이드**:
{exam_style}

📋 **교수로서의 출제 철학**:

1. **학습 목표 명확성**: 각 문제는 명확한 학습 목표를 가져야 함
2. **공정성**: 강의에서 다룬 내용 기반, 함정 문제 지양
3. **변별력**: 잘 아는 학생과 그렇지 않은 학생을 명확히 구분
4. **교육적 가치**: 틀려도 배울 수 있는 의미 있는 문제
5. **실용성**: 졸업 후에도 도움이 되는 실질적 지식

🎨 **{request.question_type.value} 문제 출제 전략**:
{self._get_professor_question_strategy(request.question_type)}

⚖️ **객관식 선택지 설계 (해당 시)**:
- **정답**: 명확하고 완전한 정답
- **매력적 오답**: 부분적 이해 학생이 선택할 만한 답
- **흔한 실수**: 자주 혼동하는 개념이나 계산 실수
- **명백한 오답**: 확실히 틀렸지만 공부 안 한 학생이 찍을 만한 답

🔍 **품질 기준** ({cognitive_focus}):
- 문제가 명확하고 모호하지 않은가?
- 해당 난이도에 적합한 인지 부하인가?
- 실제 시험에서 출제할 만한 수준인가?
- 학생이 성장할 수 있는 교육적 가치가 있는가?

출력은 반드시 다음 JSON 형식으로만 해주세요:

```json
{{
  "questions": [
    {{
      "id": 1,
      "question": "명확하고 정확한 문제 내용",
      "type": "{request.question_type.value}",
      "difficulty": "{request.difficulty.value}",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "correct_answer": "정답",
      "explanation": "정답인 이유와 오답 분석을 포함한 교육적 해설",
      "learning_objective": "이 문제로 확인하고자 하는 학습 목표",
      "keywords": ["핵심키워드1", "핵심키워드2"]
    }}
  ]
}}
```

**교수로서 당부**: 학생들이 단순 암기가 아닌 진정한 이해를 바탕으로 답할 수 있는 문제를 만들어주세요. {request.num_questions}개의 문제를 정성껏 출제해주시기 바랍니다.
"""

            messages = [
                SystemMessage(content="당신은 명문대학교의 베테랑 교수입니다. 수년간 우수한 시험 문제를 출제하며 학생들의 성장을 도운 교육 전문가입니다."),
                HumanMessage(content=question_prompt)
            ]

            response = await self.llm.ainvoke(messages)

            # JSON 파싱 시도 (기존 로직 유지)
            try:
                import json

                # JSON 추출 (코드 블록에서)
                content = response.content
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                elif "```" in content:
                    # 일반 코드 블록
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                else:
                    # JSON 없이 바로 출력된 경우
                    json_content = content.strip()
                    if not json_content.startswith("{"):
                        # 텍스트에서 JSON 부분 찾기
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

                state["generated_questions"] = questions
                state["current_step"] = "question_generator"

                logger.info(f"SUCCESS 교수급 문제 생성 완료: {len(questions)}개")
                return state

            except json.JSONDecodeError as e:
                logger.error(f"ERROR JSON 파싱 실패: {e}")
                logger.error(f"LLM 응답 내용: {response.content[:500]}...")
                # 폴백: 텍스트로 파싱 시도
                state["generated_questions"] = [{"raw_content": response.content, "parsing_error": str(e)}]
                state["errors"].append(f"JSON 파싱 실패: {str(e)}")
                return state

        except Exception as e:
            logger.error(f"ERROR 문제 생성 실패: {e}")
            state["errors"].append(f"문제 생성 실패: {str(e)}")
            return state

    def _get_exam_style_guidance(self, domain_context: Dict[str, Any], difficulty: DifficultyLevel) -> str:
        """시험 유형별 맞춤 가이드"""
        guidance_parts = []

        # 도메인별 시험 스타일
        for filename, info in domain_context.items():
            filename_lower = filename.lower()

            if "aws" in filename_lower or "cloud" in filename_lower:
                guidance_parts.append("🔧 **IT 자격증 스타일**: 실무 시나리오 중심, 서비스 선택과 설정 문제")

            elif "dynamic" in filename_lower or "algorithm" in filename_lower:
                guidance_parts.append("💻 **전산학 전공 스타일**: 알고리즘 효율성, 복잡도 분석, 구현 원리 중심")

            elif "심리" in filename_lower:
                guidance_parts.append("🧠 **인문사회 전공 스타일**: 이론 이해, 개념 적용, 사례 분석 중심")

            else:
                guidance_parts.append("📚 **일반 대학 시험 스타일**: 개념 이해와 실제 적용의 균형")

        # 난이도별 추가 가이드
        if difficulty == DifficultyLevel.EASY:
            guidance_parts.append("📝 **초급 수준**: 기본 개념 확인, 명확한 정답, 학습 동기 부여")
        elif difficulty == DifficultyLevel.MEDIUM:
            guidance_parts.append("📈 **중급 수준**: 응용 문제, 상황 판단, 실무 연결성")
        else:
            guidance_parts.append("🎯 **고급 수준**: 종합 분석, 창의적 사고, 전문가 수준 이해")

        return "\n".join(guidance_parts)

    def _get_professor_question_strategy(self, question_type: QuestionType) -> str:
        """교수 관점의 문제 유형별 전략"""
        strategies = {
            QuestionType.MULTIPLE_CHOICE: """
**객관식 출제 전략**:
- 단순 암기보다는 이해도 측정 중심
- 선택지 간 명확한 구별 기준 제시
- "가장 적절한", "옳지 않은" 등 명확한 지시문
- 실제 상황 적용 능력 평가
""",
            QuestionType.TRUE_FALSE: """
**참/거짓 출제 전략**:
- 명확한 이론적 근거가 있는 진술만 사용
- "항상", "절대", "모든" 등 절대적 표현 신중 사용
- 핵심 개념의 정확한 이해 여부 확인
- 자주 혼동하는 개념들 구별 능력 측정
""",
            QuestionType.SHORT_ANSWER: """
**단답형 출제 전략**:
- 핵심 용어의 정확한 기억과 이해
- 계산 문제의 경우 중간 과정보다 최종 답안 중심
- 명확하고 간결한 답안이 나올 수 있는 문제
- 주관적 해석 여지가 적은 객관적 답안
""",
            QuestionType.ESSAY: """
**서술형 출제 전략**:
- 논리적 사고 과정을 평가할 수 있는 문제
- 여러 개념을 종합하여 설명하는 능력 측정
- 명확한 채점 기준이 있는 구조화된 답안 요구
- 창의적 사고와 비판적 분석 능력 평가
""",
            QuestionType.FILL_BLANK: """
**빈칸 채우기 출제 전략**:
- 문맥상 핵심이 되는 용어나 개념
- 논리적 흐름을 완성하는 중요한 단어
- 전문 용어의 정확한 사용법 확인
- 문장 전체의 의미를 이해해야 풀 수 있는 문제
"""
        }
        return strategies.get(question_type, "")

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