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
        """❓ 4단계: 체계적이고 논리적인 문제 생성 (실용적 개선)"""
        try:
            logger.info("STEP4 체계적 문제 생성 시작")

            keywords = state["keywords"]
            topics = state["core_topics"]
            request = state["request"]
            domain_context = state["domain_context"]
            summary = state["summary"]

            # 🎯 도메인 기반 문제 방향성 결정
            domain_guidance = self._get_domain_guidance(domain_context)

            # 🎯 난이도별 간단명료한 전략
            if request.difficulty == DifficultyLevel.EASY:
                approach = "핵심 개념 정의와 기본 특징 확인"
                style = "명확한 개념 문제"
            elif request.difficulty == DifficultyLevel.MEDIUM:
                approach = "개념 간 비교분석과 실제 적용"
                style = "분석적 사고 문제"
            else:  # HARD
                approach = "종합적 판단과 심화 이해"
                style = "복합적 사고 문제"

            # 🔥 간단하고 효과적인 프롬프트 (JSON 파싱 성공률 최적화)
            question_prompt = f"""
당신은 전문 출제위원입니다. 다음 학습 내용을 바탕으로 정확하고 체계적인 문제를 출제하세요.

**학습 내용**:
{summary[:800]}

**출제 조건**:
- 문제 수: {request.num_questions}개
- 난이도: {request.difficulty.value} ({approach})
- 문제 유형: {request.question_type.value}

**도메인 가이드**:
{domain_guidance}

**출제 원칙**:
1. 해당 분야에서 실제로 중요한 내용 중심
2. 도메인에 자연스럽게 부합하는 문제만 출제
3. 논리적이고 명확한 해설 제공
4. 억지스러운 타 분야 연결 금지

**객관식 선택지 구성**:
- 정답: 가장 정확한 답
- 오답1: 부분적으로 맞지만 핵심 누락
- 오답2: 흔한 오개념
- 오답3: 명백히 틀린 답

반드시 다음 JSON 형식으로만 출력하세요:

```json
{{
  "questions": [
    {{
      "id": 1,
      "question": "문제 내용",
      "type": "{request.question_type.value}",
      "difficulty": "{request.difficulty.value}",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "correct_answer": "정답",
      "explanation": "정답인 이유와 오답들의 문제점을 포함한 명확한 해설",
      "keywords": ["키워드1", "키워드2"]
    }}
  ]
}}
```

위 형식을 정확히 따라 {request.num_questions}개 문제를 생성하세요.
"""

            messages = [
                SystemMessage(content="당신은 전문 출제위원입니다. 학습자에게 도움이 되는 정확한 문제를 만드는 것이 목표입니다."),
                HumanMessage(content=question_prompt)
            ]

            response = await self.llm.ainvoke(messages)

            # JSON 파싱 시도
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

                logger.info(f"SUCCESS 체계적 문제 생성 완료: {len(questions)}개")
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

    def _get_domain_guidance(self, domain_context: Dict[str, Any]) -> str:
        """도메인별 간단한 출제 가이드"""
        guidance = []

        for filename, info in domain_context.items():
            filename_lower = filename.lower()

            if "aws" in filename_lower or "cloud" in filename_lower:
                guidance.append("클라우드/AWS: 인프라 설계, 서비스 선택, 비용 최적화 중심")

            elif "dynamic" in filename_lower or "algorithm" in filename_lower:
                guidance.append("알고리즘: 효율성 분석, 문제 해결 전략, 복잡도 이해 중심")

            elif "심리" in filename_lower or "psychology" in filename_lower:
                guidance.append("심리학: 이론 이해, 개념 구분, 실제 적용 사례 중심")

            else:
                guidance.append(f"해당 분야의 핵심 개념과 실제 활용에 집중")

        return " | ".join(guidance) if guidance else "해당 분야의 핵심 개념 중심"

    def _get_question_type_guide(self, question_type: QuestionType) -> str:
        """문제 유형별 특화 가이드"""
        guides = {
            QuestionType.MULTIPLE_CHOICE: """
🎯 **객관식 설계 전략**:
- 정답: 이론적으로 가장 정확하고 완전한 답
- 매력적 오답: 부분적으로 맞지만 핵심을 놓친 답
- 흔한 오개념: 학습자가 자주 혼동하는 개념
- 명백한 오답: 확실히 틀렸지만 논리적으로 배제 가능한 답
""",
            QuestionType.TRUE_FALSE: """
🎯 **참/거짓 설계 전략**:
- 명확한 이론적 근거가 있는 진술
- 절대적 표현 vs 조건부 표현 구별
- 자주 오해되는 개념의 정확성 테스트
""",
            QuestionType.SHORT_ANSWER: """
🎯 **단답형 설계 전략**:
- 핵심 용어의 정확한 명칭
- 수치나 공식의 정확한 표현
- 간결하고 명확한 답안
""",
            QuestionType.ESSAY: """
🎯 **서술형 설계 전략**:
- 논리적 구조의 답안 요구
- 다면적 분석과 종합 판단
- 근거와 결론의 명확한 연결
""",
            QuestionType.FILL_BLANK: """
🎯 **빈칸 채우기 설계 전략**:
- 문맥상 핵심이 되는 개념어
- 논리적 흐름을 완성하는 핵심 단어
- 전문 용어의 정확한 사용
"""
        }
        return guides.get(question_type, "")

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