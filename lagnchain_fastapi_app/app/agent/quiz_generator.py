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

# 🔥 프롬프트 관리자 임포트
from .prompt import QuizPromptManager
from .prompt.quiz_prompt_manager import DifficultyLevel, QuestionType

logger = logging.getLogger(__name__)


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

        # 🎯 프롬프트 관리자 초기화
        self.prompt_manager = QuizPromptManager()

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
        """❓ 4단계: 균형 잡힌 문제 생성 (90% 일반 + 10% 응용)"""
        try:
            logger.info("STEP4 균형 잡힌 문제 생성 시작")

            request = state["request"]
            summary = state["summary"]

            # 🎯 프롬프트 관리자를 통한 프롬프트 생성
            final_prompt = self.prompt_manager.generate_final_prompt(
                summary=summary,
                num_questions=request.num_questions,
                difficulty=request.difficulty,
                question_type=request.question_type
            )

            messages = [
                SystemMessage(content=self.prompt_manager.get_system_message()),
                HumanMessage(content=final_prompt)
            ]

            response = await self.llm.ainvoke(messages)

            # JSON 파싱 (기존 로직 유지)
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

                # 🔄 수량 보장 검증 (기존 로직 간소화)
                validated_questions = self._ensure_question_count(questions, request)

                state["generated_questions"] = validated_questions
                state["current_step"] = "question_generator"

                # 📊 분포 확인 로깅
                basic_count = sum(1 for q in validated_questions if q.get("problem_level") == "basic")
                app_count = sum(1 for q in validated_questions if q.get("problem_level") == "application")

                logger.info(f"SUCCESS 균형 잡힌 문제 생성 완료: 총 {len(validated_questions)}개 (일반 {basic_count}개, 응용 {app_count}개)")
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

    def _ensure_question_count(self, questions: List[Dict], request: QuizRequest) -> List[Dict]:
        """수량 보장 시스템 (간소화)"""
        validated_questions = []

        for q in questions:
            if (isinstance(q, dict) and
                q.get("question") and
                q.get("correct_answer")):
                validated_questions.append(q)

            if len(validated_questions) >= request.num_questions:
                break

        # 🔄 수량 부족 시 기존 문제 복제로 보완
        while len(validated_questions) < request.num_questions and validated_questions:
            for q in questions:
                if len(validated_questions) >= request.num_questions:
                    break
                if isinstance(q, dict) and q.get("question"):
                    validated_questions.append(q)
                    logger.info(f"DUPLICATE 수량 부족으로 문제 복제 추가")

        return validated_questions[:request.num_questions]  # 정확한 수량만 반환

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