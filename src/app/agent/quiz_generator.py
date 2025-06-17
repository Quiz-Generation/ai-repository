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
from langchain.prompts import ChatPromptTemplate

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
    additional_instructions: Optional[List[str]] = None  # 추가 지시사항


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

        # 프롬프트 템플릿 초기화
        self.summary_template = ChatPromptTemplate.from_messages([
            ("system", "당신은 전문 교육 컨텐츠 분석가입니다."),
            ("human", "{prompt}")
        ])

        self.topic_template = ChatPromptTemplate.from_messages([
            ("system", "당신은 전문 교육과정 설계자입니다."),
            ("human", "{prompt}")
        ])

        self.keyword_template = ChatPromptTemplate.from_messages([
            ("system", "당신은 전문 시험 출제 전문가입니다."),
            ("human", "{prompt}")
        ])

        self.question_template = ChatPromptTemplate.from_messages([
            ("system", "{system_message}"),
            ("human", "{prompt}")
        ])

    def _create_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(QuizState)

        # 병렬 처리 노드 추가
        workflow.add_node("parallel_processor", self._parallel_process)
        workflow.add_node("question_generator", self._generate_questions)
        workflow.add_node("quality_validator", self._validate_questions)

        # 워크플로우 순서 정의
        workflow.set_entry_point("parallel_processor")
        workflow.add_edge("parallel_processor", "question_generator")
        workflow.add_edge("question_generator", "quality_validator")
        workflow.add_edge("quality_validator", END)

        return workflow.compile()

    async def _parallel_process(self, state: QuizState) -> QuizState:
        """📄 병렬 처리: 문서 요약, 핵심 주제 추출, 키워드 추출"""
        try:
            logger.info("병렬 처리 시작")

            # 문서 내용 결합
            combined_content = ""
            domain_info = {}
            for doc in state["documents"]:
                filename = doc.get("filename", "Unknown")
                content = doc.get("content", "")
                combined_content += f"\n\n=== {filename} ===\n{content[:2000]}"
                domain_info[filename] = {
                    "language": doc.get("language", "unknown"),
                    "file_size": doc.get("file_size", 0),
                    "chunk_count": doc.get("total_chunks", 0)
                }

            # 병렬 처리 태스크 정의
            async def summarize_documents():
                summary_prompt = f"""
당신은 전문 교육 컨텐츠 분석가입니다. 주어진 문서들을 분석하여 종합적인 요약을 작성해주세요.

📋 **분석 대상 문서들:**
{combined_content}

🎯 **요약 지침:**
1. 각 문서의 핵심 내용을 파악하고 주요 개념을 추출하세요
2. 서로 다른 도메인의 문서라면 각각의 특성을 반영하세요
3. 교육/학습 목적에 적합한 핵심 지식을 중심으로 요약하세요
4. 문제 출제가 가능한 구체적인 사실, 개념, 절차를 포함하세요

**요약 길이:** 500-800자
**출력 형식:** 각 문서별로 구분하여 요약한 후 전체 종합 요약
"""
                chain = self.summary_template | self.llm
                response = await chain.ainvoke({"prompt": summary_prompt})
                return response.content

            async def extract_topics():
                topic_prompt = f"""
문서 요약을 바탕으로 핵심 주제들을 추출해주세요.

📋 **문서 요약:**
{combined_content}

🎯 **추출 조건:**
- 난이도: {state["request"].difficulty.value}
- 목표 문제 수: {state["request"].num_questions}개
- 문제 유형: {state["request"].question_type.value}

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
                chain = self.topic_template | self.llm
                response = await chain.ainvoke({"prompt": topic_prompt})
                topics_text = response.content
                topics = []
                for line in topics_text.split('\n'):
                    if line.strip().startswith('-') or line.strip().startswith('•'):
                        topic = line.strip().lstrip('- •').strip()
                        if topic:
                            topics.append(topic)
                return topics

            async def extract_keywords():
                keyword_prompt = f"""
추출된 핵심 주제들을 바탕으로 문제 출제용 키워드들을 추출해주세요.

📋 **핵심 주제들:**
{combined_content}

🎯 **키워드 추출 조건:**
- 난이도: {state["request"].difficulty.value}
- 문제 유형: {state["request"].question_type.value}

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
                chain = self.keyword_template | self.llm
                response = await chain.ainvoke({"prompt": keyword_prompt})
                keywords_text = response.content
                keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
                return keywords

            # 병렬 실행
            import asyncio
            summary, topics, keywords = await asyncio.gather(
                summarize_documents(),
                extract_topics(),
                extract_keywords()
            )

            # 상태 업데이트
            state["summary"] = summary
            state["core_topics"] = topics
            state["keywords"] = keywords
            state["domain_context"] = domain_info
            state["current_step"] = "parallel_processor"

            logger.info("SUCCESS 병렬 처리 완료")
            return state

        except Exception as e:
            logger.error(f"ERROR 병렬 처리 실패: {e}")
            state["errors"].append(f"병렬 처리 실패: {str(e)}")
            return state

    async def _generate_questions(self, state: QuizState) -> QuizState:
        """❓ 4단계: 균형 잡힌 문제 생성 (90% 일반 + 10% 응용)"""
        try:
            logger.info("STEP4 균형 잡힌 문제 생성 시작")

            request = state["request"]
            summary = state["summary"]
            topics = state["core_topics"]
            keywords = state["keywords"]

            # 추가 지시사항이 있는 경우 프롬프트에 추가
            additional_guide = ""
            if request.additional_instructions:
                additional_guide = "\n\n📝 **추가 지시사항**:\n" + "\n".join(f"- {instruction}" for instruction in request.additional_instructions)

            # 🎯 1단계: PDF 기반 문제 생성
            pdf_prompt = f"""
당신은 전문 교육 컨텐츠 개발자입니다. 주어진 내용을 바탕으로 고품질의 문제를 생성해주세요.

📚 **컨텐츠 요약**:
{summary}

🎯 **핵심 주제들**:
{chr(10).join(f"- {topic}" for topic in topics)}

🔑 **핵심 키워드들**:
{chr(10).join(f"- {keyword}" for keyword in keywords)}

📝 **문제 생성 조건**:
- 생성할 문제 수: {request.num_questions}개
- 난이도: {request.difficulty.value}
- 문제 유형: {request.question_type.value}

🎯 **문제 품질 요구사항**:
1. 각 문제는 구체적인 예시나 실제 사례를 포함해야 합니다
2. 중복되는 개념의 문제는 피하고, 다양한 관점에서 접근해야 합니다
3. 문제는 이론적 개념과 실제 구현을 균형있게 다루어야 합니다
4. 각 문제는 명확한 학습 목표를 가져야 합니다
5. 문제의 난이도는 지정된 수준에 맞게 조정되어야 합니다
6. 선택지는 명확하고 논리적으로 구성되어야 합니다
7. 정답 해설은 상세하고 교육적으로 가치있어야 합니다

{additional_guide}

**출력 형식**:
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
      "explanation": "정답 해설",
      "learning_objective": "학습 목표",
      "problem_level": "basic 또는 application",
      "keywords": ["키워드1", "키워드2"],
      "source": "pdf_based",
      "example": "관련 예시나 실제 사례",
      "implementation": "실제 구현 방법 (해당되는 경우)"
    }}
  ]
}}
```

정확히 {request.num_questions}개의 고품질 문제를 생성해주세요.
"""

            messages = [
                SystemMessage(content=self.prompt_manager.get_system_message()),
                HumanMessage(content=pdf_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            pdf_questions = self._parse_questions(response.content)

            # 📊 PDF 기반 문제 수 확인
            if len(pdf_questions) >= request.num_questions:
                state["generated_questions"] = pdf_questions[:request.num_questions]
                logger.info(f"SUCCESS PDF 기반 문제 생성 완료: {len(pdf_questions)}개")
                return state

            # 🎯 2단계: AI 기반 추가 문제 생성
            remaining_count = request.num_questions - len(pdf_questions)
            logger.info(f"PDF 기반 문제 부족: {remaining_count}개 추가 생성 필요")

            # AI 기반 문제 생성을 위한 프롬프트
            ai_prompt = f"""
당신은 전문 교육 컨텐츠 개발자입니다. 주어진 주제와 키워드를 바탕으로 추가 문제를 생성해주세요.

📚 **기존 컨텐츠 요약**:
{summary}

🎯 **핵심 주제들**:
{chr(10).join(f"- {topic}" for topic in topics)}

🔑 **핵심 키워드들**:
{chr(10).join(f"- {keyword}" for keyword in keywords)}

📝 **문제 생성 조건**:
- 추가 생성 필요 수량: {remaining_count}개
- 난이도: {request.difficulty.value}
- 문제 유형: {request.question_type.value}

🎯 **문제 품질 요구사항**:
1. 각 문제는 구체적인 예시나 실제 사례를 포함해야 합니다
2. 중복되는 개념의 문제는 피하고, 다양한 관점에서 접근해야 합니다
3. 문제는 이론적 개념과 실제 구현을 균형있게 다루어야 합니다
4. 각 문제는 명확한 학습 목표를 가져야 합니다
5. 문제의 난이도는 지정된 수준에 맞게 조정되어야 합니다
6. 선택지는 명확하고 논리적으로 구성되어야 합니다
7. 정답 해설은 상세하고 교육적으로 가치있어야 합니다

{additional_guide}

**출력 형식**:
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
      "explanation": "정답 해설",
      "learning_objective": "학습 목표",
      "problem_level": "basic 또는 application",
      "keywords": ["키워드1", "키워드2"],
      "source": "ai_generated",
      "example": "관련 예시나 실제 사례",
      "implementation": "실제 구현 방법 (해당되는 경우)"
    }}
  ]
}}
```

정확히 {remaining_count}개의 고품질 추가 문제를 생성해주세요.
"""

            messages = [
                SystemMessage(content="당신은 전문 교육 컨텐츠 개발자입니다."),
                HumanMessage(content=ai_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            ai_questions = self._parse_questions(response.content)

            # 📊 최종 문제 목록 생성
            final_questions = pdf_questions + ai_questions[:remaining_count]

            # 🔄 문제 순서 섞기
            import random
            random.shuffle(final_questions)

            state["generated_questions"] = final_questions
            state["current_step"] = "question_generator"

            # 📊 분포 확인 로깅
            basic_count = sum(1 for q in final_questions if q.get("problem_level") == "basic")
            app_count = sum(1 for q in final_questions if q.get("problem_level") == "application")
            pdf_count = sum(1 for q in final_questions if q.get("source") != "ai_generated")
            ai_count = sum(1 for q in final_questions if q.get("source") == "ai_generated")

            logger.info(f"SUCCESS 문제 생성 완료: 총 {len(final_questions)}개")
            logger.info(f"- PDF 기반: {pdf_count}개")
            logger.info(f"- AI 기반: {ai_count}개")
            logger.info(f"- 일반 문제: {basic_count}개")
            logger.info(f"- 응용 문제: {app_count}개")

            return state

        except Exception as e:
            logger.error(f"ERROR 문제 생성 실패: {e}")
            state["errors"].append(f"문제 생성 실패: {str(e)}")
            return state

    def _parse_questions(self, content: str) -> List[Dict]:
        """JSON 응답 파싱"""
        try:
            import json

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

            questions_data = json.loads(json_content)
            return questions_data.get("questions", [])

        except json.JSONDecodeError as e:
            logger.error(f"ERROR JSON 파싱 실패: {e}")
            logger.error(f"LLM 응답 내용: {content[:500]}...")
            return []

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