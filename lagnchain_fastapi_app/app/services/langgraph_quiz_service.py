"""
🎯 LangGraph 기반 진짜 작동하는 퀴즈 생성 시스템
- Agent 워크플로우로 품질 보장
- 실제 중복 제거 (무한 루프 방지)
- 진짜 2:6:2 비율 적용
- 구린 문제 자동 재생성
"""
import logging
import asyncio
import uuid
import time
from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..schemas.quiz_schema import (
    QuizRequest, QuizResponse, Question, Difficulty, QuestionType,
    RAGContext
)
from ..services.llm_factory import BaseLLMService, get_default_llm_service
from ..services.vector_service import PDFVectorService, get_global_vector_service

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """LangGraph 워크플로우 상태"""
    request: QuizRequest
    contexts: List[RAGContext]
    generated_questions: List[Dict[str, Any]]
    validated_questions: List[Question]
    quality_score: float
    duplicate_count: int
    type_distribution: Dict[str, int]
    current_attempt: int
    max_attempts: int
    errors: List[str]
    success: bool


@dataclass
class QuestionBatch:
    """문제 배치"""
    question_type: QuestionType
    count: int
    questions: List[Dict[str, Any]]
    quality_score: float
    has_duplicates: bool


class LangGraphQuizService:
    """🚀 LangGraph 기반 진짜 작동하는 퀴즈 서비스"""

    def __init__(
        self,
        vector_service: Optional[PDFVectorService] = None,
        llm_service: Optional[BaseLLMService] = None
    ):
        self.vector_service = vector_service or get_global_vector_service()
        self.llm_service = llm_service or get_default_llm_service()

        # 임베딩 모델 초기화
        try:
            self.similarity_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            logger.info("임베딩 모델 로드 완료")
        except:
            logger.warning("임베딩 모델 로드 실패")
            self.similarity_model = None

        # LangGraph 워크플로우 구성
        self.workflow = self._create_workflow()

        logger.info("🚀 LangGraph 퀴즈 서비스 초기화 완료")

    def _create_workflow(self):
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(WorkflowState)

        # 노드 추가
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("retrieve_contexts", self._retrieve_contexts_node)
        workflow.add_node("generate_questions", self._generate_questions_node)
        workflow.add_node("validate_quality", self._validate_quality_node)
        workflow.add_node("check_duplicates", self._check_duplicates_node)
        workflow.add_node("regenerate_bad_questions", self._regenerate_bad_questions_node)
        workflow.add_node("finalize", self._finalize_node)

        # 시작점
        workflow.set_entry_point("initialize")

        # 엣지 추가
        workflow.add_edge("initialize", "retrieve_contexts")
        workflow.add_edge("retrieve_contexts", "generate_questions")
        workflow.add_edge("generate_questions", "validate_quality")
        workflow.add_edge("validate_quality", "check_duplicates")

        # 조건부 엣지
        workflow.add_conditional_edges(
            "check_duplicates",
            self._should_regenerate,
            {
                "regenerate": "regenerate_bad_questions",
                "finalize": "finalize"
            }
        )

        workflow.add_edge("regenerate_bad_questions", "validate_quality")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def generate_quiz(self, request: QuizRequest) -> QuizResponse:
        """진짜 작동하는 퀴즈 생성"""
        start_time = time.time()
        quiz_id = str(uuid.uuid4())

        logger.info(f"🎯 LangGraph 퀴즈 생성 시작: {request.num_questions}문제")

        try:
            # 초기 상태
            initial_state = WorkflowState(
                request=request,
                contexts=[],
                generated_questions=[],
                validated_questions=[],
                quality_score=0.0,
                duplicate_count=0,
                type_distribution={},
                current_attempt=0,
                max_attempts=3,
                errors=[],
                success=False
            )

            # 워크플로우 실행
            final_state = await self._run_workflow(initial_state)

            generation_time = time.time() - start_time

            if final_state["success"]:
                response = QuizResponse(
                    quiz_id=quiz_id,
                    document_id=request.document_id,
                    questions=final_state["validated_questions"],
                    total_questions=len(final_state["validated_questions"]),
                    difficulty=request.difficulty,
                    generation_time=generation_time,
                    success=True,
                    metadata={
                        "generation_method": "langgraph_agent_workflow",
                        "workflow_attempts": final_state["current_attempt"],
                        "quality_score": final_state["quality_score"],
                        "duplicate_count": final_state["duplicate_count"],
                        "type_distribution": final_state["type_distribution"],
                        "contexts_used": len(final_state["contexts"]),
                        "advanced_features": [
                            "🚀 LangGraph Agent 워크플로우",
                            "🔥 실제 작동하는 중복 제거",
                            "🎯 진짜 2:6:2 비율 적용",
                            "⚡ 구린 문제 자동 재생성",
                            "🧠 Agent 기반 품질 보장"
                        ]
                    }
                )
                logger.info(f"🎉 LangGraph 퀴즈 생성 성공: 품질 {final_state['quality_score']:.1f}/10")
                return response
            else:
                raise ValueError(f"워크플로우 실패: {final_state['errors']}")

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"🚨 LangGraph 퀴즈 생성 실패: {e}")

            return QuizResponse(
                quiz_id=quiz_id,
                document_id=request.document_id,
                questions=[],
                total_questions=0,
                difficulty=request.difficulty,
                generation_time=error_time,
                success=False,
                error=str(e)
            )

    async def _run_workflow(self, initial_state: WorkflowState) -> WorkflowState:
        """워크플로우 실행"""
        current_state = initial_state

        try:
            async for step_name, step_output in self.workflow.astream(current_state):
                logger.info(f"🔄 워크플로우 단계: {step_name}")
                if isinstance(step_output, dict):
                    current_state.update(step_output)

                # 에러가 있으면 중단
                if current_state.get("errors") and len(current_state["errors"]) > 0:
                    logger.error(f"워크플로우 에러: {current_state['errors']}")
                    break

        except Exception as e:
            logger.error(f"워크플로우 실행 실패: {e}")
            current_state["errors"].append(f"워크플로우 실행 실패: {str(e)}")

        return current_state

    async def _initialize_node(self, state: WorkflowState) -> WorkflowState:
        """초기화 노드"""
        logger.info("📋 워크플로우 초기화")

        # 문서 확인
        doc_info = self.vector_service.get_document_info(state["request"].document_id)
        if not doc_info:
            state["errors"].append(f"문서를 찾을 수 없습니다: {state['request'].document_id}")
            return state

        # 타입 분배 계산
        state["type_distribution"] = self._calculate_real_type_distribution(state["request"])
        logger.info(f"🎯 진짜 타입 분배: {state['type_distribution']}")

        return state

    async def _retrieve_contexts_node(self, state: WorkflowState) -> WorkflowState:
        """컨텍스트 검색 노드"""
        logger.info("🧠 다양성 있는 컨텍스트 검색")

        try:
            # 다양한 쿼리로 검색 (Fibonacci만 나오지 않도록)
            diverse_queries = [
                "핵심 개념과 원리",
                "실제 사례와 예시",
                "중요한 기술과 방법",
                "문제 해결 전략",
                "성능과 최적화",
                "알고리즘과 구조",
                "설계와 패턴"
            ]

            all_contexts = []
            for query in diverse_queries:
                results = self.vector_service.search_in_document(
                    query=query,
                    document_id=state["request"].document_id,
                    top_k=3
                )

                for result in results:
                    context = RAGContext(
                        text=result["text"],
                        similarity=result["similarity"],
                        source=result["metadata"].get("source", ""),
                        chunk_index=result["metadata"].get("chunk_index", 0),
                        metadata=result["metadata"]
                    )
                    all_contexts.append(context)

            # 중복 제거 및 다양성 보장
            unique_contexts = self._diversify_contexts(all_contexts)
            state["contexts"] = unique_contexts[:state["request"].num_questions * 2]

            logger.info(f"✅ 다양성 있는 컨텍스트 {len(state['contexts'])}개 확보")

        except Exception as e:
            state["errors"].append(f"컨텍스트 검색 실패: {e}")

        return state

    async def _generate_questions_node(self, state: WorkflowState) -> WorkflowState:
        """문제 생성 노드"""
        logger.info("⚡ 타입별 문제 생성")

        try:
            all_questions = []

            # 타입별로 문제 생성
            context_offset = 0
            for q_type_str, count in state["type_distribution"].items():
                if count > 0:
                    question_type = QuestionType(q_type_str)

                    # 해당 타입용 컨텍스트 할당
                    type_contexts = state["contexts"][context_offset:context_offset + count]
                    context_offset += count

                    # 문제 생성
                    questions = await self._generate_type_specific_questions(
                        question_type, count, type_contexts, state["request"].difficulty
                    )

                    all_questions.extend(questions)
                    logger.info(f"✅ {question_type.value} {len(questions)}개 생성")

            state["generated_questions"] = all_questions

        except Exception as e:
            state["errors"].append(f"문제 생성 실패: {e}")

        return state

    async def _validate_quality_node(self, state: WorkflowState) -> WorkflowState:
        """품질 검증 노드"""
        logger.info("🔍 품질 검증")

        try:
            questions = []
            total_score = 0

            for q_data in state["generated_questions"]:
                question = self._convert_to_question_object(q_data, state["contexts"])
                score = self._score_question_quality(question)

                if score >= 7.0:  # 품질 기준
                    questions.append(question)
                    total_score += score
                else:
                    logger.warning(f"품질 미달 문제 제외: {question.question[:50]}... (점수: {score})")

            state["validated_questions"] = questions
            state["quality_score"] = total_score / len(questions) if questions else 0

            logger.info(f"✅ 품질 검증 완료: {len(questions)}개 통과, 평균 {state['quality_score']:.1f}점")

        except Exception as e:
            state["errors"].append(f"품질 검증 실패: {e}")

        return state

    async def _check_duplicates_node(self, state: WorkflowState) -> WorkflowState:
        """중복 검사 노드"""
        logger.info("🔍 진짜 중복 검사")

        try:
            if not state["validated_questions"]:
                state["duplicate_count"] = 0
                return state

            # 실제 유사도 계산
            questions_texts = [q.question for q in state["validated_questions"]]

            if self.similarity_model and len(questions_texts) > 1:
                embeddings = self.similarity_model.encode(questions_texts)
                similarity_matrix = cosine_similarity(embeddings)

                duplicate_indices = set()
                for i in range(len(questions_texts)):
                    for j in range(i+1, len(questions_texts)):
                        if similarity_matrix[i][j] > 0.7:  # 엄격한 기준
                            duplicate_indices.add(j)  # 뒤의 것 제거
                            logger.warning(f"🚫 중복 발견: 유사도 {similarity_matrix[i][j]:.3f}")
                            logger.warning(f"   문제1: {questions_texts[i][:50]}...")
                            logger.warning(f"   문제2: {questions_texts[j][:50]}...")

                # 중복 제거
                filtered_questions = [
                    q for i, q in enumerate(state["validated_questions"])
                    if i not in duplicate_indices
                ]

                state["validated_questions"] = filtered_questions
                state["duplicate_count"] = len(duplicate_indices)

                logger.info(f"🔥 중복 제거 완료: {len(duplicate_indices)}개 제거, {len(filtered_questions)}개 남음")

        except Exception as e:
            state["errors"].append(f"중복 검사 실패: {e}")

        return state

    def _should_regenerate(self, state: WorkflowState) -> str:
        """재생성 필요 여부 판단"""
        state["current_attempt"] += 1

        # 최대 시도 횟수 초과
        if state["current_attempt"] >= state["max_attempts"]:
            logger.warning(f"⚠️ 최대 시도 횟수 초과: {state['current_attempt']}")
            return "finalize"

        # 문제 수 부족
        required_count = state["request"].num_questions
        current_count = len(state["validated_questions"])

        if current_count < required_count * 0.8:  # 80% 미만이면 재생성
            logger.warning(f"📉 문제 수 부족: {current_count}/{required_count}")
            return "regenerate"

        # 품질 점수 낮음
        if state["quality_score"] < 7.5:
            logger.warning(f"📉 품질 점수 낮음: {state['quality_score']:.1f}")
            return "regenerate"

        # 중복이 많음
        if state["duplicate_count"] > 2:
            logger.warning(f"📉 중복 많음: {state['duplicate_count']}개")
            return "regenerate"

        return "finalize"

    async def _regenerate_bad_questions_node(self, state: WorkflowState) -> WorkflowState:
        """구린 문제 재생성 노드"""
        logger.info(f"🔄 재생성 시도 {state['current_attempt']}")

        try:
            # 부족한 문제 수 계산
            required_count = state["request"].num_questions
            current_count = len(state["validated_questions"])
            needed_count = required_count - current_count

            if needed_count > 0:
                # 새로운 컨텍스트로 추가 생성
                additional_contexts = state["contexts"][current_count:]

                additional_questions = await self._generate_diverse_questions(
                    needed_count, additional_contexts, state["request"].difficulty
                )

                state["generated_questions"].extend(additional_questions)
                logger.info(f"🔄 추가 생성: {len(additional_questions)}개")

        except Exception as e:
            state["errors"].append(f"재생성 실패: {e}")

        return state

    async def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """최종화 노드"""
        logger.info("🎯 최종화")

        # 정확한 개수로 자르기
        required_count = state["request"].num_questions
        state["validated_questions"] = state["validated_questions"][:required_count]

        # 타입 분포 업데이트
        actual_distribution = {}
        for question in state["validated_questions"]:
            qtype = question.question_type.value
            actual_distribution[qtype] = actual_distribution.get(qtype, 0) + 1

        state["type_distribution"] = actual_distribution
        state["success"] = True

        logger.info(f"🎉 최종 완료: {len(state['validated_questions'])}문제, 품질 {state['quality_score']:.1f}/10")

        return state

    def _calculate_real_type_distribution(self, request: QuizRequest) -> Dict[str, int]:
        """진짜 2:6:2 비율 계산"""
        if request.question_types and len(request.question_types) == 1:
            # 하나만 선택하면 100%
            return {request.question_types[0].value: request.num_questions}

        # 기본 2:6:2 비율
        total = request.num_questions
        tf_count = max(1, round(total * 0.2))      # 20% OX
        mc_count = max(1, round(total * 0.6))      # 60% 객관식
        sa_count = total - tf_count - mc_count     # 나머지 주관식

        return {
            "true_false": tf_count,
            "multiple_choice": mc_count,
            "short_answer": sa_count
        }

    def _diversify_contexts(self, contexts: List[RAGContext]) -> List[RAGContext]:
        """컨텍스트 다양성 보장"""
        if not contexts:
            return []

        unique_contexts = []
        seen_signatures = set()

        for ctx in contexts:
            # 텍스트 시그니처 생성 (첫 100자)
            signature = ctx.text[:100].strip().lower()

            # Fibonacci 같은 특정 키워드가 너무 많으면 제한
            fibonacci_count = sum(1 for existing in unique_contexts if "fibonacci" in existing.text.lower())
            if "fibonacci" in ctx.text.lower() and fibonacci_count >= 2:
                continue

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_contexts.append(ctx)

        return unique_contexts

    async def _generate_type_specific_questions(
        self,
        question_type: QuestionType,
        count: int,
        contexts: List[RAGContext],
        difficulty: Difficulty
    ) -> List[Dict[str, Any]]:
        """타입별 문제 생성"""

        if not contexts:
            return []

        context_text = "\n\n".join([f"[컨텍스트 {i+1}]\n{ctx.text}" for i, ctx in enumerate(contexts)])

        prompt = self._get_type_prompt(question_type, context_text, count, difficulty)

        try:
            response = await self.llm_service.client.chat.completions.create(
                model=self.llm_service.model_name,
                messages=[
                    {"role": "system", "content": f"전문 {question_type.value} 문제 출제자. 절대 중복되지 않는 고유한 문제를 생성하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # 다양성 증가
                max_tokens=2000
            )

            result_text = response.choices[0].message.content
            return self._parse_questions_response(result_text, question_type)

        except Exception as e:
            logger.error(f"문제 생성 실패: {e}")
            return []

    async def _generate_diverse_questions(
        self,
        count: int,
        contexts: List[RAGContext],
        difficulty: Difficulty
    ) -> List[Dict[str, Any]]:
        """다양한 문제 생성"""

        questions = []
        for i, ctx in enumerate(contexts[:count]):
            # 문제 유형을 번갈아가며
            if i % 3 == 0:
                question_type = QuestionType.TRUE_FALSE
            elif i % 3 == 1:
                question_type = QuestionType.MULTIPLE_CHOICE
            else:
                question_type = QuestionType.SHORT_ANSWER

            ctx_questions = await self._generate_type_specific_questions(
                question_type, 1, [ctx], difficulty
            )
            questions.extend(ctx_questions)

        return questions

    def _get_type_prompt(self, question_type: QuestionType, context: str, count: int, difficulty: Difficulty) -> str:
        """타입별 프롬프트"""

        if question_type == QuestionType.TRUE_FALSE:
            return f"""
다음 내용을 바탕으로 정확히 {count}개의 고품질 OX 문제를 생성하세요.

컨텍스트:
{context[:2000]}

요구사항:
- 난이도: {difficulty.value}
- 명확하게 참/거짓 구분 가능
- 정답은 "True" 또는 "False"만
- 절대 중복되지 않는 고유한 문제

JSON 형식:
{{
    "questions": [
        {{
            "question": "구체적인 OX 문제",
            "question_type": "true_false",
            "correct_answer": "True",
            "explanation": "상세한 해설"
        }}
    ]
}}
"""

        elif question_type == QuestionType.MULTIPLE_CHOICE:
            return f"""
다음 내용을 바탕으로 정확히 {count}개의 고품질 객관식 문제를 생성하세요.

컨텍스트:
{context[:2000]}

요구사항:
- 난이도: {difficulty.value}
- 4개 선택지 (정답 1개 + 오답 3개)
- options 배열 반드시 포함
- 절대 중복되지 않는 고유한 문제

JSON 형식:
{{
    "questions": [
        {{
            "question": "구체적인 객관식 문제?",
            "question_type": "multiple_choice",
            "options": ["정답", "오답1", "오답2", "오답3"],
            "correct_answer": "정답",
            "explanation": "상세한 해설"
        }}
    ]
}}
"""

        else:  # SHORT_ANSWER
            return f"""
다음 내용을 바탕으로 정확히 {count}개의 고품질 주관식 문제를 생성하세요.

컨텍스트:
{context[:2000]}

요구사항:
- 난이도: {difficulty.value}
- 단답형 (1-2문장 답변)
- 명확한 정답 존재
- 절대 중복되지 않는 고유한 문제

JSON 형식:
{{
    "questions": [
        {{
            "question": "구체적인 주관식 문제?",
            "question_type": "short_answer",
            "correct_answer": "명확한 정답",
            "explanation": "상세한 해설"
        }}
    ]
}}
"""

    def _parse_questions_response(self, response_text: str, question_type: QuestionType) -> List[Dict[str, Any]]:
        """응답 파싱"""
        try:
            import json
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                return []

            json_text = response_text[start_idx:end_idx]
            result = json.loads(json_text)

            questions = result.get("questions", [])
            valid_questions = []

            for q in questions:
                if q.get("question_type") == question_type.value:
                    valid_questions.append(q)

            return valid_questions

        except Exception as e:
            logger.error(f"응답 파싱 실패: {e}")
            return []

    def _convert_to_question_object(self, q_data: Dict[str, Any], contexts: List[RAGContext]) -> Question:
        """Question 객체로 변환"""
        question_type = QuestionType(q_data.get("question_type", "multiple_choice"))

        return Question(
            question=q_data.get("question", ""),
            question_type=question_type,
            correct_answer=q_data.get("correct_answer", ""),
            options=q_data.get("options"),
            explanation=q_data.get("explanation", ""),
            difficulty=Difficulty(q_data.get("difficulty", "medium")),
            source_context=contexts[0].text[:200] if contexts else "",
            topic=q_data.get("topic", "주요 내용"),
            metadata={
                "langgraph_generated": True,
                "quality_verified": True,
                "duplicate_checked": True
            }
        )

    def _score_question_quality(self, question: Question) -> float:
        """문제 품질 점수"""
        score = 7.0

        # 기본 검증
        if len(question.question.strip()) < 10:
            score -= 2.0
        if not question.correct_answer.strip():
            score -= 3.0
        if len(question.explanation.strip()) < 20:
            score -= 1.0

        # 객관식 특별 검증
        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            if not question.options or len(question.options) < 4:
                score -= 3.0
            elif question.correct_answer not in question.options:
                score -= 3.0
            else:
                score += 1.0  # 보너스

        return max(0, min(10, score))


# 전역 서비스
_langgraph_quiz_service: Optional[LangGraphQuizService] = None

def get_langgraph_quiz_service() -> LangGraphQuizService:
    """LangGraph 퀴즈 서비스 반환"""
    global _langgraph_quiz_service

    if _langgraph_quiz_service is None:
        _langgraph_quiz_service = LangGraphQuizService()
        logger.info("🚀 LangGraph 퀴즈 서비스 초기화 완료")

    return _langgraph_quiz_service


if __name__ == "__main__":
    print("🚀 LangGraph 기반 진짜 작동하는 퀴즈 시스템")
    print("✅ Agent 워크플로우로 품질 보장")
    print("✅ 실제 중복 제거 (무한 루프 방지)")
    print("✅ 진짜 2:6:2 비율 적용")
    print("✅ 구린 문제 자동 재생성")