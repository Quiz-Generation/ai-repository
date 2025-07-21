"""
🤖 Quiz Generation AI Agent using LangGraph
"""
import logging
import os
import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum
import json
import time

# tokenizers 병렬 처리 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    category: Optional[str] = None      # 대분류(예: 컴퓨터 공학)
    sub_category: Optional[str] = None  # 소분류(예: 데이터베이스)
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
            api_key=self.openai_api_key if self.openai_api_key else None
        )

        # LangGraph 워크플로우 구성
        self.workflow = self._create_workflow()

        # 🎯 프롬프트 관리자 초기화
        self.prompt_manager = QuizPromptManager()

        # 🔥 캐시 초기화
        self._question_cache = {}

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

        self.validation_template = ChatPromptTemplate.from_messages([
            ("system", "당신은 전문 교육 컨텐츠 품질 검증 전문가입니다."),
            ("human", "{prompt}")
        ])

        # 체인 초기화
        self.summary_chain = self.summary_template | self.llm
        self.topic_chain = self.topic_template | self.llm
        self.keyword_chain = self.keyword_template | self.llm
        self.question_chain = self.question_template | self.llm
        self.validation_chain = self.validation_template | self.llm

    def _create_workflow(self):
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(QuizState)

        # 병렬 처리 노드 추가
        workflow.add_node("parallel_processor", self._parallel_process)
        workflow.add_node("question_generator", self._generate_questions)

        # 워크플로우 순서 정의
        workflow.set_entry_point("parallel_processor")
        workflow.add_edge("parallel_processor", "question_generator")
        workflow.add_edge("question_generator", END)

        return workflow.compile()

    async def _parallel_process(self, state: QuizState) -> QuizState:
        """📄 병렬 처리: 문서 요약, 핵심 주제 추출, 키워드 추출"""
        try:
            parallel_start = time.time()
            logger.info("병렬 처리 시작")

            # 문서 내용 결합 및 전처리
            combined_content = ""
            domain_info = {}
            total_sentences = 0
            total_paragraphs = 0

            for doc in state["documents"]:
                filename = doc.get("filename", "Unknown")
                content = doc.get("content", "")

                # 문장과 단락 수 계산
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                total_sentences += len(sentences)
                total_paragraphs += len(paragraphs)

                combined_content += f"\n\n=== {filename} ===\n{content}"
                domain_info[filename] = {
                    "language": doc.get("language", "unknown"),
                    "file_size": doc.get("file_size", 0),
                    "chunk_count": doc.get("total_chunks", 0),
                    "sentence_count": len(sentences),
                    "paragraph_count": len(paragraphs)
                }

            # 문제 수 재계산
            base_questions = max(3, total_sentences // 4)  # 4문장당 1문제
            complexity_factor = min(1.5, 1 + (total_paragraphs / total_sentences))
            concept_factor = min(0.5, len(domain_info) * 0.1)

            recommended_questions = int(base_questions * complexity_factor * (1 + concept_factor))
            recommended_questions = min(max(5, recommended_questions), 20)  # 5-20개 사이로 제한

            # 문제 수 업데이트
            state["request"].num_questions = recommended_questions

            # 병렬 처리 태스크 정의
            async def summarize_documents():
                summary_prompt = self.prompt_manager.get_prompt("summary").format(
                    content=combined_content
                )
                return await self.summary_chain.ainvoke({"prompt": summary_prompt})

            async def extract_topics():
                topic_prompt = self.prompt_manager.get_prompt("topic").format(
                    content=combined_content,
                    difficulty=state["request"].difficulty.value,
                    num_questions=recommended_questions,
                    question_type=state["request"].question_type.value,
                    num_topics=recommended_questions + 3
                )
                return await self.topic_chain.ainvoke({"prompt": topic_prompt})

            async def extract_keywords():
                keyword_prompt = self.prompt_manager.get_prompt("keyword").format(
                    content=combined_content,
                    difficulty=state["request"].difficulty.value,
                    question_type=state["request"].question_type.value,
                    num_keywords=recommended_questions * 3
                )
                return await self.keyword_chain.ainvoke({"prompt": keyword_prompt})

            # 병렬 실행
            topics_response, summary_response, keywords_response = await asyncio.gather(
                extract_topics(), summarize_documents(), extract_keywords()
            )
            logger.info(f"[전처리] 병렬 전체 소요 시간: {time.time() - parallel_start:.2f}초")
            summary = summary_response.content
            topics = [line.strip().lstrip('- •').strip() for line in topics_response.content.split('\n') if line.strip().startswith(('-', '•'))]
            keywords = [kw.strip() for kw in keywords_response.content.split(',') if kw.strip()]
            logger.info(f"[전처리] 완료 (총 소요 시간: {time.time() - parallel_start:.2f}초)")

            # 상태 업데이트
            state["summary"] = summary
            state["core_topics"] = topics
            state["keywords"] = keywords
            state["domain_context"] = {
                **domain_info,
                "total_sentences": total_sentences,
                "total_paragraphs": total_paragraphs,
                "recommended_questions": recommended_questions
            }
            state["current_step"] = "parallel_processor"

            logger.info(f"SUCCESS 병렬 처리 완료: {recommended_questions}개 문제 추천")
            return state

        except Exception as e:
            logger.error(f"ERROR 병렬 처리 실패: {e}")
            state["errors"].append(f"병렬 처리 실패: {str(e)}")
            return state

    async def _generate_questions(self, state: QuizState) -> QuizState:
        """❓ 4단계: 다양성과 품질을 고려한 최적화된 병렬 배치 문제 생성"""
        try:
            generate_start = time.time()
            logger.info("STEP4 다양성과 품질을 고려한 최적화된 병렬 배치 문제 생성 시작")

            request = state["request"]
            summary = state["summary"]
            topics = state["core_topics"]
            keywords = state["keywords"]

            # 추가 지시사항이 있는 경우 프롬프트에 추가
            additional_guide = ""
            if request.additional_instructions:
                additional_guide = "\n\n📝 **추가 지시사항**:\n" + "\n".join(f"- {instruction}" for instruction in request.additional_instructions)

            # 🎯 최적화된 배치 크기 계산 (더 작은 배치로 다양성 확보)
            target_questions = request.num_questions
            batch_size = min(2, max(1, target_questions // 3))  # 1-2개씩 배치로 다양성 확보
            num_batches = (target_questions + batch_size - 1) // batch_size

            logger.info(f"배치 처리 설정: {num_batches}개 배치, 배치당 {batch_size}개 문제")

            # 🎯 키워드 분산 전략
            keyword_groups = self._distribute_keywords(keywords, num_batches)
            topic_groups = self._distribute_topics(topics, num_batches)

            # 🚀 병렬 배치 생성 함수
            async def generate_batch(batch_num: int, batch_size: int, is_final_batch: bool = False) -> List[Dict]:
                """단일 배치 문제 생성 (다양성 고려)"""
                try:
                    # 마지막 배치는 남은 문제 수만큼만 생성
                    actual_batch_size = batch_size
                    if is_final_batch:
                        remaining = target_questions - (batch_num * batch_size)
                        actual_batch_size = max(1, remaining)

                    # 배치별 키워드와 주제 할당
                    batch_keywords = keyword_groups[batch_num % len(keyword_groups)]
                    batch_topics = topic_groups[batch_num % len(topic_groups)]

                    # 배치별 다양한 접근 방식으로 문제 생성
                    batch_prompts = []

                    # 난이도별 문제 생성 전략
                    if batch_num < num_batches * 0.4:  # 40% 기본 개념
                        batch_prompts.append({
                            "type": "basic_concept",
                            "prompt": self._create_diversity_prompt(
                                summary, batch_topics, batch_keywords,
                                actual_batch_size, request, "basic"
                            ),
                            "system": "당신은 기본 개념 문제 전문가입니다. 핵심 개념을 명확하게 묻는 문제를 생성하세요. 중복을 피하고 다양한 관점에서 접근하세요."
                        })
                    elif batch_num < num_batches * 0.7:  # 30% 개념 연계
                        batch_prompts.append({
                            "type": "concept_integration",
                            "prompt": self._create_diversity_prompt(
                                summary, batch_topics, batch_keywords,
                                actual_batch_size, request, "concept"
                            ),
                            "system": "당신은 개념 연계 문제 전문가입니다. 여러 개념을 연결하는 문제를 생성하세요. 다양한 예시와 응용을 포함하세요."
                        })
                    else:  # 30% 응용 문제
                        batch_prompts.append({
                            "type": "application",
                            "prompt": self._create_diversity_prompt(
                                summary, batch_topics, batch_keywords,
                                actual_batch_size, request, "application"
                            ),
                            "system": "당신은 응용 문제 전문가입니다. 실제 상황에 적용하는 문제를 생성하세요. 구체적인 사례와 분석을 포함하세요."
                        })

                    # 배치 내에서도 병렬 처리 (여러 접근 방식)
                    batch_tasks = []
                    for prompt_info in batch_prompts:
                        task = self.question_chain.ainvoke({
                            "system_message": prompt_info["system"],
                            "prompt": prompt_info["prompt"]
                        })
                        batch_tasks.append(task)

                    # 배치 병렬 실행
                    batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)

                    # 응답 처리 및 파싱
                    batch_questions = []
                    for i, response in enumerate(batch_responses):
                        if isinstance(response, Exception):
                            logger.warning(f"배치 {batch_num} 응답 {i} 실패: {response}")
                            continue

                        try:
                            if hasattr(response, 'content') and isinstance(response.content, str):
                                questions = self._parse_questions(response.content)
                                # 배치별 메타데이터 추가
                                for q in questions:
                                    q["batch_num"] = batch_num
                                    q["difficulty_level"] = batch_prompts[i]["type"]
                                batch_questions.extend(questions)
                        except Exception as e:
                            logger.warning(f"배치 {batch_num} 파싱 실패: {e}")
                            continue

                    logger.info(f"배치 {batch_num} 완료: {len(batch_questions)}개 문제 생성")
                    return batch_questions

                except Exception as e:
                    logger.error(f"배치 {batch_num} 생성 실패: {e}")
                    return []

            # 🚀 모든 배치를 병렬로 실행
            batch_tasks = []
            for i in range(num_batches):
                is_final = (i == num_batches - 1)
                task = generate_batch(i, batch_size, is_final)
                batch_tasks.append(task)

            # 병렬 실행
            all_batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # 📊 모든 배치 결과 통합
            all_questions = []
            for i, batch_result in enumerate(all_batch_results):
                if isinstance(batch_result, Exception):
                    logger.error(f"배치 {i} 전체 실패: {batch_result}")
                    continue
                if isinstance(batch_result, list):
                    all_questions.extend(batch_result)

            logger.info(f"모든 배치 완료: 총 {len(all_questions)}개 문제 생성")

            # 🔄 고급 중복 제거 및 품질 검사 (더 엄격한 기준)
            final_questions = self._advanced_quality_check_with_diversity(all_questions, target_questions)

            # 문제 수가 부족한 경우 빠른 보충 생성 (다양성 고려)
            if len(final_questions) < target_questions:
                logger.info(f"문제 수 부족 ({len(final_questions)}/{target_questions}), 다양성 고려한 보충 생성")

                # 사용되지 않은 키워드와 주제로 보충 생성
                used_keywords = set()
                used_topics = set()
                for q in final_questions:
                    question_text = q.get("question", "").lower()
                    for keyword in keywords:
                        if keyword.lower() in question_text:
                            used_keywords.add(keyword)
                    for topic in topics:
                        if topic.lower() in question_text:
                            used_topics.add(topic)

                unused_keywords = [k for k in keywords if k not in used_keywords]
                unused_topics = [t for t in topics if t not in used_topics]

                supplement_prompt = self._create_diversity_prompt(
                    summary, unused_topics[:3], unused_keywords[:5],
                    target_questions - len(final_questions), request, "mixed"
                )

                try:
                    supplement_response = await self.question_chain.ainvoke({
                        "system_message": "다양성과 품질을 중시하는 문제 생성 전문가입니다. 중복을 피하고 새로운 관점에서 문제를 생성하세요.",
                        "prompt": supplement_prompt
                    })

                    supplement_questions = self._parse_questions(supplement_response.content)
                    supplement_questions = self._basic_quality_check(supplement_questions)

                    final_questions.extend(supplement_questions)
                    logger.info(f"보충 생성 완료: {len(supplement_questions)}개 추가")

                except Exception as e:
                    logger.warning(f"보충 생성 실패: {e}")

            # 최종 중복 제거 및 품질 검사
            final_questions = self._advanced_quality_check_with_diversity(final_questions, target_questions)

            # 정확히 요청된 수만큼만 반환
            final_questions = final_questions[:target_questions]

            # ID 순차적으로 부여
            for i, question in enumerate(final_questions, 1):
                question["id"] = i

            state["generated_questions"] = final_questions
            state["current_step"] = "question_generator"

            # 📊 분포 확인 로깅
            basic_count = sum(1 for q in final_questions if q.get("difficulty_level") == "basic_concept")
            concept_count = sum(1 for q in final_questions if q.get("difficulty_level") == "concept_integration")
            app_count = sum(1 for q in final_questions if q.get("difficulty_level") == "application")

            logger.info(f"SUCCESS 다양성과 품질을 고려한 문제 생성 완료: 총 {len(final_questions)}개")
            logger.info(f"- 기본 개념: {basic_count}개")
            logger.info(f"- 개념 연계: {concept_count}개")
            logger.info(f"- 응용 문제: {app_count}개")
            logger.info(f"[실행시간] 최적화된 문제 생성 소요 시간: {time.time() - generate_start:.2f}초")

            return state

        except Exception as e:
            logger.error(f"ERROR 최적화된 문제 생성 실패: {e}")
            state["errors"].append(f"문제 생성 실패: {str(e)}")
            return state

    def _distribute_keywords(self, keywords: List[str], num_batches: int) -> List[List[str]]:
        """키워드를 배치별로 분산 배치"""
        if not keywords:
            return [[] for _ in range(num_batches)]

        # 키워드를 그룹별로 분산
        keyword_groups = []
        for i in range(num_batches):
            start_idx = (i * len(keywords)) // num_batches
            end_idx = ((i + 1) * len(keywords)) // num_batches
            group = keywords[start_idx:end_idx]
            if not group and keywords:  # 빈 그룹인 경우 전체 키워드 사용
                group = keywords
            keyword_groups.append(group)

        return keyword_groups

    def _distribute_topics(self, topics: List[str], num_batches: int) -> List[List[str]]:
        """주제를 배치별로 분산 배치"""
        if not topics:
            return [[] for _ in range(num_batches)]

        # 주제를 그룹별로 분산
        topic_groups = []
        for i in range(num_batches):
            start_idx = (i * len(topics)) // num_batches
            end_idx = ((i + 1) * len(topics)) // num_batches
            group = topics[start_idx:end_idx]
            if not group and topics:  # 빈 그룹인 경우 전체 주제 사용
                group = topics
            topic_groups.append(group)

        return topic_groups

    def _create_diversity_prompt(self, summary: str, topics: List[str], keywords: List[str],
                                num_questions: int, request: QuizRequest, difficulty_type: str) -> str:
        """다양성을 고려한 프롬프트 생성"""

        # 난이도별 특화 지시사항
        difficulty_instructions = {
            "basic": "기본 개념을 명확하게 묻는 문제를 생성하세요. 핵심 용어와 정의에 집중하세요.",
            "concept": "여러 개념을 연결하는 문제를 생성하세요. 개념 간의 관계와 비교를 포함하세요.",
            "application": "실제 상황에 적용하는 문제를 생성하세요. 구체적인 사례와 분석을 포함하세요.",
            "mixed": "다양한 난이도의 문제를 균형있게 생성하세요."
        }

        # 중복 방지 지시사항
        diversity_instruction = """
⚠️ **중복 방지 지침**:
- 같은 키워드나 주제를 반복해서 사용하지 마세요
- 비슷한 질문 형식을 피하세요
- 다양한 관점과 접근 방식을 사용하세요
- 각 문제는 독립적이고 고유해야 합니다
"""

        return self.prompt_manager.get_prompt("question").format(
            summary=summary,
            topics="\n".join(f"- {topic}" for topic in topics),
            keywords="\n".join(f"- {keyword}" for keyword in keywords),
            num_questions=num_questions,
            difficulty=request.difficulty.value,
            question_type=request.question_type.value
        ) + f"\n\n{difficulty_instructions.get(difficulty_type, '')}\n{diversity_instruction}"

    def _advanced_quality_check_with_diversity(self, questions: List[Dict], target_count: int) -> List[Dict]:
        """다양성을 고려한 고급 품질 검사 및 중복 제거"""
        if not questions:
            return []

        # 1단계: 기본 품질 검사
        valid_questions = self._basic_quality_check(questions)

        if len(valid_questions) <= target_count:
            return valid_questions

        # 2단계: 고급 중복 제거 (더 엄격한 기준)
        unique_questions = []
        seen_questions = set()
        keyword_usage = {}  # 키워드 사용 빈도 추적

        for q in valid_questions:
            question_text = q["question"].lower().strip()

            # 더 엄격한 중복 검사 (유사도 기준 상향)
            is_duplicate = False
            for seen in seen_questions:
                if self._calculate_similarity(question_text, seen) > 0.8:  # 더 엄격한 기준
                    is_duplicate = True
                    break

            # 키워드 중복 검사
            if not is_duplicate:
                question_keywords = self._extract_keywords_from_question(question_text)
                keyword_overlap = 0
                for keyword in question_keywords:
                    if keyword_usage.get(keyword, 0) >= 2:  # 같은 키워드가 2번 이상 사용된 경우
                        keyword_overlap += 1

                # 키워드 중복이 너무 많은 경우 제외
                if keyword_overlap > len(question_keywords) * 0.5:  # 50% 이상 중복
                    continue

            if not is_duplicate:
                unique_questions.append(q)
                seen_questions.add(question_text)

                # 키워드 사용 빈도 업데이트
                for keyword in self._extract_keywords_from_question(question_text):
                    keyword_usage[keyword] = keyword_usage.get(keyword, 0) + 1

                # 목표 수에 도달하면 중단
                if len(unique_questions) >= target_count:
                    break

        # 3단계: 품질 점수 기반 정렬 (다양성 가중치 추가)
        scored_questions = []
        for q in unique_questions:
            score = self._calculate_question_score_with_diversity(q, keyword_usage)
            scored_questions.append((score, q))

        # 점수 높은 순으로 정렬
        scored_questions.sort(key=lambda x: x[0], reverse=True)

        # 상위 문제들만 반환
        final_questions = [q for _, q in scored_questions[:target_count]]

        logger.info(f"다양성을 고려한 고급 품질 검사 완료: {len(questions)}개 → {len(final_questions)}개")

        return final_questions

    def _extract_keywords_from_question(self, question_text: str) -> List[str]:
        """문제에서 키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 사용 가능)
        words = question_text.split()
        # 3글자 이상의 단어만 키워드로 간주
        keywords = [word for word in words if len(word) >= 3]
        return keywords[:5]  # 상위 5개만 반환

    def _calculate_question_score_with_diversity(self, question: Dict, keyword_usage: Dict[str, int]) -> float:
        """다양성을 고려한 문제 품질 점수 계산"""
        score = self._calculate_question_score(question)

        # 다양성 보너스
        question_text = question.get("question", "").lower()
        question_keywords = self._extract_keywords_from_question(question_text)

        # 사용 빈도가 낮은 키워드에 보너스
        diversity_bonus = 0
        for keyword in question_keywords:
            usage_count = keyword_usage.get(keyword, 0)
            if usage_count == 0:
                diversity_bonus += 0.3  # 새로운 키워드
            elif usage_count == 1:
                diversity_bonus += 0.1  # 한 번만 사용된 키워드

        score += diversity_bonus

        return score

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (개선된 버전)"""
        from difflib import SequenceMatcher

        # 기본 유사도
        basic_similarity = SequenceMatcher(None, text1, text2).ratio()

        # 키워드 기반 유사도
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return basic_similarity

        # Jaccard 유사도
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_similarity = intersection / union if union > 0 else 0

        # 가중 평균
        return (basic_similarity * 0.7) + (jaccard_similarity * 0.3)

    def _calculate_question_score(self, question: Dict) -> float:
        """문제 품질 점수 계산 (설명 품질 강화)"""
        score = 0.0

        # 기본 점수
        score += 1.0

        # 문제 길이 점수 (적절한 길이)
        question_length = len(question.get("question", ""))
        if 50 <= question_length <= 200:
            score += 0.3
        elif 30 <= question_length <= 300:
            score += 0.2

        # 선택지 개수 점수
        options_count = len(question.get("options", []))
        if options_count == 4:
            score += 0.2
        elif options_count >= 3:
            score += 0.1

        # 설명 품질 점수 (강화)
        explanation = question.get("explanation", "")
        explanation_length = len(explanation)

        # 설명 길이 점수
        if 50 <= explanation_length <= 200:
            score += 0.4
        elif 30 <= explanation_length <= 300:
            score += 0.3
        elif explanation_length > 300:
            score += 0.2

        # 설명 품질 점수 (개념적 설명 포함)
        if any(keyword in explanation.lower() for keyword in ['왜냐하면', '때문에', '이유는', '핵심은', '중요한']):
            score += 0.3

        # 실무 연관성 점수
        if any(keyword in explanation.lower() for keyword in ['실제로', '실무에서', '구체적으로', '예를 들어', '사례']):
            score += 0.2

        # 오답 분석 점수
        if any(keyword in explanation.lower() for keyword in ['틀렸습니다', '오답', '혼동', '차이점']):
            score += 0.2

        # 문제 수준 점수
        level = question.get("problem_level", "basic")
        if level == "application":
            score += 0.2
        elif level == "concept":
            score += 0.1

        return score

    def _advanced_quality_check(self, questions: List[Dict], target_count: int) -> List[Dict]:
        """고급 품질 검증 및 다양성 보장"""
        if not questions:
            return []

        # 1. 기본 품질 검사
        questions = self._basic_quality_check(questions)

        # 2. 품질 점수 계산 및 필터링
        scored_questions = []
        for question in questions:
            score = self._calculate_question_score(question)
            if score >= 0.5:  # 기본 품질 기준
                question['quality_score'] = score
                scored_questions.append(question)

        # 3. 중복성 검사 및 제거
        unique_questions = []
        seen_patterns = set()

        for question in scored_questions:
            # 질문 패턴 생성 (키워드 기반)
            keywords = self._extract_keywords_from_question(question.get('question', ''))
            pattern = '|'.join(sorted(keywords))

            if pattern not in seen_patterns:
                seen_patterns.add(pattern)
                unique_questions.append(question)

        # 4. 다양성 보장
        if len(unique_questions) > target_count:
            # 품질 점수로 정렬하여 상위 문제 선택
            unique_questions.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            unique_questions = unique_questions[:target_count]

        # 5. 최종 검증
        final_questions = []
        for question in unique_questions:
            # 필수 필드 재검증
            if (question.get('question') and
                question.get('options') and
                len(question.get('options', [])) >= 2 and
                question.get('correct_answer') and
                question.get('explanation')):
                final_questions.append(question)

        return final_questions

    def _distribute_difficulty_levels(self, questions: List[Dict], target_count: int, exam_difficulty: str) -> List[Dict]:
        """난이도 분산 처리"""
        # 품질 순으로 정렬
        questions.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

        # 난이도별 분포 설정
        if exam_difficulty == "easy":
            easy_ratio, medium_ratio, hard_ratio = 0.4, 0.4, 0.2
        elif exam_difficulty == "medium":
            easy_ratio, medium_ratio, hard_ratio = 0.2, 0.5, 0.3
        else:  # hard
            easy_ratio, medium_ratio, hard_ratio = 0.1, 0.3, 0.6

        # 각 난이도별 문제 수 계산
        easy_count = int(target_count * easy_ratio)
        medium_count = int(target_count * medium_ratio)
        hard_count = target_count - easy_count - medium_count  # 나머지

        # 난이도별 문제 분류 (AI가 생성한 난이도 기반)
        easy_questions = []
        medium_questions = []
        hard_questions = []

        for question in questions:
            difficulty = question.get('difficulty', 'medium').lower()
            if difficulty in ['easy', '쉬움', '기초']:
                easy_questions.append(question)
            elif difficulty in ['hard', '어려움', '고급']:
                hard_questions.append(question)
            else:
                medium_questions.append(question)

        # 각 난이도별로 필요한 수만큼 선택
        selected_questions = []
        selected_questions.extend(easy_questions[:easy_count])
        selected_questions.extend(medium_questions[:medium_count])
        selected_questions.extend(hard_questions[:hard_count])

        # 부족한 경우 다른 난이도에서 보충
        if len(selected_questions) < target_count:
            remaining = target_count - len(selected_questions)
            remaining_questions = [q for q in questions if q not in selected_questions]
            selected_questions.extend(remaining_questions[:remaining])

        # 최종 정렬 (품질 순)
        selected_questions.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

        logger.info(f"🎯 난이도 분산 완료: 쉬움 {len(easy_questions[:easy_count])}개, 보통 {len(medium_questions[:medium_count])}개, 어려움 {len(hard_questions[:hard_count])}개")

        return selected_questions[:target_count]

    def _basic_quality_check(self, questions: List[Dict]) -> List[Dict]:
        """기본적인 품질 검사 수행"""
        valid_questions = []
        seen_questions = set()

        for q in questions:
            try:
                # 필수 필드 확인
                if not all(k in q for k in ["question", "options", "correct_answer", "explanation"]):
                    continue

                # 중복 문제 제거 (유사도 기반, 기준 완화)
                question_text = q["question"].lower().strip()
                if any(self._is_similar(question_text, seen, threshold=0.9) for seen in seen_questions):  # 유사도 기준 상향
                    continue
                seen_questions.add(question_text)

                # 선택지 검증 (최소 2개 이상)
                if len(q["options"]) < 2:
                    continue

                # 정답이 선택지에 포함되어 있는지 확인 (전처리된 형식 고려)
                correct_answer = q["correct_answer"]
                options = q["options"]

                # 정답이 선택지에 직접 포함되어 있는지 확인
                if correct_answer in options:
                    pass
                else:
                    # 정답에서 번호를 제거하고 내용만 비교
                    import re
                    answer_content = re.sub(r'^\d+\.\s*', '', correct_answer)
                    found = False
                    for option in options:
                        option_content = re.sub(r'^\d+\.\s*', '', option)
                        if answer_content == option_content:
                            found = True
                            break
                    if not found:
                        continue

                # 문제 수준 설정
                if "problem_level" not in q:
                    q["problem_level"] = "basic"

                valid_questions.append(q)
            except Exception as e:
                logger.warning(f"문제 품질 검사 중 오류 발생: {e}")
                continue

        return valid_questions

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.9) -> bool:
        """두 텍스트의 유사도 검사 (기준 상향)"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio() > threshold

    def _parse_questions(self, content: str) -> List[Dict]:
        """JSON 응답 파싱"""
        try:
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
            questions = questions_data.get("questions", [])

            # 전처리 적용
            return self._preprocess_questions(questions)

        except json.JSONDecodeError as e:
            logger.error(f"ERROR JSON 파싱 실패: {e}")
            logger.error(f"LLM 응답 내용: {content[:500]}...")
            return []

    def _preprocess_questions(self, questions: List[Dict]) -> List[Dict]:
        """문제 응답 전처리 - 선택지 번호 중복 및 정답 번호 수정"""
        processed_questions = []

        for question in questions:
            processed_question = question.copy()

            # 선택지 전처리
            if isinstance(question.get("options"), list):
                processed_options = []
                for option in question["options"]:
                    # 번호 중복 제거 (예: "1. 1. 내용" -> "1. 내용")
                    if isinstance(option, str):
                        # 정규표현식으로 번호 중복 패턴 찾기
                        import re
                        # "숫자. 숫자. 내용" 패턴을 "숫자. 내용"으로 변경
                        cleaned_option = re.sub(r'^(\d+)\.\s*\1\.\s*', r'\1. ', option)
                        processed_options.append(cleaned_option)
                    else:
                        processed_options.append(option)

                processed_question["options"] = processed_options

                # correct_answer_number 수정
                if "correct_answer" in question:
                    correct_answer = question["correct_answer"]
                    # correct_answer에서 번호 추출
                    import re
                    match = re.match(r'^(\d+)\.\s*(.+)', correct_answer)
                    if match:
                        answer_number = int(match.group(1))
                        answer_content = match.group(2).strip()
                        processed_question["correct_answer"] = f"{answer_number}. {answer_content}"
                        processed_question["correct_answer_number"] = answer_number
                    else:
                        # correct_answer가 올바른 형식이 아닌 경우, 옵션에서 찾기
                        for i, option in enumerate(processed_options, 1):
                            # 옵션에서 번호 제거 후 내용만 비교
                            option_content = re.sub(r'^\d+\.\s*', '', option)
                            if option_content == correct_answer or option_content in correct_answer:
                                processed_question["correct_answer"] = f"{i}. {option_content}"
                                processed_question["correct_answer_number"] = i
                                break
                        else:
                            # 찾지 못한 경우 첫 번째 옵션을 정답으로 설정
                            if processed_options:
                                first_option = processed_options[0]
                                first_content = re.sub(r'^\d+\.\s*', '', first_option)
                                processed_question["correct_answer"] = f"1. {first_content}"
                                processed_question["correct_answer_number"] = 1

            processed_questions.append(processed_question)

        return processed_questions

    def smart_truncate(self, text, max_length=2000):
        """앞/중간/끝 샘플링 방식으로 텍스트를 자름"""
        if len(text) <= max_length:
            return text
        part = max_length // 3
        return text[:part] + text[len(text)//2:len(text)//2+part] + text[-part:]

    async def generate_quiz(self, request: QuizRequest, documents: List[Dict[str, Any]], use_combined_prompt: bool = False, use_sampling: bool = False) -> Dict[str, Any]:
        """
        문제 생성 메인 메서드 (문서별 전처리까지 완전 비동기 + 문제 생성 2문제씩 병렬)
        use_combined_prompt: 무시(항상 분리 방식)
        use_sampling: True면 샘플링(앞/중간/끝), False면 전체 결합
        """
        import time
        try:
            total_start = time.time()
            logger.info(f"🚀 문제 생성 AI 에이전트 시작 (문서별 전처리 완전 비동기, 문제 생성 병렬 2문제씩, use_sampling={use_sampling})")

            # 난이도 값 검증
            if not isinstance(request.difficulty, DifficultyLevel):
                try:
                    request.difficulty = DifficultyLevel(request.difficulty)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"잘못된 파라미터: '{request.difficulty}' is not a valid DifficultyLevel",
                        "valid_difficulty": [level.value for level in DifficultyLevel],
                        "valid_question_types": [qtype.value for qtype in QuestionType]
                    }

            # 문제 유형 값 검증
            if not isinstance(request.question_type, QuestionType):
                try:
                    request.question_type = QuestionType(request.question_type)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"잘못된 파라미터: '{request.question_type}' is not a valid QuestionType",
                        "valid_difficulty": [level.value for level in DifficultyLevel],
                        "valid_question_types": [qtype.value for qtype in QuestionType]
                    }

            preprocess_start = time.time()
            logger.info(f"[전처리] 시작 (문서별 완전 비동기, use_sampling={use_sampling})")

            async def process_single_doc(doc):
                filename = doc.get("filename", "Unknown")
                content = doc.get("content", "")
                if use_sampling:
                    content = self.smart_truncate(content, 2000)

                                                                # 🔥 최적화: 토큰 제한 고려한 프롬프트 사용
                # 컨텐츠 길이 제한 (토큰 제한 방지)
                max_content_length = 3000  # 토큰 제한 고려
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."

                combined_prompt = self.prompt_manager.get_prompt("combined_preprocessing").format(
                    content=content,
                    difficulty=request.difficulty.value,
                    question_type=request.question_type.value,
                    num_questions=request.num_questions
                )

                # 단일 AI 호출
                combined_resp = await self.summary_chain.ainvoke({"prompt": combined_prompt})

                # 프롬프트 매니저를 통한 응답 파싱
                return self.prompt_manager.parse_combined_response(combined_resp.content)

            doc_tasks = [process_single_doc(doc) for doc in documents]
            doc_results = await asyncio.gather(*doc_tasks)

            # 결과 합치기
            summary = "\n".join([str(r.get("summary", "")) for r in doc_results])
            topics = []
            for r in doc_results:
                topics.extend([line.strip().lstrip('- •').strip() for line in r["topics"].split('\n') if line.strip().startswith(('-', '•'))])
            keywords = []
            for r in doc_results:
                keywords.extend([kw.strip() for kw in r["keywords"].split(',') if kw.strip()])
            logger.info(f"[전처리] 완료 (총 소요 시간: {time.time() - preprocess_start:.2f}초)")

            # 2. 문제 생성: 더 많은 문제를 생성하여 부족한 경우 대비
            generate_start = time.time()
            logger.info(f"[문제 생성] 시작 (목표: {request.num_questions}개)")

            # 진행률 추적
            progress = {
                "total_steps": 3,
                "current_step": 1,
                "step_name": "문제 생성",
                "progress_percent": 33
            }
            logger.info(f"📊 진행률: {progress['progress_percent']}% - {progress['step_name']}")

            # 🔥 최적화: 토큰 제한 고려한 배치 크기 조정
            target_questions = int(request.num_questions * 1.3)  # 1.3배로 조정 (토큰 제한 고려)
            batch_size = 3  # 배치 크기 5에서 3으로 줄임 (토큰 제한 고려)
            total_batches = (target_questions + batch_size - 1) // batch_size

            logger.info(f"🎯 목표 생성: {target_questions}개 (요청: {request.num_questions}개 + 여유분)")

            # 🔥 고급 캐싱: Redis + 메모리 이중 캐싱
            import hashlib
            content_hash = hashlib.md5((summary + str(topics) + str(keywords)).encode()).hexdigest()
            cache_key = f"quiz_{content_hash}_{request.difficulty.value}_{request.question_type.value}_{request.num_questions}"

            # 1. 메모리 캐시 확인
            if hasattr(self, '_question_cache') and cache_key in self._question_cache:
                logger.info(f"🎯 메모리 캐시 사용: {len(self._question_cache[cache_key])}개")
                cached_questions = self._question_cache[cache_key]
                if len(cached_questions) >= request.num_questions:
                    return {
                        "success": True,
                        "request": {
                            "file_ids": request.file_ids,
                            "num_questions": request.num_questions,
                            "difficulty": request.difficulty.value,
                            "question_type": request.question_type.value
                        },
                        "process_info": {
                            "summary": summary,
                            "core_topics": topics,
                            "keywords": keywords
                        },
                        "questions": cached_questions[:request.num_questions],
                        "meta": {
                            "generated_count": len(cached_questions[:request.num_questions]),
                            "final_step": "generate_quiz",
                            "cached": True,
                            "quality_score": sum(q.get('quality_score', 0) for q in cached_questions[:request.num_questions]) / len(cached_questions[:request.num_questions]) if cached_questions else 0,
                            "generation_strategy": "cached"
                        }
                    }

            # 2. Redis 캐시 확인 (선택적)
            try:
                import redis
                redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    import json
                    questions = json.loads(cached_result)
                    logger.info(f"🎯 Redis 캐시 사용: {len(questions)}개")
                    # 메모리 캐시에도 저장
                    if not hasattr(self, '_question_cache'):
                        self._question_cache = {}
                    self._question_cache[cache_key] = questions
                    if len(questions) >= request.num_questions:
                        return {
                            "success": True,
                            "request": {
                                "file_ids": request.file_ids,
                                "num_questions": request.num_questions,
                                "difficulty": request.difficulty.value,
                                "question_type": request.question_type.value
                            },
                            "process_info": {
                                "summary": summary,
                                "core_topics": topics,
                                "keywords": keywords
                            },
                            "questions": questions[:request.num_questions],
                            "meta": {
                                "generated_count": len(questions[:request.num_questions]),
                                "final_step": "generate_quiz",
                                "cached": True
                            }
                        }
            except Exception as e:
                logger.warning(f"Redis 캐시 확인 실패: {e}")

            async def generate_questions_batch(batch_num):
                # 🔥 최적화된 AI 기반 문제 생성 (토큰 제한 고려)
                # 요약과 주제 길이 제한
                summary_limited = summary[:1000] if len(summary) > 1000 else summary
                topics_limited = topics[:10]  # 최대 10개 주제만
                keywords_limited = keywords[:15]  # 최대 15개 키워드만

                base_prompt = self.prompt_manager.get_prompt("question").format(
                    summary=summary_limited,
                    topics="\n".join(f"- {topic}" for topic in topics_limited),
                    keywords="\n".join(f"- {keyword}" for keyword in keywords_limited),
                    num_questions=batch_size,
                    difficulty=request.difficulty.value,
                    question_type=request.question_type.value
                )

                # 카테고리 특화 프롬프트 추가 (토큰 제한 고려)
                if request.category or request.sub_category:
                    category_prompt = self.prompt_manager.get_category_prompt(
                        request.category or "general",
                        request.sub_category
                    )
                    # 카테고리 프롬프트도 길이 제한
                    if len(category_prompt) > 2000:
                        category_prompt = category_prompt[:2000] + "..."
                    question_prompt = base_prompt + "\n\n" + category_prompt
                else:
                    question_prompt = base_prompt

                # 🔥 최적화: 토큰 제한 고려한 에러 처리
                try:
                    # 프롬프트 길이 체크
                    if len(question_prompt) > 8000:  # 토큰 제한 고려
                        logger.warning(f"프롬프트가 너무 깁니다: {len(question_prompt)}자")
                        question_prompt = question_prompt[:8000] + "..."

                    response = await self.question_chain.ainvoke({
                        "system_message": "당신은 전문 교육 컨텐츠 개발자입니다. 고품질의 문제를 생성하세요.",
                        "prompt": question_prompt
                    })
                    return self._parse_questions(response.content)
                except Exception as e:
                    error_msg = str(e)
                    if "context_length_exceeded" in error_msg or "tokens" in error_msg:
                        logger.error(f"토큰 제한 초과 (배치 {batch_num}): {error_msg}")
                        # 토큰 제한 시 더 작은 배치로 재시도
                        return []
                    else:
                        logger.error(f"문제 생성 실패 (배치 {batch_num}): {error_msg}")
                        return []

            # 🔥 최적화: 배치 처리 개선
            tasks = [generate_questions_batch(i) for i in range(total_batches)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 성공한 배치만 수집
            questions = []
            failed_batches = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"배치 {i} 실패: {result}")
                    failed_batches += 1
                else:
                    questions.extend(result)

            if failed_batches > 0:
                logger.warning(f"⚠️ {failed_batches}/{total_batches} 배치 실패")

            logger.info(f"생성된 문제: {len(questions)}개")

                        # 🔥 고급 품질 검증 및 최적화
            questions = self._advanced_quality_check(questions, target_questions)

            # 품질 점수 계산
            quality_scores = [self._calculate_question_score(q) for q in questions]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            logger.info(f"평균 품질 점수: {avg_quality:.2f}")

            # 품질이 낮은 문제 필터링 (점수 0.5 미만 제거 - 기준 완화)
            high_quality_questions = [q for q, score in zip(questions, quality_scores) if score >= 0.5]
            logger.info(f"고품질 문제: {len(high_quality_questions)}/{len(questions)}개")

            # 여유분이 충분한지 확인
            if len(high_quality_questions) >= request.num_questions:
                logger.info(f"✅ 충분한 문제 확보: {len(high_quality_questions)}개 (요청: {request.num_questions}개)")

                # 🔥 난이도 분산 처리
                questions = self._distribute_difficulty_levels(
                    high_quality_questions,
                    request.num_questions,
                    request.difficulty.value
                )
            else:
                logger.warning(f"⚠️ 문제 부족: {len(high_quality_questions)}개 (요청: {request.num_questions}개)")
                questions = high_quality_questions  # 있는 만큼 반환

            logger.info(f"[문제 생성] 완료 (소요 시간: {time.time() - generate_start:.2f}초)")

            # 3. 후처리: ID 부여 및 최종 정리
            post_start = time.time()
            logger.info("[후처리] 시작")
            # 품질 검사는 이미 문제 생성 단계에서 완료됨
            questions = questions[:request.num_questions]  # 요청 수만큼만 반환
            for i, question in enumerate(questions, 1):
                question["id"] = i
                # 전처리에서 이미 선택지 번호와 correct_answer_number가 처리되었으므로 추가 처리 제거
                # 다중선택 문제의 경우 전처리에서 이미 올바른 형식으로 처리됨
            logger.info(f"[후처리] 완료 (소요 시간: {time.time() - post_start:.2f}초)")

            total_end = time.time()
            logger.info(f"[실행시간] 전체 문제 생성 프로세스 소요 시간: {total_end - total_start:.2f}초")
            logger.info("🎉 SUCCESS 문제 생성 완료")

            # 🔥 캐시에 결과 저장 (Redis + 메모리)
            self._question_cache[cache_key] = questions

            # Redis 캐시에도 저장 (선택적)
            try:
                import redis
                import json
                redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                redis_client.setex(cache_key, 3600, json.dumps(questions))  # 1시간 TTL
                logger.info(f"🎯 Redis 캐시 저장 완료: {len(questions)}개")
            except Exception as e:
                logger.warning(f"Redis 캐시 저장 실패: {e}")

            return {
                "success": True,
                "request": {
                    "file_ids": request.file_ids,
                    "num_questions": request.num_questions,
                    "difficulty": request.difficulty.value,
                    "question_type": request.question_type.value
                },
                "process_info": {
                    "summary": summary,
                    "core_topics": topics,
                    "keywords": keywords
                },
                "questions": questions,
                "meta": {
                    "generated_count": len(questions),
                    "final_step": "generate_quiz",
                    "cached": False
                }
            }
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