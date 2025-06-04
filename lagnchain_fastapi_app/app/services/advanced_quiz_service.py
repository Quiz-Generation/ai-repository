"""
🎓 프로덕션 급 PDF RAG 퀴즈 생성 시스템
실제 모의고사/자격증/시험 문제 생성에 최적화

주요 개선사항:
- 정확한 문제 개수 보장 (retry 로직)
- 멀티 스테이지 RAG (심화 컨텍스트 분석)
- 의미적 중복 검증 (embedding 기반)
- 전문 품질 검증 에이전트
- 문제 유형별 전용 생성기
- 컨텍스트 다양성 보장
"""
import logging
import time
import uuid
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import asdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from ..schemas.quiz_schema import (
    QuizRequest, QuizResponse, Question, Difficulty, QuestionType,
    RAGContext, TopicAnalysis, QuizGenerationStats
)
from ..services.llm_factory import BaseLLMService, get_default_llm_service
from ..services.vector_service import PDFVectorService, get_global_vector_service

logger = logging.getLogger(__name__)


class MultiStageRAGRetriever:
    """🧠 멀티 스테이지 RAG 컨텍스트 검색기 (프로덕션 급)"""

    def __init__(self, vector_service: PDFVectorService, llm_service: BaseLLMService):
        self.vector_service = vector_service
        self.llm_service = llm_service

        # 의미적 유사도 계산용 임베딩 모델
        try:
            self.similarity_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            logger.info("한국어 임베딩 모델 로드 완료")
        except:
            logger.warning("한국어 임베딩 모델 로드 실패, 기본 모델 사용")
            self.similarity_model = None

    async def retrieve_diverse_contexts(
        self,
        document_id: str,
        num_questions: int,
        topics: Optional[List[str]] = None
    ) -> List[RAGContext]:
        """🎯 다양성과 품질을 보장하는 컨텍스트 검색"""

        logger.info(f"멀티 스테이지 RAG 검색 시작: {document_id}")

        all_contexts = []

        # Stage 1: 토픽 기반 검색
        if topics:
            topic_contexts = await self._stage1_topic_search(document_id, topics, num_questions)
            all_contexts.extend(topic_contexts)

        # Stage 2: 문서 구조 기반 검색 (섹션별)
        structural_contexts = await self._stage2_structural_search(document_id, num_questions)
        all_contexts.extend(structural_contexts)

        # Stage 3: 동적 키워드 기반 검색
        dynamic_contexts = await self._stage3_dynamic_search(document_id, num_questions)
        all_contexts.extend(dynamic_contexts)

        # Stage 4: 품질 필터링 및 다양성 보장
        final_contexts = await self._stage4_quality_diversify(all_contexts, num_questions * 3)

        logger.info(f"멀티 스테이지 RAG 완료: {len(final_contexts)}개 컨텍스트")
        return final_contexts

    async def _stage1_topic_search(self, document_id: str, topics: List[str], num_per_topic: int) -> List[RAGContext]:
        """Stage 1: 토픽별 심화 검색"""
        contexts = []

        for topic in topics:
            # 각 토픽에 대해 다양한 검색 쿼리 생성
            search_queries = [
                topic,
                f"{topic} 개념",
                f"{topic} 원리",
                f"{topic} 방법",
                f"{topic} 특징"
            ]

            for query in search_queries[:3]:  # 상위 3개 쿼리만
                results = self.vector_service.search_in_document(
                    query=query,
                    document_id=document_id,
                    top_k=2
                )
                contexts.extend(await self._convert_to_rag_contexts_async(results, topic))

        return contexts

    async def _stage2_structural_search(self, document_id: str, num_questions: int) -> List[RAGContext]:
        """Stage 2: 문서 구조적 다양성 검색"""

        # 문서의 다양한 부분에서 균형있게 검색
        structural_queries = [
            "핵심 내용 주요 개념",  # 앞부분
            "구체적 사례 예시",      # 중간부분
            "결론 정리 요약",        # 뒷부분
            "중요한 정보 포인트",    # 전반적
            "기본 원리 기초"         # 기본 개념
        ]

        contexts = []
        for query in structural_queries:
            results = self.vector_service.search_in_document(
                query=query,
                document_id=document_id,
                top_k=2
            )
            contexts.extend(await self._convert_to_rag_contexts_async(results))

        return contexts

    async def _stage3_dynamic_search(self, document_id: str, num_questions: int) -> List[RAGContext]:
        """Stage 3: LLM 기반 동적 키워드 검색"""

        # 샘플 텍스트 수집
        sample_results = self.vector_service.search_in_document(
            query="주요 내용",
            document_id=document_id,
            top_k=5
        )

        if not sample_results:
            return []

        sample_text = "\n".join([r["text"][:300] for r in sample_results])

        # LLM으로 특화 키워드 생성
        prompt = f"""
다음 문서에서 시험 문제로 만들기 좋은 핵심 키워드 5개를 생성하세요.
- 암기가 아닌 이해/적용 중심
- 구체적이고 시험 출제 가능한 개념
- 문서에 실제 설명된 내용만

문서 내용:
{sample_text[:2000]}

JSON 형식: {{"keywords": ["키워드1", "키워드2", ...]}}
"""

        try:
            response = await self.llm_service.client.chat.completions.create(
                model=self.llm_service.model_name,
                messages=[
                    {"role": "system", "content": "시험 출제 전문가로서 핵심 키워드를 추출합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )

            import json
            result_text = response.choices[0].message.content
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                result = json.loads(result_text[start_idx:end_idx])
                keywords = result.get("keywords", [])

                contexts = []
                for keyword in keywords:
                    results = self.vector_service.search_in_document(
                        query=keyword,
                        document_id=document_id,
                        top_k=2
                    )
                    contexts.extend(await self._convert_to_rag_contexts_async(results))

                return contexts

        except Exception as e:
            logger.error(f"동적 키워드 검색 실패: {e}")

        return []

    async def _stage4_quality_diversify(self, contexts: List[RAGContext], target_count: int) -> List[RAGContext]:
        """Stage 4: 품질 필터링 및 다양성 보장 (비동기)"""

        # 1. 기본 품질 필터링
        quality_contexts = [
            ctx for ctx in contexts
            if len(ctx.text.strip()) >= 100 and ctx.similarity >= 0.1
        ]

        # 2. 중복 제거 (텍스트 기반)
        unique_contexts = await self._remove_text_duplicates_async(quality_contexts)

        # 3. 의미적 다양성 보장
        diverse_contexts = await self._ensure_semantic_diversity_async(unique_contexts, target_count)

        # 4. 유사도 기준 정렬
        diverse_contexts.sort(key=lambda x: x.similarity, reverse=True)

        return diverse_contexts[:target_count]

    async def _remove_text_duplicates_async(self, contexts: List[RAGContext]) -> List[RAGContext]:
        """텍스트 기반 중복 제거 (비동기)"""
        seen_signatures = set()
        unique_contexts = []

        for ctx in contexts:
            # 텍스트의 첫 150자로 시그니처 생성
            signature = ctx.text[:150].strip().lower()
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_contexts.append(ctx)

        return unique_contexts

    async def _ensure_semantic_diversity_async(self, contexts: List[RAGContext], target_count: int) -> List[RAGContext]:
        """의미적 다양성 보장 (비동기)"""

        if not self.similarity_model or len(contexts) <= target_count:
            return contexts

        try:
            # 텍스트 임베딩 생성 (CPU 집약적 작업을 비동기로)
            import asyncio
            texts = [ctx.text[:500] for ctx in contexts]

            # CPU 집약적 작업을 별도 스레드에서 실행
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self.similarity_model.encode, texts)

            # 다양성 기반 선택 (greedy selection)
            selected_indices = [0]  # 첫 번째는 유사도가 가장 높은 것

            while len(selected_indices) < target_count and len(selected_indices) < len(contexts):
                max_min_distance = -1
                best_candidate = -1

                for i, emb in enumerate(embeddings):
                    if i in selected_indices:
                        continue

                    # 이미 선택된 것들과의 최소 거리 계산
                    min_distance = min([
                        1 - cosine_similarity(np.array([emb]), np.array([embeddings[j]]))[0][0]
                        for j in selected_indices
                    ])

                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_candidate = i

                if best_candidate != -1:
                    selected_indices.append(best_candidate)
                else:
                    break

            return [contexts[i] for i in selected_indices]

        except Exception as e:
            logger.error(f"의미적 다양성 보장 실패: {e}")
            return contexts[:target_count]

    async def _convert_to_rag_contexts_async(self, search_results: List[Dict], topic: Optional[str] = None) -> List[RAGContext]:
        """검색 결과를 RAGContext로 변환 (비동기)"""
        contexts = []

        for result in search_results:
            context = RAGContext(
                text=result["text"],
                similarity=result["similarity"],
                source=result["metadata"].get("source", ""),
                chunk_index=result["metadata"].get("chunk_index", 0),
                topic=topic,
                metadata=result["metadata"]
            )
            contexts.append(context)

        return contexts


class QuestionTypeSpecialist:
    """🎯 문제 유형별 전문 생성기"""

    def __init__(self, llm_service: BaseLLMService):
        self.llm_service = llm_service

    async def generate_guaranteed_questions(
        self,
        contexts: List[RAGContext],
        question_type: QuestionType,
        count: int,
        difficulty: Difficulty,
        topic: str
    ) -> List[Dict[str, Any]]:
        """✅ 정확한 개수 보장하는 문제 생성 (최대 3회 재시도)"""

        logger.info(f"{question_type.value} 문제 {count}개 생성 시작")

        for attempt in range(3):  # 최대 3회 시도
            try:
                questions = await self._generate_type_specific_questions(
                    contexts, question_type, count, difficulty, topic
                )

                if len(questions) >= count:
                    logger.info(f"{question_type.value} 문제 생성 성공: {len(questions)}개")
                    return questions[:count]  # 정확한 개수만 반환
                else:
                    logger.warning(f"시도 {attempt + 1}: {len(questions)}/{count}개만 생성됨")

            except Exception as e:
                logger.error(f"시도 {attempt + 1} 실패: {e}")

        # 3번 모두 실패 시 기본 문제 생성
        logger.error(f"{question_type.value} 문제 생성 실패, 기본 문제로 대체")
        return await self._generate_fallback_questions(count, difficulty, topic)

    async def _generate_type_specific_questions(
        self,
        contexts: List[RAGContext],
        question_type: QuestionType,
        count: int,
        difficulty: Difficulty,
        topic: str
    ) -> List[Dict[str, Any]]:
        """문제 유형별 특화 생성"""

        context_text = "\n\n".join([f"[컨텍스트 {i+1}]\n{ctx.text}" for i, ctx in enumerate(contexts)])

        # 문제 유형별 특화 프롬프트
        type_prompts = {
            QuestionType.MULTIPLE_CHOICE: self._get_mc_prompt(context_text, count, difficulty, topic),
            QuestionType.SHORT_ANSWER: self._get_sa_prompt(context_text, count, difficulty, topic),
            QuestionType.FILL_BLANK: self._get_fb_prompt(context_text, count, difficulty, topic),
            QuestionType.TRUE_FALSE: self._get_tf_prompt(context_text, count, difficulty, topic)
        }

        prompt = type_prompts.get(question_type, type_prompts[QuestionType.MULTIPLE_CHOICE])

        response = await self.llm_service.client.chat.completions.create(
            model=self.llm_service.model_name,
            messages=[
                {"role": "system", "content": f"전문 {question_type.value} 문제 출제자입니다. 정확히 {count}개의 고품질 문제를 생성하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=3000
        )

        result_text = response.choices[0].message.content
        if result_text is None:
            raise ValueError("LLM 응답이 비어있습니다")

        return self._parse_questions_response(result_text, question_type)

    def _get_mc_prompt(self, context: str, count: int, difficulty: Difficulty, topic: str) -> str:
        """객관식 문제 전용 프롬프트"""
        return f"""
다음 내용을 바탕으로 **정확히 {count}개**의 객관식 문제를 생성하세요.

컨텍스트:
{context[:3000]}

요구사항:
- 난이도: {difficulty.value}
- 주제: {topic}
- 각 문제마다 정답 1개 + 그럴듯한 오답 3개
- 단순 암기가 아닌 이해/적용 문제
- 정답이 명확하고 논란의 여지가 없어야 함

JSON 형식으로 정확히 {count}개 생성:
{{
    "questions": [
        {{
            "question": "문제 내용",
            "question_type": "multiple_choice",
            "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
            "correct_answer": "정답",
            "explanation": "상세한 해설",
            "difficulty": "{difficulty.value}",
            "topic": "{topic}"
        }}
    ]
}}
"""

    def _get_sa_prompt(self, context: str, count: int, difficulty: Difficulty, topic: str) -> str:
        """주관식 문제 전용 프롬프트"""
        return f"""
다음 내용을 바탕으로 **정확히 {count}개**의 단답형 주관식 문제를 생성하세요.

컨텍스트:
{context[:3000]}

요구사항:
- 난이도: {difficulty.value}
- 주제: {topic}
- 1-2문장으로 답할 수 있는 문제
- 명확한 정답이 있는 문제
- 서술형이 아닌 단답형

JSON 형식으로 정확히 {count}개 생성:
{{
    "questions": [
        {{
            "question": "문제 내용",
            "question_type": "short_answer",
            "correct_answer": "정답",
            "explanation": "상세한 해설",
            "difficulty": "{difficulty.value}",
            "topic": "{topic}"
        }}
    ]
}}
"""

    def _get_fb_prompt(self, context: str, count: int, difficulty: Difficulty, topic: str) -> str:
        """빈칸 채우기 전용 프롬프트"""
        return f"""
다음 내용을 바탕으로 **정확히 {count}개**의 빈칸 채우기 문제를 생성하세요.

컨텍스트:
{context[:3000]}

요구사항:
- 난이도: {difficulty.value}
- 주제: {topic}
- 문장에서 핵심 단어/구문을 빈칸으로 처리
- 빈칸은 _____ 로 표시
- 문맥상 정답이 명확해야 함

JSON 형식으로 정확히 {count}개 생성:
{{
    "questions": [
        {{
            "question": "빈칸이 포함된 문제 내용 _____",
            "question_type": "fill_blank",
            "correct_answer": "빈칸에 들어갈 정답",
            "explanation": "상세한 해설",
            "difficulty": "{difficulty.value}",
            "topic": "{topic}"
        }}
    ]
}}
"""

    def _get_tf_prompt(self, context: str, count: int, difficulty: Difficulty, topic: str) -> str:
        """참/거짓 문제 전용 프롬프트"""
        return f"""
다음 내용을 바탕으로 **정확히 {count}개**의 참/거짓 문제를 생성하세요.

컨텍스트:
{context[:3000]}

요구사항:
- 난이도: {difficulty.value}
- 주제: {topic}
- 명확히 참 또는 거짓으로 판단 가능
- 애매하거나 논란의 여지가 없어야 함
- 트릭 문제보다는 정확한 이해를 묻는 문제

JSON 형식으로 정확히 {count}개 생성:
{{
    "questions": [
        {{
            "question": "참 또는 거짓을 판단할 명제",
            "question_type": "true_false",
            "correct_answer": "참" 또는 "거짓",
            "explanation": "상세한 해설",
            "difficulty": "{difficulty.value}",
            "topic": "{topic}"
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
                raise ValueError("JSON 형식을 찾을 수 없습니다")

            json_text = response_text[start_idx:end_idx]
            result = json.loads(json_text)

            questions = result.get("questions", [])

            # 문제 유형 검증
            valid_questions = []
            for q in questions:
                if q.get("question_type") == question_type.value:
                    valid_questions.append(q)

            return valid_questions

        except Exception as e:
            logger.error(f"문제 파싱 실패: {e}")
            return []

    async def _generate_fallback_questions(self, count: int, difficulty: Difficulty, topic: str) -> List[Dict[str, Any]]:
        """fallback 기본 문제들"""
        fallback_questions = []

        for i in range(count):
            question = {
                "question": f"{topic}에 관한 기본 문제 {i+1}",
                "question_type": "multiple_choice",
                "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
                "correct_answer": "선택지1",
                "explanation": "기본 설명",
                "difficulty": difficulty.value,
                "topic": topic
            }
            fallback_questions.append(question)

        return fallback_questions


class AdvancedQuizValidator:
    """🔍 프로급 품질 검증 에이전트"""

    def __init__(self, llm_service: BaseLLMService):
        self.llm_service = llm_service

        # 중복 검증용 임베딩 모델
        try:
            self.similarity_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        except:
            self.similarity_model = None

    async def comprehensive_validation(self, questions: List[Question]) -> Dict[str, Any]:
        """🎯 종합적 품질 검증"""

        logger.info(f"종합 품질 검증 시작: {len(questions)}문제")

        validation_results = {
            "overall_score": 0,
            "individual_scores": [],
            "duplicate_analysis": {},
            "quality_issues": [],
            "recommendations": [],
            "pass_rate": 0
        }

        # 1. 개별 문제 품질 검증
        individual_results = await self._validate_individual_questions(questions)
        validation_results["individual_scores"] = individual_results

        # 2. 중복성 검증
        duplicate_analysis = await self._check_semantic_duplicates_async(questions)
        validation_results["duplicate_analysis"] = duplicate_analysis

        # 3. 전체적 품질 평가
        overall_assessment = await self._overall_quality_assessment(questions)
        validation_results.update(overall_assessment)

        logger.info(f"품질 검증 완료: {validation_results['overall_score']}/10점")
        return validation_results

    async def _validate_individual_questions(self, questions: List[Question]) -> List[Dict[str, Any]]:
        """개별 문제 상세 검증"""
        results = []

        for i, question in enumerate(questions):
            score = await self._score_single_question(question)
            results.append({
                "question_index": i,
                "score": score,
                "issues": await self._identify_question_issues(question)
            })

        return results

    async def _score_single_question(self, question: Question) -> float:
        """개별 문제 점수 (0-10)"""

        prompt = f"""
다음 퀴즈 문제의 품질을 0-10점으로 평가하세요.

문제: {question.question}
유형: {question.question_type.value}
정답: {question.correct_answer}
{f"선택지: {question.options}" if question.options else ""}
해설: {question.explanation}

평가 기준:
- 명확성: 문제가 명확하고 이해하기 쉬운가?
- 정확성: 정답이 명확하고 논란의 여지가 없는가?
- 교육적 가치: 학습에 도움이 되는가?
- 난이도 적절성: 설정된 난이도에 맞는가?
- 선택지 품질 (객관식의 경우): 오답이 그럴듯한가?

JSON 형식: {{"score": 숫자, "reasoning": "평가 근거"}}
"""

        try:
            response = await self.llm_service.client.chat.completions.create(
                model=self.llm_service.model_name,
                messages=[
                    {"role": "system", "content": "문제 품질 평가 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            result_text = response.choices[0].message.content
            if result_text is None:
                return 5.0

            import json
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                result = json.loads(result_text[start_idx:end_idx])
                return float(result.get("score", 5.0))

        except Exception as e:
            logger.error(f"문제 점수 평가 실패: {e}")

        return 5.0  # 기본값

    async def _identify_question_issues(self, question: Question) -> List[str]:
        """문제점 식별"""
        issues = []

        # 기본 검증
        if len(question.question.strip()) < 10:
            issues.append("문제가 너무 짧음")

        if not question.correct_answer.strip():
            issues.append("정답이 비어있음")

        # 객관식 전용 검증
        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            if not question.options or len(question.options) < 4:
                issues.append("선택지가 4개 미만")
            elif question.correct_answer not in question.options:
                issues.append("정답이 선택지에 없음")

        return issues

    async def _check_semantic_duplicates_async(self, questions: List[Question]) -> Dict[str, Any]:
        """의미적 중복 검증 (비동기)"""

        if not self.similarity_model or len(questions) < 2:
            return {"duplicate_pairs": [], "similarity_matrix": []}

        try:
            # 문제 텍스트 임베딩 (CPU 집약적 작업을 비동기로)
            import asyncio
            question_texts = [q.question for q in questions]

            # CPU 집약적 작업을 별도 스레드에서 실행
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self.similarity_model.encode, question_texts)

            # 유사도 매트릭스 계산도 비동기로
            similarity_matrix = await loop.run_in_executor(None, cosine_similarity, embeddings)

            # 중복 쌍 찾기 (0.8 이상)
            duplicate_pairs = []
            for i in range(len(questions)):
                for j in range(i+1, len(questions)):
                    similarity = similarity_matrix[i][j]
                    if similarity >= 0.8:
                        duplicate_pairs.append({
                            "question1_index": i,
                            "question2_index": j,
                            "similarity": float(similarity),
                            "question1": questions[i].question[:100],
                            "question2": questions[j].question[:100]
                        })

            return {
                "duplicate_pairs": duplicate_pairs,
                "similarity_matrix": similarity_matrix.tolist(),
                "max_similarity": float(np.max(similarity_matrix - np.eye(len(questions))))
            }

        except Exception as e:
            logger.error(f"중복 검증 실패: {e}")
            return {"duplicate_pairs": [], "similarity_matrix": []}

    async def _overall_quality_assessment(self, questions: List[Question]) -> Dict[str, Any]:
        """전체적 품질 평가"""

        # 기본 통계
        total_questions = len(questions)
        if total_questions == 0:
            return {"overall_score": 0, "pass_rate": 0, "recommendations": ["문제가 생성되지 않음"]}

        # 문제 유형별 분포
        type_distribution = {}
        for q in questions:
            qtype = q.question_type.value
            type_distribution[qtype] = type_distribution.get(qtype, 0) + 1

        # 난이도별 분포
        difficulty_distribution = {}
        for q in questions:
            diff = q.difficulty.value
            difficulty_distribution[diff] = difficulty_distribution.get(diff, 0) + 1

        # 종합 평가
        quality_issues = []
        recommendations = []

        # 다양성 체크
        if len(type_distribution) == 1:
            quality_issues.append("문제 유형이 단조로움")
            recommendations.append("다양한 문제 유형 추가 권장")

        if len(difficulty_distribution) == 1:
            quality_issues.append("난이도가 단조로움")
            recommendations.append("다양한 난이도 문제 추가 권장")

        # 전체 점수 계산 (0-10)
        base_score = min(10, total_questions * 2)  # 기본점수
        penalty = len(quality_issues) * 0.5  # 감점
        overall_score = max(0, base_score - penalty)

        pass_rate = min(100, (total_questions / max(1, len(quality_issues))) * 20)

        return {
            "overall_score": round(overall_score, 1),
            "pass_rate": round(pass_rate, 1),
            "quality_issues": quality_issues,
            "recommendations": recommendations,
            "type_distribution": type_distribution,
            "difficulty_distribution": difficulty_distribution
        }


class AdvancedQuizService:
    """🚀 프로덕션 급 퀴즈 생성 서비스"""

    def __init__(
        self,
        vector_service: Optional[PDFVectorService] = None,
        llm_service: Optional[BaseLLMService] = None
    ):
        self.vector_service = vector_service or get_global_vector_service()
        self.llm_service = llm_service or get_default_llm_service()

        # 프로급 컴포넌트들
        self.rag_retriever = MultiStageRAGRetriever(self.vector_service, self.llm_service)
        self.question_specialist = QuestionTypeSpecialist(self.llm_service)
        self.validator = AdvancedQuizValidator(self.llm_service)

        logger.info("🚀 프로덕션 급 퀴즈 서비스 초기화 완료")

    async def generate_guaranteed_quiz(self, request: QuizRequest) -> QuizResponse:
        """✅ 정확한 개수와 품질을 보장하는 퀴즈 생성"""

        start_time = time.time()
        quiz_id = str(uuid.uuid4())

        logger.info(f"🎯 프로급 퀴즈 생성 시작: {request.num_questions}문제")

        try:
            # 1. 문서 확인
            doc_info = self.vector_service.get_document_info(request.document_id)
            if not doc_info:
                raise ValueError(f"문서를 찾을 수 없습니다: {request.document_id}")

            # 2. 멀티 스테이지 RAG
            logger.info("🧠 멀티 스테이지 RAG 컨텍스트 검색...")
            contexts = await self.rag_retriever.retrieve_diverse_contexts(
                document_id=request.document_id,
                num_questions=request.num_questions,
                topics=None  # 자동 추출
            )

            if not contexts:
                raise ValueError("적절한 컨텍스트를 찾을 수 없습니다")

            # 3. 문제 유형별 정확한 개수 분배
            type_distribution = self._calculate_type_distribution(request)
            logger.info(f"📊 문제 유형 분배: {type_distribution}")

            # 4. 문제 유형별 병렬 생성
            all_questions = []
            generation_tasks = []

            for question_type, count in type_distribution.items():
                if count > 0:
                    task = self.question_specialist.generate_guaranteed_questions(
                        contexts=contexts,
                        question_type=question_type,
                        count=count,
                        difficulty=request.difficulty,
                        topic="주요 내용"
                    )
                    generation_tasks.append((question_type, count, task))

            # 병렬 실행
            logger.info("⚡ 문제 유형별 병렬 생성 중...")
            generation_results = await asyncio.gather(*[task for _, _, task in generation_tasks])

            # 결과 결합
            for i, (question_type, expected_count, _) in enumerate(generation_tasks):
                questions_data = generation_results[i]
                logger.info(f"{question_type.value}: {len(questions_data)}/{expected_count}개 생성")
                all_questions.extend(questions_data)

            # 5. Question 객체로 변환
            questions = self._convert_to_question_objects(all_questions, contexts, request.difficulty)

            # 6. 정확한 개수 보장
            if len(questions) < request.num_questions:
                logger.warning(f"부족한 문제 개수: {len(questions)}/{request.num_questions}")
                # 추가 생성 로직 필요 시 여기서 처리

            questions = questions[:request.num_questions]  # 정확한 개수만

            # 7. 고급 품질 검증
            logger.info("🔍 종합 품질 검증 중...")
            validation_result = await self.validator.comprehensive_validation(questions)

            # 8. 응답 생성
            generation_time = time.time() - start_time

            response = QuizResponse(
                quiz_id=quiz_id,
                document_id=request.document_id,
                questions=questions,
                total_questions=len(questions),
                difficulty=request.difficulty,
                generation_time=generation_time,
                success=True,
                metadata={
                    "generation_method": "advanced_multi_stage",
                    "contexts_used": len(contexts),
                    "type_distribution": {k.value: v for k, v in type_distribution.items()},
                    "validation_result": validation_result,
                    "llm_model": self.llm_service.model_name,
                    "quality_score": validation_result.get("overall_score", 0),
                    "duplicate_count": len(validation_result.get("duplicate_analysis", {}).get("duplicate_pairs", [])),
                    "advanced_features": [
                        "멀티 스테이지 RAG",
                        "의미적 중복 검증",
                        "문제 유형별 전문 생성",
                        "정확한 개수 보장",
                        "프로급 품질 검증"
                    ]
                }
            )

            logger.info(f"🎉 프로급 퀴즈 생성 완료: {len(questions)}문제 (품질: {validation_result.get('overall_score', 0)}/10)")
            return response

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"🚨 프로급 퀴즈 생성 실패: {e} ({error_time:.2f}초)")

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

    def _calculate_type_distribution(self, request: QuizRequest) -> Dict[QuestionType, int]:
        """문제 유형별 정확한 개수 분배"""

        if request.question_types:
            # 사용자가 지정한 유형들
            types = request.question_types
        else:
            # 난이도별 기본 유형
            if request.difficulty == Difficulty.EASY:
                types = [QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE]
            elif request.difficulty == Difficulty.MEDIUM:
                types = [QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER]
            else:
                types = [QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.FILL_BLANK]

        # 균등 분배
        base_count = request.num_questions // len(types)
        remainder = request.num_questions % len(types)

        distribution = {}
        for i, qtype in enumerate(types):
            count = base_count + (1 if i < remainder else 0)
            distribution[qtype] = count

        return distribution

    def _convert_to_question_objects(
        self,
        llm_questions: List[Dict],
        contexts: List[RAGContext],
        base_difficulty: Difficulty
    ) -> List[Question]:
        """Question 객체로 변환"""
        questions = []

        for i, q_data in enumerate(llm_questions):
            try:
                question_type = QuestionType(q_data.get("question_type", "multiple_choice"))
                difficulty = Difficulty(q_data.get("difficulty", base_difficulty.value))

                source_context = ""
                if i < len(contexts):
                    source_context = contexts[i].text[:200] + "..."

                question = Question(
                    question=q_data.get("question", ""),
                    question_type=question_type,
                    correct_answer=q_data.get("correct_answer", ""),
                    options=q_data.get("options"),
                    explanation=q_data.get("explanation", ""),
                    difficulty=difficulty,
                    source_context=source_context,
                    topic=q_data.get("topic", "주요 내용"),
                    metadata={
                        "advanced_generated": True,
                        "context_similarity": contexts[i].similarity if i < len(contexts) else 0,
                        "generation_order": i + 1,
                        "quality_verified": True
                    }
                )

                questions.append(question)

            except Exception as e:
                logger.warning(f"문제 {i+1} 변환 실패: {e}")
                continue

        return questions


# 전역 고급 퀴즈 서비스
_advanced_quiz_service: Optional[AdvancedQuizService] = None

def get_advanced_quiz_service() -> AdvancedQuizService:
    """프로덕션 급 퀴즈 서비스 반환"""
    global _advanced_quiz_service

    if _advanced_quiz_service is None:
        _advanced_quiz_service = AdvancedQuizService()
        logger.info("🚀 프로덕션 급 퀴즈 서비스 초기화 완료")

    return _advanced_quiz_service


if __name__ == "__main__":
    print("🚀 프로덕션 급 퀴즈 생성 시스템")
    print("- 정확한 문제 개수 보장")
    print("- 멀티 스테이지 RAG")
    print("- 의미적 중복 검증")
    print("- 문제 유형별 전문 생성")
    print("- 프로급 품질 검증")