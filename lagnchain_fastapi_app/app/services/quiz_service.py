"""
PDF 기반 RAG 퀴즈 생성 서비스
최적화된 퀴즈 생성을 위한 메인 서비스
"""
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Union
from dataclasses import asdict

from ..schemas.quiz_schema import (
    QuizRequest, QuizResponse, Question, Difficulty, QuestionType,
    RAGContext, TopicAnalysis, QuizGenerationStats
)
from ..services.llm_factory import LLMFactory, BaseLLMService, get_default_llm_service
from ..services.vector_service import PDFVectorService, get_global_vector_service

logger = logging.getLogger(__name__)


class RAGRetriever:
    """RAG 컨텍스트 검색 클래스"""

    def __init__(self, vector_service: PDFVectorService, llm_service: Optional[BaseLLMService] = None):
        self.vector_service = vector_service
        self.llm_service = llm_service

    def retrieve_contexts_for_quiz(
        self,
        document_id: str,
        num_questions: int,
        topics: Optional[List[str]] = None
    ) -> List[RAGContext]:
        """퀴즈 생성을 위한 최적 컨텍스트 검색"""

        logger.info(f"RAG 컨텍스트 검색 시작: {document_id} ({num_questions}문제)")

        # 문서 정보 확인
        doc_info = self.vector_service.get_document_info(document_id)
        if not doc_info:
            raise ValueError(f"문서를 찾을 수 없습니다: {document_id}")

        contexts = []

        # 주제별 검색 또는 동적 검색
        if topics:
            # 특정 주제들에 대한 검색
            for topic in topics:
                search_results = self.vector_service.search_in_document(
                    query=topic,
                    document_id=document_id,
                    top_k=max(3, num_questions // len(topics))
                )
                contexts.extend(self._convert_to_rag_contexts(search_results, topic))
        else:
            # 🧠 LLM 기반 동적 키워드 생성
            logger.info("토픽이 없음 → LLM으로 문서 맞춤 검색 키워드 생성 중...")
            dynamic_queries = self._generate_dynamic_search_queries(document_id, num_questions)

            logger.info(f"생성된 동적 검색 키워드: {dynamic_queries}")

            for query in dynamic_queries:
                search_results = self.vector_service.search_in_document(
                    query=query,
                    document_id=document_id,
                    top_k=2
                )
                contexts.extend(self._convert_to_rag_contexts(search_results))

        # 중복 제거 및 품질 필터링
        contexts = self._deduplicate_contexts(contexts)
        contexts = self._filter_context_quality(contexts)

        # 유사도 기준 정렬
        contexts.sort(key=lambda x: x.similarity, reverse=True)

        logger.info(f"RAG 컨텍스트 검색 완료: {len(contexts)}개")
        return contexts[:num_questions * 2]  # 여유분 확보

    def _generate_dynamic_search_queries(self, document_id: str, num_questions: int) -> List[str]:
        """📚 LLM을 활용하여 문서에 맞는 동적 검색 키워드 생성"""

        if not self.llm_service:
            # LLM이 없으면 기본 범용 키워드 사용 (fallback)
            logger.warning("LLM 서비스가 없어 기본 키워드 사용")
            return ["핵심 내용", "주요 개념", "중요한 정보", "기본 원리", "주된 내용"]

        try:
            # 문서의 샘플 텍스트 수집 (문서 전체 개요 파악용)
            sample_contexts = self.vector_service.search_in_document(
                query="주요 내용 핵심 정보",
                document_id=document_id,
                top_k=3
            )

            if not sample_contexts:
                logger.warning("문서에서 샘플 컨텍스트를 찾을 수 없음")
                return ["주요 내용", "핵심 개념"]

            # 샘플 텍스트 결합
            sample_text = "\n".join([ctx["text"][:500] for ctx in sample_contexts])

            # LLM으로 문서 맞춤 검색 키워드 생성
            prompt = f"""
다음은 어떤 문서의 일부 내용입니다. 이 문서에서 퀴즈 생성을 위한 최적의 검색 키워드를 {num_questions//2 + 3}개 생성해주세요.

문서 내용:
{sample_text[:2000]}

요구사항:
1. 이 문서의 주제와 분야에 맞는 구체적인 키워드
2. 퀴즈로 만들기 좋은 핵심 개념들
3. 너무 일반적이지 않고, 이 문서에 특화된 용어들
4. 단순히 단어가 아닌 짧은 구문도 가능

JSON 형식으로 응답해주세요:
{{
    "search_keywords": ["키워드1", "키워드2", "키워드3", ...]
}}
"""

            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model_name,
                messages=[
                    {"role": "system", "content": "문서 분석 전문가로서 퀴즈 생성에 최적화된 검색 키워드를 추출하는 역할입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            result_text = response.choices[0].message.content
            if result_text is None:
                raise ValueError("LLM 응답이 비어있습니다")

            # JSON 파싱
            import json
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("JSON 형식을 찾을 수 없습니다")

            json_text = result_text[start_idx:end_idx]
            result = json.loads(json_text)

            keywords = result.get("search_keywords", [])

            if not keywords:
                raise ValueError("검색 키워드가 생성되지 않았습니다")

            logger.info(f"LLM이 생성한 동적 키워드: {keywords}")
            return keywords[:num_questions//2 + 3]  # 적절한 개수로 제한

        except Exception as e:
            logger.error(f"동적 검색 키워드 생성 실패: {e}")
            # 실패 시 기본 범용 키워드 반환
            return ["핵심 내용", "주요 개념", "중요한 정보", "기본 원리", "주된 주제"]

    def _convert_to_rag_contexts(
        self,
        search_results: List[Dict],
        topic: Optional[str] = None
    ) -> List[RAGContext]:
        """검색 결과를 RAGContext로 변환"""
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

    def _deduplicate_contexts(self, contexts: List[RAGContext]) -> List[RAGContext]:
        """중복 컨텍스트 제거"""
        seen_texts = set()
        unique_contexts = []

        for context in contexts:
            # 텍스트의 첫 100자로 중복 체크
            text_signature = context.text[:100].strip()
            if text_signature not in seen_texts:
                seen_texts.add(text_signature)
                unique_contexts.append(context)

        return unique_contexts

    def _filter_context_quality(self, contexts: List[RAGContext]) -> List[RAGContext]:
        """컨텍스트 품질 필터링"""
        return [
            ctx for ctx in contexts
            if len(ctx.text.strip()) >= 50 and ctx.similarity >= 0.05
        ]


class TopicExtractor:
    """주제 추출 클래스"""

    def __init__(self, llm_service: BaseLLMService, vector_service: PDFVectorService):
        self.llm_service = llm_service
        self.vector_service = vector_service

    def extract_document_topics(self, document_id: str) -> List[TopicAnalysis]:
        """📚 문서에서 주요 토픽 추출 및 분석 (개선된 버전)"""

        logger.info(f"문서 토픽 추출 시작: {document_id}")

        # 문서의 더 많은 샘플 텍스트 수집 (전체적인 이해를 위해)
        sample_contexts = self.vector_service.search_in_document(
            query="주요 내용 핵심 개념 중요한 정보",
            document_id=document_id,
            top_k=8  # 더 많은 샘플 수집
        )

        if not sample_contexts:
            logger.warning(f"문서 {document_id}에서 샘플 컨텍스트를 찾을 수 없음")
            return []

        # 더 큰 텍스트 샘플 결합 (문서 전체 파악)
        combined_text = "\n".join([ctx["text"] for ctx in sample_contexts])

        # 🧠 개선된 LLM 토픽 추출 프롬프트
        enhanced_prompt = f"""
다음은 특정 문서의 주요 내용들입니다. 이 문서의 핵심 주제들을 분석하여 퀴즈 생성에 적합한 토픽들을 추출해주세요.

문서 내용:
{combined_text[:4000]}

분석 요구사항:
1. 이 문서의 주요 분야/도메인 식별
2. 퀴즈로 만들기 좋은 구체적인 주제들 추출
3. 각 토픽의 중요도와 난이도 평가
4. 문서에 실제로 나타나는 개념들만 포함

JSON 형식으로 응답해주세요:
{{
    "document_domain": "문서의 주요 분야 (예: 컴퓨터과학, 의학, 역사, 문학 등)",
    "main_topics": [
        {{
            "topic": "구체적인 주제명",
            "importance": 1-10,
            "quiz_potential": 1-10,
            "keywords": ["관련", "키워드", "목록"],
            "description": "이 토픽에 대한 간단한 설명"
        }}
    ]
}}
"""

        try:
            response = self.llm_service.client.chat.completions.create(
                model=self.llm_service.model_name,
                messages=[
                    {"role": "system", "content": "문서 분석 및 토픽 추출 전문가입니다. 주어진 문서에서 퀴즈 생성에 최적화된 주제들을 정확히 식별합니다."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.2,  # 더 일관된 결과를 위해 낮은 온도
                max_tokens=1000
            )

            result_text = response.choices[0].message.content
            if result_text is None:
                raise ValueError("LLM 토픽 추출 응답이 비어있습니다")

            # JSON 파싱
            import json
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("JSON 형식을 찾을 수 없습니다")

            json_text = result_text[start_idx:end_idx]
            result = json.loads(json_text)

            # 결과 파싱 및 TopicAnalysis 객체 생성
            topic_analyses = []
            main_topics = result.get("main_topics", [])
            document_domain = result.get("document_domain", "일반")

            logger.info(f"문서 도메인 식별: {document_domain}")

            for topic_data in main_topics:
                # 각 토픽별 실제 문서 검색으로 검증
                topic_name = topic_data.get("topic", "")
                if not topic_name:
                    continue

                analysis = self._analyze_topic_enhanced(document_id, topic_name, topic_data)
                if analysis.confidence > 0.1:  # 최소 신뢰도 필터
                    topic_analyses.append(analysis)

            # 중요도와 퀴즈 가능성 기준으로 정렬
            topic_analyses.sort(key=lambda x: (x.question_potential, x.confidence), reverse=True)

            logger.info(f"토픽 추출 완료: {len(topic_analyses)}개 (도메인: {document_domain})")
            return topic_analyses[:12]  # 최대 12개 토픽

        except Exception as e:
            logger.error(f"개선된 토픽 추출 실패: {e}")
            # 실패 시 기존 방식으로 fallback
            return self._fallback_topic_extraction(combined_text)

    def _analyze_topic_enhanced(self, document_id: str, topic: str, topic_data: Dict) -> TopicAnalysis:
        """개선된 개별 토픽 분석"""

        # 토픽 관련 컨텍스트 검색 (더 정확한 검색)
        search_results = self.vector_service.search_in_document(
            query=topic,
            document_id=document_id,
            top_k=4
        )

        if not search_results:
            return TopicAnalysis(
                topic=topic,
                confidence=0.1,
                keywords=topic_data.get("keywords", []),
                context_chunks=[],
                question_potential=1
            )

        # 평균 유사도로 신뢰도 계산
        avg_similarity = sum(r["similarity"] for r in search_results) / len(search_results)

        # LLM에서 제공한 메타데이터 활용
        importance = topic_data.get("importance", 5)
        quiz_potential_base = topic_data.get("quiz_potential", 5)

        # 실제 검색 결과와 LLM 평가 조합
        final_quiz_potential = min(10, int(
            (quiz_potential_base * 0.7) + (avg_similarity * 10 * 0.3)
        ))

        return TopicAnalysis(
            topic=topic,
            confidence=avg_similarity,
            keywords=topic_data.get("keywords", []),
            context_chunks=[r["text"][:300] for r in search_results],
            question_potential=final_quiz_potential
        )

    def _fallback_topic_extraction(self, text: str) -> List[TopicAnalysis]:
        """LLM 실패 시 기본 토픽 추출 방식"""
        logger.info("기본 토픽 추출 방식으로 fallback")

        # 기존 간단한 방식
        topics = self.llm_service.extract_topics(text)

        topic_analyses = []
        for topic in topics:
            analysis = TopicAnalysis(
                topic=topic,
                confidence=0.5,
                keywords=[],
                context_chunks=[],
                question_potential=5
            )
            topic_analyses.append(analysis)

        return topic_analyses


class QuizValidator:
    """퀴즈 품질 검증 클래스"""

    def __init__(self, llm_service: BaseLLMService):
        self.llm_service = llm_service

    def validate_quiz_quality(self, questions: List[Question]) -> Dict[str, Any]:
        """퀴즈 전체 품질 검증"""

        validation_result = {
            "overall_quality": "good",
            "total_questions": len(questions),
            "valid_questions": 0,
            "issues": [],
            "recommendations": []
        }

        valid_count = 0

        for i, question in enumerate(questions):
            question_dict = asdict(question)
            if self.llm_service.validate_question_quality(question_dict):
                valid_count += 1
            else:
                validation_result["issues"].append(f"문제 {i+1}: 품질 기준 미달")

        validation_result["valid_questions"] = valid_count

        # 전체 품질 평가
        quality_ratio = valid_count / len(questions) if questions else 0

        if quality_ratio >= 0.8:
            validation_result["overall_quality"] = "excellent"
        elif quality_ratio >= 0.6:
            validation_result["overall_quality"] = "good"
        elif quality_ratio >= 0.4:
            validation_result["overall_quality"] = "fair"
            validation_result["recommendations"].append("문제 품질 개선 필요")
        else:
            validation_result["overall_quality"] = "poor"
            validation_result["recommendations"].append("문제 재생성 권장")

        return validation_result


class QuizService:
    """PDF 기반 퀴즈 생성 메인 서비스"""

    def __init__(
        self,
        vector_service: Optional[PDFVectorService] = None,
        llm_service: Optional[BaseLLMService] = None
    ):
        """퀴즈 서비스 초기화"""

        # 벡터 서비스 (PDF 서비스와 동일한 싱글톤 인스턴스 공유)
        if vector_service is None:
            self.vector_service = get_global_vector_service()
        else:
            self.vector_service = vector_service

        # LLM 서비스 (기본: OpenAI GPT-4o-mini)
        self.llm_service = llm_service or get_default_llm_service()

        # 하위 컴포넌트들
        self.rag_retriever = RAGRetriever(self.vector_service, self.llm_service)
        self.topic_extractor = TopicExtractor(self.llm_service, self.vector_service)
        self.quiz_validator = QuizValidator(self.llm_service)

        logger.info(f"퀴즈 서비스 초기화 완료: LLM={self.llm_service.model_name}, VectorDB={self.vector_service.db_type}")

    def generate_quiz(self, request: QuizRequest) -> QuizResponse:
        """메인 퀴즈 생성 메서드 - 토픽은 항상 자동 추출"""

        start_time = time.time()
        quiz_id = str(uuid.uuid4())

        logger.info(f"퀴즈 생성 시작: {request.document_id} ({request.num_questions}문제)")

        try:
            # 1단계: 문서 존재 확인
            doc_info = self.vector_service.get_document_info(request.document_id)
            if not doc_info:
                raise ValueError(f"문서를 찾을 수 없습니다: {request.document_id}")

            logger.info(f"문서 확인 완료: {doc_info['source_filename']} ({doc_info['chunk_count']}개 청크)")

            # 2단계: 토픽 자동 추출 (완전 자동화)
            logger.info("STEP1: 문서 토픽 자동 추출 중...")
            topic_analyses = self.topic_extractor.extract_document_topics(request.document_id)
            extracted_topics = [ta.topic for ta in topic_analyses[:7]]  # 상위 7개 토픽

            logger.info(f"자동 추출된 토픽: {extracted_topics}")

            # 3단계: RAG 컨텍스트 검색
            logger.info("STEP2: RAG 컨텍스트 검색 중...")
            contexts = self.rag_retriever.retrieve_contexts_for_quiz(
                document_id=request.document_id,
                num_questions=request.num_questions,
                topics=extracted_topics
            )

            if not contexts:
                raise ValueError("퀴즈 생성을 위한 적절한 컨텍스트를 찾을 수 없습니다")

            # 4단계: 컨텍스트 결합
            combined_context = self._combine_contexts(contexts)
            logger.info(f"결합된 컨텍스트 길이: {len(combined_context)}자")

            # 5단계: 문제 유형 결정
            question_types = self._determine_question_types(request)

            # 6단계: LLM으로 퀴즈 생성
            logger.info("STEP3: LLM 퀴즈 생성 중...")
            llm_result = self.llm_service.generate_quiz(
                context=combined_context,
                num_questions=request.num_questions,
                difficulty=request.difficulty.value,
                question_types=[qt.value for qt in question_types],
                topics=extracted_topics
            )

            if not llm_result.get("success", False):
                raise ValueError(f"LLM 퀴즈 생성 실패: {llm_result.get('error', '알 수 없는 오류')}")

            # 7단계: 응답 데이터를 Question 객체로 변환
            logger.info("STEP4: 문제 데이터 변환 중...")
            questions = self._convert_to_question_objects(
                llm_result["questions"],
                contexts,
                request.difficulty  # base_difficulty로 전달
            )

            # 8단계: 품질 검증
            logger.info("STEP5: 문제 품질 검증 중...")
            validation_result = self.quiz_validator.validate_quiz_quality(questions)

            # 9단계: 응답 생성
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
                    "extracted_topics": extracted_topics,
                    "user_hint_topics": [],
                    "contexts_used": len(contexts),
                    "avg_context_similarity": sum(c.similarity for c in contexts) / len(contexts),
                    "validation_result": validation_result,
                    "llm_model": self.llm_service.model_name,
                    "document_info": doc_info,
                    "generation_stats": {
                        "context_retrieval_count": len(contexts),
                        "topic_extraction_count": len(extracted_topics),
                        "question_types_used": [qt.value for qt in question_types]
                    }
                }
            )

            logger.info(f"퀴즈 생성 완료: {len(questions)}문제 ({generation_time:.2f}초)")
            return response

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"퀴즈 생성 실패: {str(e)} ({error_time:.2f}초)")

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

    def extract_topics(self, document_id: str) -> List[str]:
        """문서 토픽 추출 (외부 API용)"""
        try:
            topic_analyses = self.topic_extractor.extract_document_topics(document_id)
            return [ta.topic for ta in topic_analyses]
        except Exception as e:
            logger.error(f"토픽 추출 실패: {e}")
            return []

    def validate_question_quality(self, question: Question) -> bool:
        """개별 문제 품질 검증 (외부 API용)"""
        question_dict = asdict(question)
        return self.llm_service.validate_question_quality(question_dict)

    def retrieve_topic_contexts(self, document_id: str, topic: str) -> List[Dict]:
        """특정 토픽의 컨텍스트 검색 (외부 API용)"""
        try:
            contexts = self.rag_retriever.retrieve_contexts_for_quiz(
                document_id=document_id,
                num_questions=5,  # 기본값
                topics=[topic]
            )
            return [asdict(ctx) for ctx in contexts]
        except Exception as e:
            logger.error(f"토픽 컨텍스트 검색 실패: {e}")
            return []

    def switch_llm_model(self, llm_service: BaseLLMService):
        """LLM 모델 교체"""
        old_model = self.llm_service.model_name
        self.llm_service = llm_service

        # 하위 컴포넌트들도 업데이트
        self.topic_extractor.llm_service = llm_service
        self.quiz_validator.llm_service = llm_service

        logger.info(f"LLM 모델 교체: {old_model} → {llm_service.model_name}")

    def _combine_contexts(self, contexts: List[RAGContext]) -> str:
        """여러 컨텍스트를 결합"""
        combined = []

        for i, context in enumerate(contexts):
            section = f"[섹션 {i+1}]\n{context.text}\n"
            combined.append(section)

        return "\n".join(combined)

    def _determine_question_types(self, request: QuizRequest) -> List[QuestionType]:
        """문제 유형 결정"""
        if request.question_types:
            return request.question_types

        # 기본 문제 유형 조합 (난이도별)
        if request.difficulty == Difficulty.EASY:
            return [QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE]
        elif request.difficulty == Difficulty.MEDIUM:
            return [QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.FILL_BLANK]
        else:  # HARD
            return [QuestionType.SHORT_ANSWER, QuestionType.MULTIPLE_CHOICE, QuestionType.FILL_BLANK]

    def _convert_to_question_objects(
        self,
        llm_questions: List[Dict],
        contexts: List[RAGContext],
        base_difficulty: Difficulty
    ) -> List[Question]:
        """LLM 응답을 Question 객체로 변환 (문제별 난이도 다양화)"""
        questions = []

        for i, q_data in enumerate(llm_questions):
            try:
                # 문제 유형 변환
                question_type = QuestionType(q_data.get("question_type", "multiple_choice"))

                # 📊 문제별 난이도 자동 할당 (다양화)
                if len(llm_questions) >= 3:
                    # 3문제 이상이면 난이도 분산
                    if i % 3 == 0:
                        difficulty = Difficulty.EASY
                    elif i % 3 == 1:
                        difficulty = base_difficulty  # 기본 난이도 유지
                    else:
                        difficulty = Difficulty.HARD
                else:
                    # 3문제 미만이면 기본 난이도 사용
                    difficulty = base_difficulty

                # 소스 컨텍스트 찾기
                source_context = ""
                if i < len(contexts):
                    source_context = contexts[i].text[:200] + "..."

                question = Question(
                    question=q_data.get("question", ""),
                    question_type=question_type,
                    correct_answer=q_data.get("correct_answer", ""),
                    options=q_data.get("options"),
                    explanation=q_data.get("explanation", ""),
                    difficulty=difficulty,  # 개별 문제 난이도
                    source_context=source_context,
                    topic=q_data.get("topic", "일반"),
                    metadata={
                        "llm_generated": True,
                        "context_similarity": contexts[i].similarity if i < len(contexts) else 0,
                        "generation_order": i + 1,
                        "assigned_difficulty": difficulty.value  # 할당된 난이도 추가
                    }
                )

                questions.append(question)

            except Exception as e:
                logger.warning(f"문제 {i+1} 변환 실패: {e}")
                continue

        return questions


# 전역 퀴즈 서비스 인스턴스 (싱글톤)
_default_quiz_service: Optional[QuizService] = None


def get_default_quiz_service() -> QuizService:
    """기본 퀴즈 서비스 반환"""
    global _default_quiz_service

    if _default_quiz_service is None:
        _default_quiz_service = QuizService()
        logger.info("기본 퀴즈 서비스 초기화 완료")

    return _default_quiz_service


if __name__ == "__main__":
    # 간단한 테스트
    print("=== 퀴즈 서비스 테스트 ===")

    try:
        quiz_service = QuizService()
        print(f"퀴즈 서비스 초기화 성공: LLM={quiz_service.llm_service.model_name}")
    except Exception as e:
        print(f"퀴즈 서비스 초기화 실패: {e}")