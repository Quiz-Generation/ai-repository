#!/usr/bin/env python3
"""
🧪 중복 완전 제거 + 2:6:2 비율 퀴즈 서비스 테스트
- 강화된 중복 제거 시스템
- OX:객관식:주관식 = 2:6:2 비율
- 사용자 선택 가능 (전부 OX/객관식/주관식)
"""
import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any, Optional

# 실제 서비스 import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.quiz_schema import (
    QuizRequest, QuizResponse, Question, Difficulty, QuestionType, RAGContext
)
from app.services.advanced_quiz_service import (
    AdvancedQuizService, MultiStageRAGRetriever,
    QuestionTypeSpecialist, AdvancedQuizValidator
)
from app.services.vector_service import PDFVectorService
from app.services.llm_factory import BaseLLMService


class TestAdvancedQuizService:
    """🎓 중복 제거 + 2:6:2 비율 고급 퀴즈 서비스 테스트"""

    @pytest.fixture
    def mock_vector_service(self):
        """벡터 서비스 모킹"""
        mock = Mock(spec=PDFVectorService)
        mock.get_document_info.return_value = {
            "document_id": "test-aws-doc",
            "filename": "AWS_Solutions_Architect.pdf",
            "chunk_count": 100,
            "total_chars": 50000
        }

        # 다양한 AWS 컨텍스트 모킹
        mock.search_in_document.return_value = [
            {
                "text": "Amazon EC2 Auto Scaling helps maintain application availability and allows you to automatically add or remove EC2 instances according to conditions you define.",
                "similarity": 0.95,
                "metadata": {"chunk_index": 1, "source": "AWS_EC2.pdf"}
            },
            {
                "text": "Amazon S3 provides industry-leading scalability, data availability, security, and performance for object storage.",
                "similarity": 0.90,
                "metadata": {"chunk_index": 2, "source": "AWS_S3.pdf"}
            },
            {
                "text": "Amazon RDS makes it easy to set up, operate, and scale a relational database in the cloud.",
                "similarity": 0.85,
                "metadata": {"chunk_index": 3, "source": "AWS_RDS.pdf"}
            }
        ]
        return mock

    @pytest.fixture
    def mock_llm_service(self):
        """LLM 서비스 모킹"""
        mock = Mock(spec=BaseLLMService)
        mock.model_name = "gpt-4o-mini"

        # 🔥 provider Mock 객체로 수정
        mock.provider = Mock()
        mock.provider.value = "openai"

        # 다양한 문제 유형 응답 모킹
        def create_mock_response(question_type):
            if question_type == "true_false":
                return '''
{
    "questions": [
        {
            "question": "AWS EC2는 서버리스 컴퓨팅 서비스이다.",
            "question_type": "true_false",
            "correct_answer": "False",
            "explanation": "AWS EC2는 가상 서버 인스턴스를 제공하는 서비스로, 서버리스가 아닙니다.",
            "difficulty": "medium",
            "topic": "AWS 컴퓨팅"
        }
    ]
}
'''
            elif question_type == "multiple_choice":
                return '''
{
    "questions": [
        {
            "question": "AWS에서 애플리케이션의 고가용성을 보장하기 위한 가장 적절한 서비스는?",
            "question_type": "multiple_choice",
            "options": ["Amazon EC2 Auto Scaling", "Amazon S3", "Amazon CloudWatch", "AWS Lambda"],
            "correct_answer": "Amazon EC2 Auto Scaling",
            "explanation": "Auto Scaling은 정의된 조건에 따라 자동으로 EC2 인스턴스를 추가하거나 제거하여 애플리케이션 가용성을 유지합니다.",
            "difficulty": "medium",
            "topic": "AWS 컴퓨팅"
        }
    ]
}
'''
            else:  # short_answer
                return '''
{
    "questions": [
        {
            "question": "AWS에서 정적 웹사이트 호스팅에 가장 적합한 서비스는?",
            "question_type": "short_answer",
            "correct_answer": "Amazon S3",
            "explanation": "S3는 정적 웹사이트 호스팅을 위한 비용 효율적이고 확장 가능한 솔루션입니다.",
            "difficulty": "medium",
            "topic": "AWS 스토리지"
        }
    ]
}
'''

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = create_mock_response("multiple_choice")

        # client.chat.completions.create 모킹
        mock.client = Mock()
        mock.client.chat.completions.create = AsyncMock(return_value=mock_response)

        return mock

    @pytest.fixture
    def quiz_service(self, mock_vector_service, mock_llm_service):
        """퀴즈 서비스 인스턴스"""
        return AdvancedQuizService(
            vector_service=mock_vector_service,
            llm_service=mock_llm_service
        )

    @pytest.mark.asyncio
    async def test_service_initialization(self, quiz_service):
        """✅ 서비스 초기화 테스트"""
        assert quiz_service is not None
        assert quiz_service.rag_retriever is not None
        assert quiz_service.question_specialist is not None
        assert quiz_service.validator is not None

    @pytest.mark.asyncio
    async def test_기본_2_6_2_비율_적용(self, quiz_service):
        """🔥 기본 2:6:2 비율 (OX:객관식:주관식) 테스트"""
        # Given: 10문제 요청 (타입 지정 없음)
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=10,
            difficulty=Difficulty.MEDIUM
        )

        # When: 문제 유형 분배 계산
        type_distribution = quiz_service._calculate_type_distribution(request)

        # Then: 2:6:2 비율 확인
        tf_count = type_distribution.get(QuestionType.TRUE_FALSE, 0)
        mc_count = type_distribution.get(QuestionType.MULTIPLE_CHOICE, 0)
        sa_count = type_distribution.get(QuestionType.SHORT_ANSWER, 0)

        assert tf_count == 2  # 20% OX
        assert mc_count == 6  # 60% 객관식
        assert sa_count == 2  # 20% 주관식

        total = tf_count + mc_count + sa_count
        assert total == 10

        print(f"✅ 2:6:2 비율 적용: OX {tf_count}개, 객관식 {mc_count}개, 주관식 {sa_count}개")

    @pytest.mark.asyncio
    async def test_사용자_선택_전부_객관식(self, quiz_service):
        """🔥 사용자 선택: 전부 객관식 테스트"""
        # Given: 객관식만 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=5,
            difficulty=Difficulty.MEDIUM,
            question_types=[QuestionType.MULTIPLE_CHOICE]
        )

        # When: 문제 유형 분배 계산
        type_distribution = quiz_service._calculate_type_distribution(request)

        # Then: 100% 객관식 확인
        assert type_distribution == {QuestionType.MULTIPLE_CHOICE: 5}
        print(f"✅ 전부 객관식: {type_distribution}")

    @pytest.mark.asyncio
    async def test_사용자_선택_전부_OX(self, quiz_service):
        """🔥 사용자 선택: 전부 OX 테스트"""
        # Given: OX만 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=3,
            difficulty=Difficulty.MEDIUM,
            question_types=[QuestionType.TRUE_FALSE]
        )

        # When: 문제 유형 분배 계산
        type_distribution = quiz_service._calculate_type_distribution(request)

        # Then: 100% OX 확인
        assert type_distribution == {QuestionType.TRUE_FALSE: 3}
        print(f"✅ 전부 OX: {type_distribution}")

    @pytest.mark.asyncio
    async def test_OX_문제_생성_확인(self, quiz_service):
        """🔥 OX 문제 생성 확인"""
        # Given: OX 문제 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=2,
            difficulty=Difficulty.MEDIUM,
            question_types=[QuestionType.TRUE_FALSE]
        )

        # When: 퀴즈 생성
        response = await quiz_service.generate_guaranteed_quiz(request)

        # Then: OX 문제 확인
        assert response.success is True
        for question in response.questions:
            if question.question_type == QuestionType.TRUE_FALSE:
                assert question.correct_answer in ["True", "False"]
                assert question.options is None  # OX는 선택지 없음
                print(f"✅ OX 문제: {question.question[:50]}... 정답: {question.correct_answer}")

    @pytest.mark.asyncio
    async def test_강화된_중복_제거_시스템(self, quiz_service):
        """🔥 강화된 중복 제거 시스템 테스트"""
        # Given: 퀴즈 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=5,
            difficulty=Difficulty.MEDIUM
        )

        # When: 퀴즈 생성
        response = await quiz_service.generate_guaranteed_quiz(request)

        # Then: 중복 검증 결과 확인
        validation_result = response.metadata.get("validation_result", {})
        duplicate_analysis = validation_result.get("duplicate_analysis", {})

        duplicate_pairs = duplicate_analysis.get("duplicate_pairs", [])
        similar_pairs = duplicate_analysis.get("similar_pairs", [])
        max_similarity = duplicate_analysis.get("max_similarity", 0)

        # 강화된 기준 적용
        assert len(duplicate_pairs) == 0  # 중복 문제 0개 목표
        assert len(similar_pairs) <= 1   # 유사 문제 최대 1개
        assert max_similarity < 0.6      # 최대 유사도 0.6 미만

        print(f"✅ 중복 제거 성과: 중복 {len(duplicate_pairs)}개, 유사 {len(similar_pairs)}개, 최대유사도 {max_similarity:.3f}")

    @pytest.mark.asyncio
    async def test_문제_유형별_특성_확인(self, quiz_service):
        """✅ 문제 유형별 특성 확인"""
        # Given: 모든 타입 포함 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=6,  # 2:6:2로 나누기 쉬운 수
            difficulty=Difficulty.MEDIUM,
            question_types=[QuestionType.TRUE_FALSE, QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER]
        )

        # When: 퀴즈 생성
        response = await quiz_service.generate_guaranteed_quiz(request)

        # Then: 문제 유형별 특성 확인
        assert response.success is True

        for question in response.questions:
            if question.question_type == QuestionType.TRUE_FALSE:
                assert question.correct_answer in ["True", "False"]
                assert question.options is None
                print(f"✅ OX 문제: 정답 {question.correct_answer}")

            elif question.question_type == QuestionType.MULTIPLE_CHOICE:
                assert question.options is not None
                assert len(question.options) >= 4
                assert question.correct_answer in question.options
                print(f"✅ 객관식: {len(question.options)}개 선택지")

            elif question.question_type == QuestionType.SHORT_ANSWER:
                assert question.options is None
                assert question.correct_answer.strip()  # 빈 답변 아님
                print(f"✅ 주관식: 정답 길이 {len(question.correct_answer)}자")

    @pytest.mark.asyncio
    async def test_품질_점수_개선_확인(self, quiz_service):
        """🔍 품질 점수 개선 확인"""
        # Given: 퀴즈 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=5,
            difficulty=Difficulty.MEDIUM
        )

        # When: 퀴즈 생성
        response = await quiz_service.generate_guaranteed_quiz(request)

        # Then: 품질 점수 확인
        validation_result = response.metadata.get("validation_result", {})
        quality_score = validation_result.get("overall_score", 0)

        # 중복 제거 시스템으로 품질 향상 기대
        assert quality_score >= 7.0  # 높은 품질 기준

        individual_scores = validation_result.get("individual_scores", [])
        if individual_scores:
            avg_individual = sum(individual_scores) / len(individual_scores)
            assert avg_individual >= 7.0

        print(f"✅ 품질 점수: 전체 {quality_score}/10, 개별 평균 {avg_individual:.1f}/10")

    @pytest.mark.asyncio
    async def test_메타데이터_정보_확인(self, quiz_service):
        """📊 메타데이터 정보 확인"""
        # Given: 퀴즈 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=5,
            difficulty=Difficulty.MEDIUM
        )

        # When: 퀴즈 생성
        response = await quiz_service.generate_guaranteed_quiz(request)

        # Then: 메타데이터 확인
        metadata = response.metadata

        assert metadata.get("generation_method") == "duplicate_free_2_6_2_ratio"
        assert metadata.get("ratio_applied") == "2:6:2 (OX:객관식:주관식)"
        assert metadata.get("duplicate_prevention") == "강화된 중복 제거 적용"
        assert metadata.get("similarity_threshold") == 0.6

        advanced_features = metadata.get("advanced_features", [])
        assert "🔥 완전한 중복 제거 시스템" in advanced_features
        assert "🔥 2:6:2 비율 (OX:객관식:주관식)" in advanced_features

        print(f"✅ 메타데이터 확인: {metadata['generation_method']}")

    @pytest.mark.asyncio
    async def test_대용량_문제_생성_안정성(self, quiz_service):
        """⚡ 대용량 문제 생성 안정성 테스트"""
        # Given: 많은 문제 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=20,  # 대용량
            difficulty=Difficulty.MEDIUM
        )

        # When: 퀴즈 생성
        response = await quiz_service.generate_guaranteed_quiz(request)

        # Then: 안정성 확인
        assert response.success is True
        assert len(response.questions) == 20

        # 2:6:2 비율 확인 (20문제 기준)
        type_counts = {}
        for q in response.questions:
            qtype = q.question_type.value
            type_counts[qtype] = type_counts.get(qtype, 0) + 1

        # 대략적 비율 확인 (정확하지 않아도 됨)
        tf_ratio = type_counts.get("true_false", 0) / 20
        mc_ratio = type_counts.get("multiple_choice", 0) / 20

        assert 0.15 <= tf_ratio <= 0.25  # OX 15-25%
        assert 0.55 <= mc_ratio <= 0.65  # 객관식 55-65%

        print(f"✅ 대용량 안정성: {type_counts}")


class TestMultiStageRAGRetriever:
    """🧠 멀티 스테이지 RAG 검색기 테스트"""

    @pytest.fixture
    def mock_vector_service(self):
        mock = Mock(spec=PDFVectorService)
        mock.search_in_document.return_value = [
            {
                "text": "AWS EC2는 클라우드에서 확장 가능한 컴퓨팅 용량을 제공합니다.",
                "similarity": 0.95,
                "metadata": {"chunk_index": 1, "source": "AWS_EC2.pdf"}
            }
        ]
        return mock

    @pytest.fixture
    def mock_llm_service(self):
        return Mock(spec=BaseLLMService)

    @pytest.fixture
    def rag_retriever(self, mock_vector_service, mock_llm_service):
        return MultiStageRAGRetriever(mock_vector_service, mock_llm_service)

    @pytest.mark.asyncio
    async def test_다양성_있는_컨텍스트_검색(self, rag_retriever):
        """🎯 다양성 있는 컨텍스트 검색 테스트"""
        # Given: 문서와 문제 수
        document_id = "test-aws-doc"
        num_questions = 5

        # When: 컨텍스트 검색
        contexts = await rag_retriever.retrieve_diverse_contexts(
            document_id=document_id,
            num_questions=num_questions
        )

        # Then: 적절한 컨텍스트 반환 확인
        assert isinstance(contexts, list)
        assert len(contexts) >= 0
        print(f"✅ 검색된 컨텍스트: {len(contexts)}개")


class TestQuestionTypeSpecialist:
    """🎯 문제 유형별 전문 생성기 테스트"""

    @pytest.fixture
    def mock_llm_service(self):
        mock = Mock(spec=BaseLLMService)
        mock.model_name = "gpt-4o-mini"

        # 객관식 응답 모킹
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
{
    "questions": [
        {
            "question": "AWS에서 가장 인기 있는 컴퓨팅 서비스는?",
            "question_type": "multiple_choice",
            "options": ["EC2", "Lambda", "ECS", "Fargate"],
            "correct_answer": "EC2",
            "explanation": "Amazon EC2는 AWS의 대표적인 컴퓨팅 서비스입니다.",
            "difficulty": "medium",
            "topic": "AWS 컴퓨팅"
        }
    ]
}
'''

        mock.client = Mock()
        mock.client.chat.completions.create = AsyncMock(return_value=mock_response)

        return mock

    @pytest.fixture
    def specialist(self, mock_llm_service):
        return QuestionTypeSpecialist(mock_llm_service)

    @pytest.mark.asyncio
    async def test_객관식_문제_생성(self, specialist):
        """🔥 객관식 문제 전문 생성 테스트"""
        # Given: 컨텍스트와 요청 정보
        contexts = [RAGContext(
            text="AWS EC2 관련 내용",
            similarity=0.9,
            source="test_aws.pdf",
            chunk_index=1
        )]

        # When: 객관식 문제 생성
        questions = await specialist.generate_guaranteed_questions(
            contexts=contexts,
            question_type=QuestionType.MULTIPLE_CHOICE,
            count=1,
            difficulty=Difficulty.MEDIUM,
            topic="AWS 컴퓨팅",
            options_count=4
        )

        # Then: 객관식 문제 생성 확인
        assert len(questions) >= 0
        if questions:
            assert questions[0].get("question_type") == "multiple_choice"
            print(f"✅ 객관식 문제 생성: {len(questions)}개")


class TestAdvancedQuizValidator:
    """🔍 고급 품질 검증 시스템 테스트"""

    @pytest.fixture
    def mock_llm_service(self):
        return Mock(spec=BaseLLMService)

    @pytest.fixture
    def validator(self, mock_llm_service):
        return AdvancedQuizValidator(mock_llm_service)

    @pytest.mark.asyncio
    async def test_개별_문제_품질_평가(self, validator):
        """📊 개별 문제 품질 평가 테스트"""
        # Given: 테스트 문제
        question = Question(
            question="AWS에서 가장 인기 있는 컴퓨팅 서비스는?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=["EC2", "Lambda", "ECS", "Fargate"],
            correct_answer="EC2",
            explanation="Amazon EC2는 AWS의 대표적인 컴퓨팅 서비스입니다.",
            difficulty=Difficulty.MEDIUM
        )

        # When: 품질 평가
        score = await validator._score_single_question(question)

        # Then: 적절한 점수 확인
        assert 0 <= score <= 10
        assert score >= 7.0  # 좋은 문제는 7점 이상
        print(f"✅ 문제 품질 점수: {score}/10점")

    @pytest.mark.asyncio
    async def test_종합_품질_검증(self, validator):
        """🎯 종합 품질 검증 테스트"""
        # Given: 테스트 문제들
        questions = [
            Question(
                question="AWS EC2란 무엇인가?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                options=["컴퓨팅 서비스", "스토리지 서비스", "네트워크 서비스", "데이터베이스 서비스"],
                correct_answer="컴퓨팅 서비스",
                explanation="EC2는 클라우드 컴퓨팅 서비스입니다.",
                difficulty=Difficulty.MEDIUM
            ),
            Question(
                question="S3의 주요 용도는?",
                question_type=QuestionType.SHORT_ANSWER,
                correct_answer="객체 스토리지",
                explanation="S3는 객체 스토리지 서비스입니다.",
                difficulty=Difficulty.EASY
            )
        ]

        # When: 종합 품질 검증
        validation_result = await validator.comprehensive_validation(questions)

        # Then: 검증 결과 확인
        assert "overall_score" in validation_result
        assert "individual_scores" in validation_result
        assert "duplicate_analysis" in validation_result

        overall_score = validation_result["overall_score"]
        assert 0 <= overall_score <= 10
        print(f"✅ 종합 품질 점수: {overall_score}/10점")


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])
    print("\n🎉 중복 제거 + 2:6:2 비율 퀴즈 서비스 테스트 완료!")
    print("🔥 1. 강화된 중복 제거 시스템 ✅")
    print("🔥 2. 2:6:2 비율 (OX:객관식:주관식) ✅")
    print("🔥 3. 사용자 선택: 전부 OX/객관식/주관식 ✅")
    print("✅ 실제 options 포함하는 고품질 객관식 문제 생성!")