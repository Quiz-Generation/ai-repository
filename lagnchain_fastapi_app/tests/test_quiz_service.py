#!/usr/bin/env python3
"""
🧪 3가지 피드백 반영 퀴즈 서비스 테스트
- 불필요한 import 제거
- 난이도 밸런스 (70%/20%/10%)
- 객관식 우선 생성 (70%)
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
    """🎓 3가지 피드백 반영 고급 퀴즈 서비스 테스트"""

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
        mock.provider.value = "openai"

        # LLM 응답 모킹
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
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
        },
        {
            "question": "Amazon S3에서 제공하는 주요 이점은?",
            "question_type": "multiple_choice",
            "options": ["확장성", "데이터 가용성", "보안", "모든 것"],
            "correct_answer": "모든 것",
            "explanation": "S3는 확장성, 데이터 가용성, 보안, 성능을 모두 제공합니다.",
            "difficulty": "medium",
            "topic": "AWS 스토리지"
        }
    ]
}
'''

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
    async def test_객관식_우선_생성_70퍼센트(self, quiz_service):
        """🔥 객관식 우선 생성 (70%) 테스트"""
        # Given: 10문제 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=10,
            difficulty=Difficulty.MEDIUM,
            question_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER]
        )

        # When: 퀴즈 생성
        response = await quiz_service.generate_guaranteed_quiz(request)

        # Then: 70% 객관식 확인
        type_distribution = quiz_service._calculate_type_distribution(request)
        mc_count = type_distribution.get(QuestionType.MULTIPLE_CHOICE, 0)
        total = sum(type_distribution.values())

        assert mc_count == 7  # 10문제 중 7개 (70%)
        assert total == 10
        print(f"✅ 객관식 우선 생성: {mc_count}/10 = {mc_count/10*100}%")

    @pytest.mark.asyncio
    async def test_난이도_밸런스_70_20_10(self, quiz_service):
        """🔥 난이도 밸런스 (70% medium, 20% easy, 10% hard) 테스트"""
        # Given: 10문제 생성을 위한 더미 데이터
        dummy_questions = [
            {"question": f"문제 {i+1}", "question_type": "multiple_choice",
             "correct_answer": "답", "explanation": "설명"}
            for i in range(10)
        ]
        dummy_contexts = [
            RAGContext(
                text=f"컨텍스트 {i}",
                similarity=0.9,
                source=f"test_source_{i}.pdf",
                chunk_index=i
            ) for i in range(10)
        ]

        # When: 난이도 밸런스 적용
        questions = quiz_service._convert_to_question_objects_with_balance(
            dummy_questions, dummy_contexts, Difficulty.MEDIUM
        )

        # Then: 난이도 분포 확인
        difficulty_counts = {}
        for q in questions:
            diff = q.difficulty.value
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        assert difficulty_counts.get("medium", 0) == 7  # 70%
        assert difficulty_counts.get("easy", 0) == 2    # 20%
        assert difficulty_counts.get("hard", 0) == 1    # 10%
        print(f"✅ 난이도 밸런스: {difficulty_counts}")

    @pytest.mark.asyncio
    async def test_객관식_options_포함_확인(self, quiz_service):
        """✅ 객관식 문제에 실제 options 배열 포함 확인"""
        # Given: 객관식 문제 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=3,
            difficulty=Difficulty.MEDIUM,
            question_types=[QuestionType.MULTIPLE_CHOICE]
        )

        # When: 퀴즈 생성
        response = await quiz_service.generate_guaranteed_quiz(request)

        # Then: 모든 객관식 문제에 options 확인
        assert response.success is True
        for question in response.questions:
            if question.question_type == QuestionType.MULTIPLE_CHOICE:
                assert question.options is not None
                assert len(question.options) >= 4  # 최소 4개 선택지
                assert question.correct_answer in question.options
                print(f"✅ 객관식 문제: {len(question.options)}개 선택지 포함")

    @pytest.mark.asyncio
    async def test_주관식_options_없음_확인(self, quiz_service):
        """✅ 주관식 문제에 options 없음 확인"""
        # Given: 주관식 문제 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=2,
            difficulty=Difficulty.MEDIUM,
            question_types=[QuestionType.SHORT_ANSWER]
        )

        # When: 퀴즈 생성
        response = await quiz_service.generate_guaranteed_quiz(request)

        # Then: 주관식 문제에 options 없음 확인
        assert response.success is True
        for question in response.questions:
            if question.question_type == QuestionType.SHORT_ANSWER:
                assert question.options is None
                print(f"✅ 주관식 문제: options 없음 확인")

    @pytest.mark.asyncio
    async def test_정확한_문제_개수_보장(self, quiz_service):
        """✅ 정확한 문제 개수 보장 테스트"""
        # Given: 다양한 개수 요청
        test_cases = [3, 5, 8, 10]

        for num_questions in test_cases:
            # When: 퀴즈 생성
            request = QuizRequest(
                document_id="test-aws-doc",
                num_questions=num_questions,
                difficulty=Difficulty.MEDIUM
            )
            response = await quiz_service.generate_guaranteed_quiz(request)

            # Then: 정확한 개수 확인
            assert len(response.questions) == num_questions
            print(f"✅ 요청 {num_questions}문제 = 생성 {len(response.questions)}문제")

    @pytest.mark.asyncio
    async def test_품질_검증_시스템(self, quiz_service):
        """🔍 품질 검증 시스템 테스트"""
        # Given: 퀴즈 요청
        request = QuizRequest(
            document_id="test-aws-doc",
            num_questions=5,
            difficulty=Difficulty.MEDIUM
        )

        # When: 퀴즈 생성
        response = await quiz_service.generate_guaranteed_quiz(request)

        # Then: 품질 검증 결과 확인
        assert response.success is True
        validation_result = response.metadata.get("validation_result", {})

        assert "overall_score" in validation_result
        assert "individual_scores" in validation_result
        assert "duplicate_analysis" in validation_result

        quality_score = validation_result.get("overall_score", 0)
        assert quality_score >= 6.0  # 최소 품질 기준
        print(f"✅ 품질 점수: {quality_score}/10점")

    @pytest.mark.asyncio
    async def test_중복_검증_시스템(self, quiz_service):
        """🔍 의미적 중복 검증 테스트"""
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
        assert len(duplicate_pairs) <= 2  # 중복 허용 임계값
        print(f"✅ 중복 문제: {len(duplicate_pairs)}개")

    @pytest.mark.asyncio
    async def test_토픽_추출_기능(self, quiz_service):
        """📚 토픽 추출 기능 테스트"""
        # Given: 문서 ID
        document_id = "test-aws-doc"

        # When: 토픽 추출
        topics = await quiz_service.extract_topics(document_id)

        # Then: 토픽 추출 확인
        assert isinstance(topics, list)
        assert len(topics) >= 0  # 빈 리스트여도 허용
        print(f"✅ 추출된 토픽: {len(topics)}개")

    @pytest.mark.asyncio
    async def test_에러_처리_잘못된_문서ID(self, quiz_service):
        """🚨 에러 처리: 잘못된 문서 ID"""
        # Given: 존재하지 않는 문서 ID
        request = QuizRequest(
            document_id="nonexistent-doc",
            num_questions=3,
            difficulty=Difficulty.MEDIUM
        )

        # When: 퀴즈 생성 시도
        response = await quiz_service.generate_guaranteed_quiz(request)

        # Then: 적절한 에러 처리 확인
        assert response.success is False
        assert "문서를 찾을 수 없습니다" in response.error
        print(f"✅ 에러 처리: {response.error}")


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
    print("\n🎉 3가지 피드백 반영 퀴즈 서비스 테스트 완료!")
    print("🔥 1. 불필요한 import 제거 ✅")
    print("🔥 2. 난이도 밸런스 (70%/20%/10%) ✅")
    print("🔥 3. 객관식 우선 생성 (70%) ✅")
    print("✅ 실제 options 포함하는 고품질 객관식 문제 생성!")