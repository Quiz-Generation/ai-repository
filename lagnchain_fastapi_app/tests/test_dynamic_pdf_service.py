#!/usr/bin/env python3
"""
동적 PDF 서비스 테스트
- 내용 유형 감지 테스트
- 추출기 선택 로직 테스트
- 실제 PDF 파일 추출 테스트
- 성능 벤치마크 테스트
"""
import pytest
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.dynamic_pdf import (
    DynamicPDFService
)
from app.schemas.dynamic_pdf import (
    ContentType,
    Priority,
    ExtractionResult
)


class TestDynamicPDFService:
    """동적 PDF 서비스 테스트"""

    @pytest.fixture
    def service(self):
        """서비스 인스턴스"""
        return DynamicPDFService()

    @pytest.fixture
    def mock_pdf_files(self):
        """테스트용 PDF 파일 경로"""
        return {
            "korean_small": "static/temp/lecture-DynamicProgramming.pdf",
            "english_large": "static/temp/AWS Certified Solutions Architect Associate SAA-C03.pdf"
        }

    def test_content_type_detection(self, service):
        """내용 유형 감지 테스트"""

        # Mock PyMuPDF
        with patch('fitz.open') as mock_fitz:
            # 한글 문서 시뮬레이션
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = "동적계획법은 알고리즘의 한 분야입니다. Dynamic Programming is important."
            mock_doc.__len__.return_value = 1
            mock_doc.__getitem__.return_value = mock_page
            mock_fitz.return_value = mock_doc

            content_type = service.detect_content_type("/fake/korean.pdf")
            assert content_type == ContentType.MIXED

            # 영문 문서 시뮬레이션
            mock_page.get_text.return_value = "This is a technical document about AWS API and HTTP methods."
            content_type = service.detect_content_type("/fake/english.pdf")
            assert content_type == ContentType.TECHNICAL

            # 순수 영문 문서
            mock_page.get_text.return_value = "This is a regular English document without technical terms."
            content_type = service.detect_content_type("/fake/english2.pdf")
            assert content_type == ContentType.ENGLISH

    def test_extractor_selection_logic(self, service):
        """추출기 선택 로직 테스트"""

        # 한글 문서는 항상 PDFMiner
        extractor = service._select_extractor_v2(1.0, ContentType.KOREAN, Priority.SPEED)
        assert extractor == "pdfminer"

        extractor = service._select_extractor_v2(10.0, ContentType.MIXED, Priority.BALANCED)
        assert extractor == "pdfminer"

        # 품질 우선은 항상 PDFMiner
        extractor = service._select_extractor_v2(1.0, ContentType.ENGLISH, Priority.QUALITY)
        assert extractor == "pdfminer"

        # 속도 우선 + 영문 = PyMuPDF
        extractor = service._select_extractor_v2(10.0, ContentType.ENGLISH, Priority.SPEED)
        assert extractor == "pymupdf"

        # 대용량 영문 = PyMuPDF
        extractor = service._select_extractor_v2(25.0, ContentType.TECHNICAL, Priority.BALANCED)
        assert extractor == "pymupdf"

        # 소용량 = PDFMiner
        extractor = service._select_extractor_v2(2.0, ContentType.UNKNOWN, Priority.BALANCED)
        assert extractor == "pdfminer"

    def test_extraction_result_structure(self, service):
        """추출 결과 구조 테스트"""

        # Mock 추출기와 파일
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024*1024), \
             patch.object(service, 'detect_content_type', return_value=ContentType.KOREAN), \
             patch('app.services.pdf_extractor.PDFExtractorFactory.create') as mock_factory:

            # Mock 추출기
            mock_extractor = MagicMock()
            mock_extractor.extract_text.return_value = "테스트 텍스트"
            mock_factory.return_value = mock_extractor

            result = service.extract_text("/fake/test.pdf", Priority.BALANCED)

            # 결과 구조 검증
            assert isinstance(result, ExtractionResult)
            assert result.success is True
            assert result.text == "테스트 텍스트"
            assert result.extractor_used == "pdfminer"
            assert result.content_type == "korean"
            assert result.priority == "balanced"
            assert result.file_size_mb == 1.0
            assert result.text_length == 6
            assert "auto_selected" in result.metadata
            assert "selection_reason" in result.metadata

    def test_error_handling(self, service):
        """오류 처리 테스트"""

        # 존재하지 않는 파일
        with pytest.raises(FileNotFoundError):
            service.select_optimal_extractor("/nonexistent/file.pdf")

        # 추출 실패 시뮬레이션
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024*1024), \
             patch.object(service, 'detect_content_type', return_value=ContentType.ENGLISH), \
             patch('app.services.pdf_extractor.PDFExtractorFactory.create') as mock_factory:

            mock_extractor = MagicMock()
            mock_extractor.extract_text.side_effect = Exception("추출 실패")
            mock_factory.return_value = mock_extractor

            result = service.extract_text("/fake/test.pdf")

            assert result.success is False
            assert "추출 실패" in result.error
            assert result.metadata["auto_selected"] is True

    def test_manual_extractor_selection(self, service):
        """수동 추출기 선택 테스트"""

        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=2*1024*1024), \
             patch.object(service, 'detect_content_type', return_value=ContentType.ENGLISH), \
             patch('app.services.pdf_extractor.PDFExtractorFactory.create') as mock_factory:

            mock_extractor = MagicMock()
            mock_extractor.extract_text.return_value = "Manual extraction"
            mock_factory.return_value = mock_extractor

            result = service.extract_with_specific_extractor("/fake/test.pdf", "pymupdf")

            assert result.success is True
            assert result.extractor_used == "pymupdf"
            assert result.priority == "manual"
            assert result.metadata["auto_selected"] is False
            assert result.metadata["manual_choice"] is True

    def test_recommendations(self, service):
        """추천 정보 테스트"""

        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=5*1024*1024), \
             patch.object(service, 'detect_content_type', return_value=ContentType.ENGLISH):

            recommendations = service.get_extractor_recommendations("/fake/test.pdf")

            assert "file_info" in recommendations
            assert "recommendations" in recommendations
            assert "extractor_profiles" in recommendations

            assert recommendations["file_info"]["size_mb"] == 5.0
            assert recommendations["file_info"]["content_type"] == "english"

            # 3가지 우선순위별 추천 확인
            assert "speed" in recommendations["recommendations"]
            assert "quality" in recommendations["recommendations"]
            assert "balanced" in recommendations["recommendations"]

            # 각 추천에 추출기와 이유 포함
            for priority, rec in recommendations["recommendations"].items():
                assert "extractor" in rec
                assert "reason" in rec


class TestRealPDFFiles:
    """실제 PDF 파일 테스트"""

    @pytest.fixture
    def service(self):
        return DynamicPDFService()

    @pytest.fixture
    def pdf_files(self):
        """실제 PDF 파일 경로"""
        return {
            "korean": "static/temp/lecture-DynamicProgramming.pdf",
            "english": "static/temp/AWS Certified Solutions Architect Associate SAA-C03.pdf"
        }

    @pytest.mark.skipif(
        not os.path.exists("static/temp/lecture-DynamicProgramming.pdf"),
        reason="테스트 PDF 파일이 없습니다"
    )
    def test_korean_pdf_extraction(self, service, pdf_files):
        """한글 PDF 추출 테스트"""

        result = service.extract_text(pdf_files["korean"], Priority.BALANCED)

        assert result.success is True
        assert result.extractor_used == "pdfminer"  # 한글은 항상 PDFMiner
        assert result.content_type == "mixed"
        assert "Dynamic" in result.text
        assert "동적계획법" in result.text
        assert result.text_length > 10000  # 충분한 텍스트 추출
        assert result.file_size_mb > 0
        assert result.extraction_time > 0

        # 메타데이터 확인
        assert result.metadata["auto_selected"] is True
        assert "한글 문서 감지" in result.metadata["selection_reason"]

    @pytest.mark.skipif(
        not os.path.exists("static/temp/AWS Certified Solutions Architect Associate SAA-C03.pdf"),
        reason="테스트 PDF 파일이 없습니다"
    )
    def test_english_pdf_extraction(self, service, pdf_files):
        """영문 PDF 추출 테스트"""

        # 속도 우선 테스트
        result_speed = service.extract_text(pdf_files["english"], Priority.SPEED)

        assert result_speed.success is True
        assert result_speed.extractor_used == "pymupdf"  # 대용량 영문은 PyMuPDF
        assert "AWS" in result_speed.text
        assert result_speed.file_size_mb > 5

        # 품질 우선 테스트
        result_quality = service.extract_text(pdf_files["english"], Priority.QUALITY)

        assert result_quality.success is True
        assert result_quality.extractor_used == "pdfminer"  # 품질은 항상 PDFMiner

        # 속도 비교 (PyMuPDF가 더 빨라야 함)
        assert result_speed.extraction_time < result_quality.extraction_time
        assert result_speed.speed_mbps > result_quality.speed_mbps

    def test_extractor_comparison(self, service, pdf_files):
        """추출기별 성능 비교 테스트"""

        if not os.path.exists(pdf_files["korean"]):
            pytest.skip("테스트 PDF 파일이 없습니다")

        extractors = ["pdfminer", "pdfplumber", "pymupdf"]
        results = {}

        for extractor in extractors:
            result = service.extract_with_specific_extractor(pdf_files["korean"], extractor)
            if result.success:
                results[extractor] = result

        # 모든 추출기가 성공해야 함
        assert len(results) == 3

        # PyMuPDF가 가장 빨라야 함
        assert results["pymupdf"].extraction_time < results["pdfminer"].extraction_time
        assert results["pymupdf"].extraction_time < results["pdfplumber"].extraction_time

        # 모든 추출기가 키워드를 찾아야 함
        for result in results.values():
            assert "Dynamic" in result.text or "Programming" in result.text


class TestPerformanceBenchmark:
    """성능 벤치마크 테스트"""

    @pytest.fixture
    def service(self):
        return DynamicPDFService()

    def test_selection_speed(self, service):
        """추출기 선택 속도 테스트"""

        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024*1024), \
             patch.object(service, 'detect_content_type', return_value=ContentType.KOREAN):

            start_time = time.time()

            # 100번 선택 테스트
            for _ in range(100):
                extractor = service.select_optimal_extractor("/fake/test.pdf")
                assert extractor == "pdfminer"

            elapsed = time.time() - start_time

            # 선택 속도가 충분히 빨라야 함 (100번에 1초 미만)
            assert elapsed < 1.0
            print(f"선택 속도: {elapsed:.3f}초 (100회)")

    @pytest.mark.skipif(
        not os.path.exists("static/temp/lecture-DynamicProgramming.pdf"),
        reason="테스트 PDF 파일이 없습니다"
    )
    def test_memory_usage(self, service):
        """메모리 사용량 테스트"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 여러 번 추출 테스트
        for _ in range(5):
            result = service.extract_text("static/temp/lecture-DynamicProgramming.pdf")
            assert result.success

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 메모리 증가가 적당해야 함 (100MB 미만)
        print(f"메모리 증가: {memory_increase:.1f}MB")
        assert memory_increase < 100


class TestIntegrationScenarios:
    """통합 시나리오 테스트"""

    @pytest.fixture
    def service(self):
        return DynamicPDFService()

    def test_quiz_generation_scenario(self, service):
        """퀴즈 생성 시나리오 테스트"""

        # 한글 학술 자료 처리
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=2*1024*1024), \
             patch.object(service, 'detect_content_type', return_value=ContentType.MIXED), \
             patch('app.services.pdf_extractor.PDFExtractorFactory.create') as mock_factory:

            mock_extractor = MagicMock()
            mock_extractor.extract_text.return_value = "동적계획법은 알고리즘 설계 기법 중 하나입니다."
            mock_factory.return_value = mock_extractor

            # 품질 우선으로 추출
            result = service.extract_text("/fake/lecture.pdf", Priority.QUALITY)

            assert result.success
            assert result.extractor_used == "pdfminer"  # 한글이므로 PDFMiner
            assert "동적계획법" in result.text
            assert result.metadata["auto_selected"]

    def test_realtime_api_scenario(self, service):
        """실시간 API 시나리오 테스트"""

        # 대용량 영문 문서 빠른 처리
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=50*1024*1024), \
             patch.object(service, 'detect_content_type', return_value=ContentType.TECHNICAL), \
             patch('app.services.pdf_extractor.PDFExtractorFactory.create') as mock_factory:

            mock_extractor = MagicMock()
            mock_extractor.extract_text.return_value = "AWS API documentation"
            mock_factory.return_value = mock_extractor

            # 속도 우선으로 추출
            result = service.extract_text("/fake/api_doc.pdf", Priority.SPEED)

            assert result.success
            assert result.extractor_used == "pymupdf"  # 대용량 영문이므로 PyMuPDF
            assert result.file_size_mb == 50.0
            assert "고속 처리" in result.metadata["selection_reason"]

    def test_document_analysis_scenario(self, service):
        """문서 분석 시나리오 테스트"""

        # 구조 보존이 중요한 문서 분석
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=5*1024*1024), \
             patch.object(service, 'detect_content_type', return_value=ContentType.ENGLISH), \
             patch('app.services.pdf_extractor.PDFExtractorFactory.create') as mock_factory:

            mock_extractor = MagicMock()
            mock_extractor.extract_text.return_value = "Document structure analysis"
            mock_factory.return_value = mock_extractor

            # 품질 우선으로 추출 (구조 보존)
            result = service.extract_text("/fake/analysis.pdf", Priority.QUALITY)

            assert result.success
            assert result.extractor_used == "pdfminer"  # 품질 우선이므로 PDFMiner
            assert "구조 보존" in result.metadata["selection_reason"]


if __name__ == "__main__":
    # 개별 테스트 실행
    pytest.main([__file__, "-v"])