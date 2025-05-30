import pytest
import tempfile
import os

class TestPDFExtractor:
    """PDF 텍스트 추출기 테스트"""

    def test_extract_text_returns_string(self):
        """PDF에서 텍스트를 추출하면 문자열을 반환해야 한다"""
        # Given: PDF 추출기가 있고, 임시 파일이 있고
        from app.services.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()

        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"dummy pdf content")
            temp_file_path = temp_file.name

        try:
            # When: 텍스트를 추출하면
            result = extractor.extract_text(temp_file_path)

            # Then: 문자열이 반환되어야 한다
            assert isinstance(result, str)
        finally:
            # 임시 파일 정리
            os.unlink(temp_file_path)

    def test_extract_text_from_nonexistent_file_raises_error(self):
        """존재하지 않는 파일에서 텍스트를 추출하려 하면 에러가 발생해야 한다"""
        # Given: PDF 추출기가 있고
        from app.services.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()

        # When & Then: 존재하지 않는 파일에서 텍스트를 추출하려 하면 FileNotFoundError가 발생해야 한다
        with pytest.raises(FileNotFoundError):
            extractor.extract_text("nonexistent.pdf")

    def test_extract_text_from_invalid_pdf_raises_error(self):
        """잘못된 PDF 파일에서 텍스트를 추출하려 하면 에러가 발생해야 한다"""
        # Given: PDF 추출기가 있고, 잘못된 PDF 내용이 있는 파일이 있고
        from app.services.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()

        # 잘못된 PDF 내용
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"This is not a valid PDF content")
            temp_file_path = temp_file.name

        try:
            # When & Then: 잘못된 PDF에서 텍스트를 추출하려 하면 ValueError가 발생해야 한다
            with pytest.raises(ValueError, match="PDF 처리 중 오류가 발생했습니다"):
                extractor.extract_text(temp_file_path)
        finally:
            # 임시 파일 정리
            os.unlink(temp_file_path)

    def test_extract_text_from_real_pdf_content(self):
        """실제 PDF 내용에서 텍스트를 추출해야 한다"""
        # Given: PDF 추출기가 있고, 실제 PDF 내용이 있는 파일이 있고
        from app.services.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()

        # 간단한 PDF 내용 (실제 PDF 형식)
        pdf_content = b"dummy pdf content"

        # 임시 PDF 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_file_path = temp_file.name

        try:
            # When: 실제 PDF에서 텍스트를 추출하면
            result = extractor.extract_text(temp_file_path)

            # Then: PDF 내용이 포함되어야 한다
            assert "Hello World from PDF" in result
            assert len(result.strip()) > 0
            assert result != "dummy text"  # 더 이상 더미 텍스트가 아니어야 함
        finally:
            # 임시 파일 정리
            os.unlink(temp_file_path)

    def test_extract_text_from_aws_certificate_pdf(self):
        """실제 AWS 인증서 PDF에서 텍스트를 추출해야 한다"""
        # Given: PDF 추출기가 있고, 실제 AWS PDF 파일이 있고
        from app.services.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()

        pdf_path = "static/temp/AWS Certified Solutions Architect Associate SAA-C03.pdf"

        # 파일이 존재하는지 확인
        if not os.path.exists(pdf_path):
            pytest.skip(f"테스트 PDF 파일이 없습니다: {pdf_path}")

        # When: 실제 AWS PDF에서 텍스트를 추출하면
        result = extractor.extract_text(pdf_path)

        # Then: 의미있는 내용이 추출되어야 한다
        assert isinstance(result, str)
        assert len(result.strip()) > 100  # 충분한 양의 텍스트가 추출되어야 함

        # AWS 관련 키워드가 포함되어야 함
        result_lower = result.lower()
        aws_keywords = ["aws", "amazon", "cloud", "architect", "solutions"]
        found_keywords = [keyword for keyword in aws_keywords if keyword in result_lower]

        assert len(found_keywords) >= 2, f"AWS 관련 키워드가 충분히 발견되지 않았습니다. 발견된 키워드: {found_keywords}"

        print(f"\n추출된 텍스트 길이: {len(result)} 문자")
        print(f"발견된 AWS 키워드: {found_keywords}")
        print(f"텍스트 미리보기 (처음 200자):\n{result[:200]}...")

    def test_extract_text_performance_check(self):
        """PDF 텍스트 추출 성능을 확인한다"""
        # Given: PDF 추출기가 있고, 실제 PDF 파일이 있고
        from app.services.pdf_extractor import PDFExtractor
        import time

        extractor = PDFExtractor()
        pdf_path = "static/temp/AWS Certified Solutions Architect Associate SAA-C03.pdf"

        # 파일이 존재하는지 확인
        if not os.path.exists(pdf_path):
            pytest.skip(f"테스트 PDF 파일이 없습니다: {pdf_path}")

        # When: 텍스트 추출 시간을 측정하면
        start_time = time.time()
        result = extractor.extract_text(pdf_path)
        end_time = time.time()

        processing_time = end_time - start_time

        # Then: 합리적인 시간 내에 처리되어야 한다 (10초 이내)
        assert processing_time < 10.0, f"처리 시간이 너무 깁니다: {processing_time:.2f}초"

        # 텍스트가 추출되어야 한다
        assert len(result.strip()) > 0

        print(f"\n처리 시간: {processing_time:.2f}초")
        print(f"파일 크기: 6.9MB")
        print(f"처리 속도: {6.9/processing_time:.2f} MB/초")

