import pytest
import os


class TestPDFExtractor:
    """PDF 텍스트 추출기 테스트"""

    def test_can_extract_text_from_pdf(self):
        """PDF 파일에서 텍스트를 추출할 수 있다"""
        # Given: PDF 파일이 있고
        pdf_path = "static/temp/lecture-DynamicProgramming.pdf"

        # When: PDF 추출기로 텍스트를 추출하면
        from app.services.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()
        text = extractor.extract_text(pdf_path)

        # Then: 텍스트가 반환되어야 한다
        assert isinstance(text, str)
        assert len(text) > 0

    def test_should_raise_error_when_file_not_exists(self):
        """존재하지 않는 파일에 대해 에러를 발생시켜야 한다"""
        # Given: 존재하지 않는 PDF 파일 경로가 있고
        non_existent_path = "not_exists.pdf"

        # When & Then: PDF 추출기로 텍스트를 추출하면 에러가 발생해야 한다
        from app.services.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract_text(non_existent_path)

    def test_should_extract_meaningful_text_from_dynamic_programming_pdf(self):
        """다이나믹 프로그래밍 PDF에서 의미있는 텍스트가 추출되어야 한다"""
        # Given: 다이나믹 프로그래밍 PDF 파일이 있고
        pdf_path = "static/temp/lecture-DynamicProgramming.pdf"

        # PDF 파일이 실제로 존재하는지 확인
        if not os.path.exists(pdf_path):
            pytest.skip(f"테스트 PDF 파일이 없습니다: {pdf_path}")

        # When: PDF 추출기로 텍스트를 추출하면
        from app.services.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()
        text = extractor.extract_text(pdf_path)

        # Then: 다이나믹 프로그래밍 관련 키워드가 포함되어야 한다
        text_lower = text.lower()
        assert "dynamic" in text_lower or "programming" in text_lower, f"추출된 텍스트에 Dynamic/Programming 키워드가 없습니다. 텍스트: {text[:200]}..."

    def test_can_use_different_pdf_loaders_via_factory(self):
        """팩토리 패턴으로 다양한 PDF 로더를 사용할 수 있어야 한다"""
        # Given: 다이나믹 프로그래밍 PDF 파일이 있고
        pdf_path = "static/temp/lecture-DynamicProgramming.pdf"

        if not os.path.exists(pdf_path):
            pytest.skip(f"테스트 PDF 파일이 없습니다: {pdf_path}")

        # When: 팩토리로 PDFMiner 추출기를 생성하면
        from app.services.pdf_extractor import PDFExtractorFactory
        pdfminer_extractor = PDFExtractorFactory.create("pdfminer")
        pdfminer_text = pdfminer_extractor.extract_text(pdf_path)

        # Then: PDFMiner로 텍스트가 추출되어야 한다
        assert isinstance(pdfminer_text, str)
        assert len(pdfminer_text) > 0
        assert "dynamic" in pdfminer_text.lower() or "programming" in pdfminer_text.lower()

        # When: 팩토리로 PDFPlumber 추출기를 생성하면
        pdfplumber_extractor = PDFExtractorFactory.create("pdfplumber")
        pdfplumber_text = pdfplumber_extractor.extract_text(pdf_path)

        # Then: PDFPlumber로도 텍스트가 추출되어야 한다
        assert isinstance(pdfplumber_text, str)
        assert len(pdfplumber_text) > 0
        assert "dynamic" in pdfplumber_text.lower() or "programming" in pdfplumber_text.lower()

    def test_factory_should_return_pdfminer_as_default(self):
        """팩토리는 기본값으로 PDFMiner를 반환해야 한다 (AutoRAG 실험 1위)"""
        # When: 팩토리에서 기본 추출기를 생성하면
        from app.services.pdf_extractor import PDFExtractorFactory
        default_extractor = PDFExtractorFactory.create()

        # Then: PDFMiner 추출기가 반환되어야 한다
        assert default_extractor.__class__.__name__ == "PDFMinerExtractor"