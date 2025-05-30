import pytest
import os


class TestPDFLoaderStrategy:
    """PDF 로더 전략 패턴 테스트"""

    def test_pdf_loader_factory_creates_different_loaders(self):
        """PDF 로더 팩토리가 다양한 로더를 생성할 수 있어야 한다"""
        # Given: PDF 로더 팩토리가 있고
        from app.services.pdf_loader_factory import PDFLoaderFactory

        factory = PDFLoaderFactory()

        # When: 다양한 로더 타입을 요청하면
        pypdf_loader = factory.create_loader("pypdf")
        pymupdf_loader = factory.create_loader("pymupdf")
        unstructured_loader = factory.create_loader("unstructured")

        # Then: 각각 다른 타입의 로더가 생성되어야 한다
        assert pypdf_loader.__class__.__name__ == "PyPDFLoader"
        assert pymupdf_loader.__class__.__name__ == "PyMuPDFLoader"
        assert unstructured_loader.__class__.__name__ == "UnstructuredLoader"

    def test_pdf_extractor_with_different_loaders(self):
        """PDF 추출기가 다양한 로더를 사용할 수 있어야 한다"""
        # Given: PDF 추출기가 있고, 실제 PDF 파일이 있고
        from app.services.pdf_extractor import PDFExtractor

        pdf_path = "static/temp/AWS Certified Solutions Architect Associate SAA-C03.pdf"

        if not os.path.exists(pdf_path):
            pytest.skip(f"테스트 PDF 파일이 없습니다: {pdf_path}")

        # When: 다양한 로더로 텍스트를 추출하면
        extractor_pypdf = PDFExtractor(loader_type="pypdf")
        extractor_pymupdf = PDFExtractor(loader_type="pymupdf")

        result_pypdf = extractor_pypdf.extract_text(pdf_path)
        result_pymupdf = extractor_pymupdf.extract_text(pdf_path)

        # Then: 모두 텍스트가 추출되어야 한다
        assert isinstance(result_pypdf, str)
        assert isinstance(result_pymupdf, str)
        assert len(result_pypdf.strip()) > 100
        assert len(result_pymupdf.strip()) > 100

        print(f"\nPyPDF2 추출 길이: {len(result_pypdf)} 문자")
        print(f"PyMuPDF 추출 길이: {len(result_pymupdf)} 문자")

    def test_pdf_loader_performance_comparison(self):
        """다양한 PDF 로더의 성능을 비교할 수 있어야 한다"""
        # Given: PDF 추출기가 있고, 실제 PDF 파일이 있고
        from app.services.pdf_extractor import PDFExtractor
        import time

        pdf_path = "static/temp/AWS Certified Solutions Architect Associate SAA-C03.pdf"

        if not os.path.exists(pdf_path):
            pytest.skip(f"테스트 PDF 파일이 없습니다: {pdf_path}")

        loaders = ["pypdf", "pymupdf"]
        results = {}

        # When: 각 로더의 성능을 측정하면
        for loader_type in loaders:
            extractor = PDFExtractor(loader_type=loader_type)

            start_time = time.time()
            try:
                text = extractor.extract_text(pdf_path)
                end_time = time.time()

                results[loader_type] = {
                    "success": True,
                    "time": end_time - start_time,
                    "text_length": len(text),
                    "error": None
                }
            except Exception as e:
                end_time = time.time()
                results[loader_type] = {
                    "success": False,
                    "time": end_time - start_time,
                    "text_length": 0,
                    "error": str(e)
                }

        # Then: 성능 비교 결과를 출력한다
        print(f"\n=== PDF 로더 성능 비교 ===")
        for loader_type, result in results.items():
            if result["success"]:
                speed = 6.9 / result["time"]  # MB/초
                print(f"{loader_type}: {result['time']:.2f}초, {result['text_length']} 문자, {speed:.2f} MB/초")
            else:
                print(f"{loader_type}: 실패 - {result['error']}")

        # 적어도 하나는 성공해야 한다
        successful_loaders = [k for k, v in results.items() if v["success"]]
        assert len(successful_loaders) >= 1, "적어도 하나의 로더는 성공해야 합니다"

    def test_pdf_loader_recommendations(self):
        """사용 사례별 로더 추천 기능을 테스트한다"""
        # Given: PDF 로더 팩토리가 있고
        from app.services.pdf_loader_factory import PDFLoaderFactory

        factory = PDFLoaderFactory()

        # When: 다양한 사용 사례에 대한 추천을 요청하면
        general_loader = factory.get_recommended_loader("general")
        tables_loader = factory.get_recommended_loader("tables")
        complex_loader = factory.get_recommended_loader("complex")
        fast_loader = factory.get_recommended_loader("fast")

        # Then: 적절한 로더가 추천되어야 한다
        assert general_loader == "pypdf"
        assert tables_loader == "pymupdf"
        assert complex_loader == "unstructured"
        assert fast_loader == "pypdf"

        print(f"\n=== 사용 사례별 추천 로더 ===")
        print(f"일반적인 용도: {general_loader}")
        print(f"테이블이 많은 문서: {tables_loader}")
        print(f"복잡한 구조의 문서: {complex_loader}")
        print(f"빠른 처리가 필요한 경우: {fast_loader}")

    def test_pdf_extractor_auto_recommendation(self):
        """PDF 추출기가 자동으로 최적 로더를 선택할 수 있어야 한다"""
        # Given: PDF 추출기가 있고
        from app.services.pdf_extractor import PDFExtractor
        from app.services.pdf_loader_factory import PDFLoaderFactory

        factory = PDFLoaderFactory()

        # When: 사용 사례에 따른 추출기를 생성하면
        fast_extractor = PDFExtractor(loader_type=factory.get_recommended_loader("fast"))
        tables_extractor = PDFExtractor(loader_type=factory.get_recommended_loader("tables"))

        # Then: 올바른 로더가 설정되어야 한다
        fast_info = fast_extractor.get_loader_info()
        tables_info = tables_extractor.get_loader_info()

        assert fast_info["loader_type"] == "pypdf"
        assert tables_info["loader_type"] == "pymupdf"

        print(f"\n빠른 처리용 추출기: {fast_info}")
        print(f"테이블 처리용 추출기: {tables_info}")