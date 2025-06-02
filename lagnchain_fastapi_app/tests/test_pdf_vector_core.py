#!/usr/bin/env python3
"""
핵심 PDF 벡터 저장 TDD 테스트
"""
import pytest
from pathlib import Path

# PDF 추출용
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# 우리 서비스 import
from app.services.vector_service import (
    VectorDBFactory,
    PDFVectorService,
    TextEmbedder,
    TextChunker
)


class SimplePDFReader:
    """간단한 PDF 텍스트 추출기"""

    def extract_text(self, pdf_path: str) -> str:
        if not HAS_PYMUPDF:
            return f"Mock text from {Path(pdf_path).name} (PyMuPDF 없음)"

        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            return f"Error: {str(e)}"


class TestVectorDBFactory:
    """1단계: 팩토리 패턴 테스트"""

    def test_factory_creates_weaviate_db(self):
        """Weaviate DB 생성 테스트"""
        db = VectorDBFactory.create_vector_db("weaviate")
        assert db.name == "weaviate"
        assert db.count_documents() == 0
        print("✅ Weaviate DB 생성 성공")

    def test_factory_creates_chroma_db(self):
        """Chroma DB 생성 테스트"""
        db = VectorDBFactory.create_vector_db("chroma")
        assert db.name == "chroma"
        assert db.count_documents() == 0
        print("✅ Chroma DB 생성 성공")

    def test_factory_invalid_db_type(self):
        """잘못된 DB 타입 테스트"""
        with pytest.raises(ValueError):
            VectorDBFactory.create_vector_db("invalid_db")
        print("✅ 잘못된 DB 타입 예외 처리 성공")

    def test_factory_supported_types(self):
        """지원하는 DB 타입 확인"""
        supported = VectorDBFactory.get_supported_types()
        assert "weaviate" in supported
        assert "chroma" in supported
        print(f"✅ 지원 DB 타입: {supported}")


class TestTextProcessing:
    """2단계: 텍스트 처리 테스트"""

    @pytest.fixture
    def embedder(self):
        return TextEmbedder()

    @pytest.fixture
    def chunker(self):
        return TextChunker(chunk_size=500, overlap=50)

    def test_text_embedding(self, embedder):
        """텍스트 임베딩 테스트"""
        text = "동적계획법은 복잡한 문제를 해결하는 알고리즘 기법입니다."
        vector = embedder.embed_text(text)

        assert isinstance(vector, list)
        assert len(vector) == 384
        assert all(isinstance(x, float) for x in vector)
        print(f"✅ 텍스트 임베딩: {len(vector)}차원")

    def test_text_chunking(self, chunker):
        """텍스트 청킹 테스트"""
        long_text = "동적계획법은 복잡한 문제를 해결하는 방법입니다. " * 50  # 긴 텍스트
        chunks = chunker.chunk_text(long_text)

        assert len(chunks) > 1
        assert all(len(chunk) > 50 for chunk in chunks)
        print(f"✅ 텍스트 청킹: {len(chunks)}개 청크")

    def test_same_text_same_vector(self, embedder):
        """동일 텍스트 → 동일 벡터 확인"""
        text = "테스트 텍스트"
        vector1 = embedder.embed_text(text)
        vector2 = embedder.embed_text(text)
        assert vector1 == vector2
        print("✅ 동일 텍스트 → 동일 벡터 확인")


class TestPDFVectorService:
    """3단계: PDF 벡터 서비스 핵심 테스트"""

    @pytest.fixture
    def pdf_service(self):
        return PDFVectorService(db_type="weaviate")

    @pytest.fixture
    def korean_pdf_path(self):
        return Path(__file__).parent.parent / "static" / "temp" / "lecture-DynamicProgramming.pdf"

    def test_service_initialization(self, pdf_service):
        """서비스 초기화 테스트"""
        assert pdf_service.db_type == "weaviate"
        stats = pdf_service.get_stats()
        assert stats["total_documents"] == 0
        assert stats["db_type"] == "weaviate"
        print("✅ 서비스 초기화 성공")

    def test_process_sample_text(self, pdf_service):
        """샘플 텍스트 처리 테스트"""
        sample_text = """
        동적계획법(Dynamic Programming)은 복잡한 문제를 더 작은 하위 문제로 분할하여 해결하는 알고리즘 설계 기법입니다.
        이 기법의 핵심 아이디어는 중복되는 부분 문제의 해를 저장하여 재계산을 피하는 것입니다.
        동적계획법은 최적 부분 구조와 중복되는 부분 문제라는 두 가지 조건을 만족할 때 적용할 수 있습니다.
        """ * 3  # 텍스트를 늘려서 청킹 테스트

        result = pdf_service.process_pdf_text(sample_text, "test_document")

        assert result["success"] is True
        assert result["total_chunks"] > 0
        assert result["stored_chunks"] > 0
        assert result["db_type"] == "weaviate"

        print(f"✅ 샘플 텍스트 처리: {result['stored_chunks']}개 청크 저장")

    def test_search_functionality(self, pdf_service):
        """검색 기능 테스트"""
        # 먼저 문서 저장
        sample_text = "동적계획법은 최적화 문제를 해결하는 알고리즘 기법입니다. 메모이제이션을 사용합니다."
        pdf_service.process_pdf_text(sample_text, "dp_document")

        # 검색 테스트
        results = pdf_service.search_documents("동적계획법", top_k=3)

        assert len(results) > 0
        assert all(-1.0 <= result["similarity"] <= 1.0 for result in results)

        print(f"✅ 검색 기능: {len(results)}개 결과")
        for i, result in enumerate(results):
            print(f"   {i+1}. 유사도: {result['similarity']:.3f}")

    def test_database_switching(self, pdf_service):
        """데이터베이스 전환 테스트"""
        # 초기 상태: weaviate
        assert pdf_service.db_type == "weaviate"

        # chroma로 전환
        success = pdf_service.switch_database("chroma")
        assert success is True
        assert pdf_service.db_type == "chroma"

        # 잘못된 DB로 전환 시도
        success = pdf_service.switch_database("invalid_db")
        assert success is False
        assert pdf_service.db_type == "chroma"  # 기존 유지

        print("✅ 데이터베이스 전환 테스트 성공")

    def test_real_pdf_processing(self, pdf_service, korean_pdf_path):
        """실제 PDF 처리 테스트"""
        if not korean_pdf_path.exists():
            pytest.skip("PDF 파일이 없습니다")

        # PDF 텍스트 추출
        pdf_reader = SimplePDFReader()
        pdf_text = pdf_reader.extract_text(str(korean_pdf_path))

        # 처리 결과 확인
        assert len(pdf_text) > 100, "PDF 텍스트가 너무 짧습니다"

        # 벡터 저장
        result = pdf_service.process_pdf_text(pdf_text, korean_pdf_path.name)

        assert result["success"] is True
        assert result["total_chunks"] > 0
        assert result["stored_chunks"] > 0

        print(f"✅ 실제 PDF 처리:")
        print(f"   - 파일: {korean_pdf_path.name}")
        print(f"   - 텍스트: {len(pdf_text):,}자")
        print(f"   - 청크: {result['total_chunks']}개")
        print(f"   - 저장: {result['stored_chunks']}개")

        # 실제 검색 테스트
        search_results = pdf_service.search_documents("동적계획법", top_k=3)
        assert len(search_results) > 0

        print(f"   - 검색: {len(search_results)}개 결과")


class TestVectorDBComparison:
    """4단계: 벡터 DB 비교 테스트"""

    @pytest.mark.parametrize("db_type", ["weaviate", "chroma"])
    def test_db_performance_comparison(self, db_type):
        """DB별 성능 비교"""
        service = PDFVectorService(db_type=db_type)

        # 테스트 텍스트
        test_text = "테스트용 문서입니다. " * 100

        # 처리 및 결과 확인
        result = service.process_pdf_text(test_text, f"test_{db_type}")

        assert result["success"] is True
        assert result["db_type"] == db_type

        # 검색 테스트
        search_results = service.search_documents("테스트", top_k=3)
        assert len(search_results) > 0

        print(f"✅ {db_type.upper()}: {result['stored_chunks']}개 저장, {len(search_results)}개 검색")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])