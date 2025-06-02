#!/usr/bin/env python3
"""
통합 PDF 벡터 API 테스트
"""
import pytest
from pathlib import Path
from io import BytesIO

from lagnchain_fastapi_app.app.api.pdf_service import router
from lagnchain_fastapi_app.app.services.vector_service import PDFVectorService


class TestIntegratedPDFVector:
    """통합 PDF 벡터 기능 테스트"""

    @pytest.fixture
    def vector_service(self):
        return PDFVectorService(db_type="weaviate")

    @pytest.fixture
    def sample_pdf_path(self):
        return Path(__file__).parent.parent / "static" / "temp" / "lecture-DynamicProgramming.pdf"

    def test_vector_service_integration(self, vector_service):
        """벡터 서비스 통합 테스트"""
        # 샘플 텍스트 처리
        sample_text = "동적계획법은 최적화 문제를 해결하는 알고리즘 기법입니다. " * 20

        # 처리 및 저장
        result = vector_service.process_pdf_text(sample_text, "integration_test.pdf")

        assert result["success"] is True
        assert result["total_chunks"] > 0
        assert result["stored_chunks"] > 0
        assert result["db_type"] == "weaviate"

        # 검색 테스트
        search_results = vector_service.search_documents("동적계획법", top_k=3)
        assert len(search_results) > 0
        assert all(result["similarity"] > 0 for result in search_results)

        # 통계 확인
        stats = vector_service.get_stats()
        assert stats["total_documents"] > 0
        assert stats["db_type"] == "weaviate"
        assert "weaviate" in stats["supported_dbs"]

        print(f"✅ 통합 테스트 성공:")
        print(f"   - 처리: {result['stored_chunks']}개 청크")
        print(f"   - 검색: {len(search_results)}개 결과")
        print(f"   - 총 문서: {stats['total_documents']}개")

    def test_database_switching(self, vector_service):
        """데이터베이스 전환 테스트"""
        # 초기: weaviate
        assert vector_service.db_type == "weaviate"

        # chroma로 전환
        success = vector_service.switch_database("chroma")
        assert success is True
        assert vector_service.db_type == "chroma"

        # 다시 weaviate로 전환
        success = vector_service.switch_database("weaviate")
        assert success is True
        assert vector_service.db_type == "weaviate"

        print("✅ 데이터베이스 전환 테스트 성공")

    def test_real_pdf_workflow(self, vector_service, sample_pdf_path):
        """실제 PDF 워크플로우 테스트"""
        if not sample_pdf_path.exists():
            pytest.skip("샘플 PDF 파일이 없습니다")

        # PDF 텍스트 직접 추출하여 테스트
        try:
            import fitz
            doc = fitz.open(str(sample_pdf_path))
            pdf_text = ""
            for page in doc:
                pdf_text += page.get_text()
            doc.close()
        except ImportError:
            pytest.skip("PyMuPDF가 설치되지 않았습니다")

        # 벡터 처리
        process_result = vector_service.process_pdf_text(pdf_text, sample_pdf_path.name)

        assert process_result["success"] is True
        assert process_result["total_chunks"] > 0

        # 실제 개념 검색
        queries = ["동적계획법", "메모이제이션", "최적화"]

        for query in queries:
            search_results = vector_service.search_documents(query, top_k=3)
            assert len(search_results) > 0

            # 유사도가 합리적인 범위인지 확인
            for result in search_results:
                assert -1.0 <= result["similarity"] <= 1.0

        print(f"✅ 실제 PDF 워크플로우 성공:")
        print(f"   - 파일: {sample_pdf_path.name}")
        print(f"   - 텍스트: {len(pdf_text):,}자")
        print(f"   - 청크: {process_result['total_chunks']}개")
        print(f"   - 검색 쿼리: {len(queries)}개 테스트")

    def test_performance_benchmark(self, vector_service):
        """성능 벤치마크 테스트"""
        import time

        # 테스트 데이터 생성
        test_texts = [
            "동적계획법은 최적화 문제를 해결합니다. " * 50,
            "FastAPI는 Python 웹 프레임워크입니다. " * 50,
            "벡터 데이터베이스는 유사성 검색을 지원합니다. " * 50
        ]

        # 저장 성능 측정
        store_start = time.time()

        for i, text in enumerate(test_texts):
            result = vector_service.process_pdf_text(text, f"benchmark_{i}.pdf")
            assert result["success"] is True

        store_time = time.time() - store_start

        # 검색 성능 측정
        search_queries = ["동적계획법", "FastAPI", "벡터"]
        search_times = []

        for query in search_queries:
            search_start = time.time()
            results = vector_service.search_documents(query, top_k=5)
            search_time = time.time() - search_start
            search_times.append(search_time)

            assert len(results) > 0

        avg_search_time = sum(search_times) / len(search_times)

        # 성능 기준 검증
        assert store_time < 5.0, f"저장 시간 초과: {store_time:.2f}초"
        assert avg_search_time < 0.1, f"검색 시간 초과: {avg_search_time:.3f}초"

        print(f"✅ 성능 벤치마크:")
        print(f"   - 저장: {store_time:.2f}초 ({len(test_texts)}개 문서)")
        print(f"   - 검색: {avg_search_time:.3f}초 (평균)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])