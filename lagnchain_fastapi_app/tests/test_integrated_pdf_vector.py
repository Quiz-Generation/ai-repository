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
        assert "document_id" in result

        # 검색 테스트
        search_results = vector_service.search_documents("동적계획법", top_k=3)
        assert len(search_results) > 0
        assert all(result["similarity"] != 0 for result in search_results)

        # 통계 확인
        stats = vector_service.get_stats()
        assert stats["total_documents"] > 0
        assert stats["db_type"] == "weaviate"
        assert "weaviate" in stats["supported_dbs"]

        print(f"✅ 통합 테스트 성공:")
        print(f"   - 처리: {result['stored_chunks']}개 청크")
        print(f"   - 문서 ID: {result['document_id']}")
        print(f"   - 검색: {len(search_results)}개 결과")
        print(f"   - 총 문서: {stats['total_documents']}개")

    def test_document_management_workflow(self, vector_service):
        """📁 문서 관리 워크플로우 테스트 (RAG용)"""
        # 1단계: 여러 문서 업로드
        doc1_result = vector_service.process_pdf_text(
            "Python은 강력한 프로그래밍 언어입니다. " * 10,
            "python_guide.pdf"
        )
        doc2_result = vector_service.process_pdf_text(
            "데이터베이스는 정보를 저장하는 시스템입니다. " * 10,
            "database_intro.pdf"
        )
        doc3_result = vector_service.process_pdf_text(
            "웹 개발에는 다양한 기술이 필요합니다. " * 10,
            "web_development.pdf"
        )

        # 문서 ID 확인
        doc1_id = doc1_result["document_id"]
        doc2_id = doc2_result["document_id"]
        doc3_id = doc3_result["document_id"]

        assert all([doc1_id, doc2_id, doc3_id])
        print(f"📁 3개 문서 업로드: {doc1_id[:8]}..., {doc2_id[:8]}..., {doc3_id[:8]}...")

        # 2단계: 문서 목록 조회
        document_list = vector_service.get_document_list()
        assert len(document_list) >= 3

        uploaded_ids = {doc["document_id"] for doc in document_list}
        assert doc1_id in uploaded_ids
        assert doc2_id in uploaded_ids
        assert doc3_id in uploaded_ids

        print(f"📋 문서 목록: {len(document_list)}개 확인")

        # 3단계: 특정 문서 정보 조회
        doc1_info = vector_service.get_document_info(doc1_id)
        assert doc1_info is not None
        assert doc1_info["source_filename"] == "python_guide.pdf"
        assert doc1_info["document_id"] == doc1_id

        print(f"📄 문서 정보: {doc1_info['source_filename']} ({doc1_info['chunk_count']}개 청크)")

        # 4단계: 특정 문서에서만 검색 (RAG 컨텍스트 추출용)
        python_results = vector_service.search_in_document("프로그래밍", doc1_id, top_k=3)
        db_results = vector_service.search_in_document("데이터베이스", doc2_id, top_k=3)

        assert len(python_results) > 0
        assert len(db_results) > 0

        # 검색 결과가 해당 문서에서만 나오는지 확인
        for result in python_results:
            assert result["metadata"]["document_id"] == doc1_id
        for result in db_results:
            assert result["metadata"]["document_id"] == doc2_id

        print(f"🎯 문서별 검색: Python({len(python_results)}개), DB({len(db_results)}개)")

        # 5단계: RAG용 컨텍스트 추출 시뮬레이션
        rag_context = ""
        for result in python_results:
            rag_context += result["text"] + "\n\n"

        assert len(rag_context) > 100
        print(f"🤖 RAG 컨텍스트: {len(rag_context)}자 추출 완료")

        print("✅ 문서 관리 워크플로우 테스트 성공 (RAG 준비 완료)")

    def test_rag_context_extraction(self, vector_service):
        """🤖 RAG 컨텍스트 추출 테스트"""
        # 다양한 주제의 학습 문서 업로드
        topics_and_content = {
            "algorithm": "동적계획법은 복잡한 문제를 해결하는 효율적인 방법입니다. 메모이제이션을 활용합니다. " * 10,
            "database": "관계형 데이터베이스는 테이블 구조로 데이터를 저장합니다. SQL을 사용하여 쿼리합니다. " * 10,
            "web": "웹 개발에는 프론트엔드와 백엔드 기술이 필요합니다. React, Node.js 등을 활용합니다. " * 10
        }

        document_ids = {}
        for topic, content in topics_and_content.items():
            result = vector_service.process_pdf_text(content, f"{topic}_study.pdf")
            document_ids[topic] = result["document_id"]

        # RAG용 문서별 컨텍스트 추출 시뮬레이션
        rag_contexts = {}

        # 각 문서에서 핵심 개념 검색
        search_queries = {
            "algorithm": ["동적계획법", "메모이제이션", "효율적"],
            "database": ["관계형", "테이블", "SQL"],
            "web": ["프론트엔드", "백엔드", "React"]
        }

        for topic, queries in search_queries.items():
            doc_id = document_ids[topic]
            topic_context = ""

            for query in queries:
                results = vector_service.search_in_document(query, doc_id, top_k=2)
                for result in results:
                    topic_context += result["text"] + " "

            rag_contexts[topic] = {
                "document_id": doc_id,
                "context": topic_context.strip(),
                "context_length": len(topic_context),
                "ready_for_rag": len(topic_context) > 100
            }

        # RAG 준비도 검증
        assert len(rag_contexts) == 3
        for topic, context_info in rag_contexts.items():
            assert context_info["ready_for_rag"] is True
            assert context_info["context_length"] > 100
            assert context_info["document_id"] in document_ids.values()

        print(f"🤖 RAG 컨텍스트 추출 성공:")
        for topic, context_info in rag_contexts.items():
            print(f"   - {topic}: {context_info['context_length']}자 (문서 ID: {context_info['document_id'][:8]}...)")

        print("✅ RAG 컨텍스트 추출 테스트 성공")

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
        assert "document_id" in process_result

        # 실제 개념 검색
        queries = ["동적계획법", "메모이제이션", "최적화"]

        for query in queries:
            search_results = vector_service.search_documents(query, top_k=3)
            assert len(search_results) > 0

            # 유사도가 유효한 범위인지 확인
            for result in search_results:
                assert -1.0 <= result["similarity"] <= 1.0

        print(f"✅ 실제 PDF 워크플로우 성공:")
        print(f"   - 파일: {sample_pdf_path.name}")
        print(f"   - 문서 ID: {process_result['document_id'][:8]}...")
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
        document_ids = []

        for i, text in enumerate(test_texts):
            result = vector_service.process_pdf_text(text, f"benchmark_{i}.pdf")
            assert result["success"] is True
            document_ids.append(result["document_id"])

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
        print(f"   - 문서 ID들: {[doc_id[:8] + '...' for doc_id in document_ids]}")
        print(f"   - 검색: {avg_search_time:.3f}초 (평균)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])