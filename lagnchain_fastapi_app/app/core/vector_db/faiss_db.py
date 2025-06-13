"""
📊 FAISS Vector Database Implementation (3순위)
"""
import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from .base import VectorDatabase, VectorDocument, SearchResult

logger = logging.getLogger(__name__)


class FaissDB(VectorDatabase):
    """FAISS 벡터 데이터베이스 구현체 (3순위 - 로컬)"""

    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.index = None
        self.documents = {}  # id -> VectorDocument 매핑
        self.dimension = 384  # sentence-transformers all-MiniLM-L6-v2 기본 차원
        self.index_path = os.path.join(db_path, "faiss.index")
        self.metadata_path = os.path.join(db_path, "metadata.json")
        self.documents_path = os.path.join(db_path, "documents.pkl")

    async def initialize(self) -> None:
        """FAISS 인덱스 초기화"""
        try:
            import faiss

            # 디렉토리 생성
            os.makedirs(self.db_path, exist_ok=True)

            # 기존 인덱스 로드 시도
            if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
                logger.info("STEP_VECTOR 기존 FAISS 인덱스 로드")
                self.index = faiss.read_index(self.index_path)

                with open(self.documents_path, 'rb') as f:
                    self.documents = pickle.load(f)

                logger.info(f"STEP_VECTOR FAISS 인덱스 로드 완료: {len(self.documents)}개 문서")
            else:
                # 새 인덱스 생성 (L2 거리 기준 Flat 인덱스)
                logger.info("STEP_VECTOR 새 FAISS 인덱스 생성")
                self.index = faiss.IndexFlatL2(self.dimension)

            logger.info("SUCCESS FAISS 초기화 완료")

        except ImportError:
            logger.error("ERROR FAISS 라이브러리가 설치되지 않았습니다. 'pip install faiss-cpu' 실행하세요.")
            raise ImportError("FAISS 라이브러리가 필요합니다")
        except Exception as e:
            logger.error(f"ERROR FAISS 초기화 실패: {e}")
            raise

    async def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """문서들을 FAISS 인덱스에 추가"""
        try:
            if not self.index:
                await self.initialize()

            import faiss

            # 임베딩 벡터 준비
            embeddings = []
            doc_ids = []

            for doc in documents:
                embeddings.append(doc.embedding)
                doc_ids.append(doc.id)
                self.documents[doc.id] = doc

            # numpy 배열로 변환
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # FAISS 인덱스에 추가
            self.index.add(embeddings_array)

            # 디스크에 저장
            await self._save_to_disk()

            logger.info(f"SUCCESS FAISS에 {len(documents)}개 문서 추가 완료")
            return doc_ids

        except Exception as e:
            logger.error(f"ERROR FAISS 문서 추가 실패: {e}")
            raise

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """FAISS에서 유사도 검색"""
        try:
            if not self.index or self.index.ntotal == 0:
                logger.warning("WARNING FAISS 인덱스가 비어있습니다")
                return []

            # 쿼리 임베딩을 numpy 배열로 변환
            query_array = np.array([query_embedding], dtype=np.float32)

            # FAISS 검색 수행
            distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))

            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0:  # 유효한 인덱스인 경우
                    # 문서 ID 찾기 (순서대로 저장되어 있다고 가정)
                    doc_id = list(self.documents.keys())[idx]
                    doc = self.documents[doc_id]

                    # 필터 적용
                    if filters:
                        if not self._apply_filters(doc, filters):
                            continue

                    # 유사도 점수 계산 (거리를 점수로 변환)
                    score = 1.0 / (1.0 + distance)

                    results.append(SearchResult(
                        document=doc,
                        score=score,
                        distance=float(distance)
                    ))

            logger.info(f"SUCCESS FAISS 검색 완료: {len(results)}개 결과")
            return results

        except Exception as e:
            logger.error(f"ERROR FAISS 검색 실패: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """문서 삭제 (FAISS는 직접 삭제 불가능, 재구성 필요)"""
        try:
            if document_id not in self.documents:
                logger.warning(f"WARNING 문서 {document_id}를 찾을 수 없습니다")
                return False

            # 문서 제거
            del self.documents[document_id]

            # 인덱스 재구성
            await self._rebuild_index()

            logger.info(f"SUCCESS 문서 {document_id} 삭제 완료")
            return True

        except Exception as e:
            logger.error(f"ERROR 문서 삭제 실패: {e}")
            return False

    async def get_document_count(self) -> int:
        """총 문서 수 조회"""
        return len(self.documents)

    async def get_all_documents(self, limit: Optional[int] = None) -> List[VectorDocument]:
        """모든 문서 조회"""
        try:
            all_docs = list(self.documents.values())

            if limit:
                all_docs = all_docs[:limit]

            logger.info(f"SUCCESS FAISS에서 {len(all_docs)}개 문서 조회 완료")
            return all_docs

        except Exception as e:
            logger.error(f"ERROR FAISS 문서 조회 실패: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """FAISS 헬스체크"""
        try:
            import faiss

            return {
                "status": "healthy",
                "db_type": "faiss",
                "priority": 3,
                "document_count": len(self.documents),
                "index_size": self.index.ntotal if self.index else 0,
                "dimension": self.dimension,
                "library_available": True,
                "index_type": "IndexFlatL2",
                "storage": "local_disk"
            }

        except ImportError:
            return {
                "status": "unhealthy",
                "db_type": "faiss",
                "error": "FAISS 라이브러리가 설치되지 않음",
                "library_available": False
            }
        except Exception as e:
            logger.error(f"ERROR FAISS 헬스체크 실패: {e}")
            return {
                "status": "unhealthy",
                "db_type": "faiss",
                "error": str(e),
                "library_available": False
            }

    async def _save_to_disk(self) -> None:
        """인덱스와 메타데이터를 디스크에 저장"""
        try:
            import faiss

            # FAISS 인덱스 저장
            faiss.write_index(self.index, self.index_path)

            # 문서 메타데이터 저장
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)

            # 메타데이터 JSON 저장
            metadata = {
                "document_count": len(self.documents),
                "dimension": self.dimension,
                "index_type": "IndexFlatL2"
            }

            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"ERROR FAISS 저장 실패: {e}")
            raise

    async def _rebuild_index(self) -> None:
        """인덱스 재구성 (삭제 후)"""
        try:
            import faiss

            if not self.documents:
                # 문서가 없으면 새 인덱스 생성
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                # 남은 문서들로 인덱스 재구성
                embeddings = [doc.embedding for doc in self.documents.values()]
                embeddings_array = np.array(embeddings, dtype=np.float32)

                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(embeddings_array)

            await self._save_to_disk()

        except Exception as e:
            logger.error(f"ERROR FAISS 인덱스 재구성 실패: {e}")
            raise

    def _apply_filters(self, doc: VectorDocument, filters: Dict[str, Any]) -> bool:
        """필터 조건 확인"""
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            if doc.metadata[key] != value:
                return False
        return True