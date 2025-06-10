"""
🗄️ Vector Repository
"""
from typing import List, Dict, Any, Optional
from ..core.vector_db.base import VectorDocument, SearchResult
from ..core.vector_db.factory import VectorDBFactory
from ..core.config import settings


class VectorRepository:
    """벡터 데이터 접근 계층"""

    def __init__(self):
        self.vector_db = VectorDBFactory.create(
            settings.VECTOR_DB_TYPE,
            settings.VECTOR_DB_PATH
        )

    async def initialize(self) -> None:
        """벡터 DB 초기화"""
        await self.vector_db.initialize()

    async def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """벡터 문서들 추가"""
        return await self.vector_db.add_documents(documents)

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """벡터 검색"""
        return await self.vector_db.search(query_embedding, top_k, filters)

    async def delete_document(self, document_id: str) -> bool:
        """벡터 문서 삭제"""
        return await self.vector_db.delete_document(document_id)

    async def get_document_count(self) -> int:
        """총 문서 수 조회"""
        return await self.vector_db.get_document_count()

    async def health_check(self) -> Dict[str, Any]:
        """벡터 DB 헬스체크"""
        return await self.vector_db.health_check()