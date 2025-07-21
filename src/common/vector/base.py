"""
🗄️ Vector Database Abstract Interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class VectorDocument:
    """벡터 문서 모델"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """검색 결과 모델"""
    document: VectorDocument
    score: float
    distance: float


class VectorDatabase(ABC):
    """벡터 데이터베이스 추상 인터페이스"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    @abstractmethod
    async def initialize(self) -> None:
        """데이터베이스 초기화"""
        pass

    @abstractmethod
    async def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """문서들 추가"""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """유사도 검색"""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """문서 삭제"""
        pass

    @abstractmethod
    async def get_document_count(self) -> int:
        """총 문서 수 조회"""
        pass

    @abstractmethod
    async def get_all_documents(self, limit: Optional[int] = None) -> List[VectorDocument]:
        """모든 문서 조회"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """헬스체크"""
        pass