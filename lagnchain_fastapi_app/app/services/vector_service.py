#!/usr/bin/env python3
"""
벡터 데이터베이스 서비스 (팩토리 패턴)
"""
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path


class VectorDatabase(ABC):
    """벡터 데이터베이스 추상 클래스"""

    @abstractmethod
    def store_document(self, doc_id: str, text: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """문서 저장"""
        pass

    @abstractmethod
    def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """유사 문서 검색"""
        pass

    @abstractmethod
    def count_documents(self) -> int:
        """저장된 문서 수"""
        pass


class WeaviateDB(VectorDatabase):
    """Weaviate 벡터 데이터베이스 (추천 선택)"""

    def __init__(self):
        self.documents = {}
        self.name = "weaviate"

    def store_document(self, doc_id: str, text: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        try:
            self.documents[doc_id] = {
                "text": text,
                "vector": vector,
                "metadata": metadata
            }
            return True
        except Exception:
            return False

    def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = []

        for doc_id, doc_data in self.documents.items():
            doc_vector = doc_data["vector"]
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            )

            results.append({
                "doc_id": doc_id,
                "text": doc_data["text"],
                "metadata": doc_data["metadata"],
                "similarity": float(similarity)
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def count_documents(self) -> int:
        return len(self.documents)


class ChromaDB(VectorDatabase):
    """Chroma 벡터 데이터베이스 (개발용)"""

    def __init__(self):
        self.documents = {}
        self.name = "chroma"

    def store_document(self, doc_id: str, text: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        try:
            self.documents[doc_id] = {
                "text": text,
                "vector": vector,
                "metadata": metadata
            }
            return True
        except Exception:
            return False

    def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = []

        for doc_id, doc_data in self.documents.items():
            doc_vector = doc_data["vector"]
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            )

            results.append({
                "doc_id": doc_id,
                "text": doc_data["text"],
                "metadata": doc_data["metadata"],
                "similarity": float(similarity)
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def count_documents(self) -> int:
        return len(self.documents)


class VectorDBFactory:
    """벡터 DB 팩토리 클래스"""

    _db_types = {
        "weaviate": WeaviateDB,
        "chroma": ChromaDB,
    }

    @classmethod
    def create_vector_db(cls, db_type: str = "weaviate") -> VectorDatabase:
        """벡터 DB 인스턴스 생성"""
        if db_type not in cls._db_types:
            raise ValueError(f"지원하지 않는 DB 타입: {db_type}")

        return cls._db_types[db_type]()

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """지원하는 DB 타입 목록"""
        return list(cls._db_types.keys())


class TextEmbedder:
    """텍스트 임베딩 클래스"""

    def embed_text(self, text: str) -> List[float]:
        """텍스트를 384차원 벡터로 변환"""
        text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(text_hash % (2**32))
        return np.random.normal(0, 1, 384).tolist()


class TextChunker:
    """텍스트 청킹 클래스"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if len(chunk) > 50:  # 너무 짧은 청크 제외
                chunks.append(chunk)

            start = end - self.overlap
            if start >= len(text):
                break

        return chunks


class PDFVectorService:
    """PDF → 벡터 저장 메인 서비스"""

    def __init__(self, db_type: str = "weaviate"):
        self.vector_db = VectorDBFactory.create_vector_db(db_type)
        self.embedder = TextEmbedder()
        self.chunker = TextChunker()
        self.db_type = db_type

    def process_pdf_text(self, pdf_text: str, source_name: str) -> Dict[str, Any]:
        """PDF 텍스트를 처리하여 벡터 DB에 저장"""
        # 1. 텍스트 청킹
        chunks = self.chunker.chunk_text(pdf_text)

        if not chunks:
            return {
                "success": False,
                "error": "유효한 청크가 생성되지 않았습니다"
            }

        # 2. 청크별 벡터 저장
        stored_count = 0

        for i, chunk in enumerate(chunks):
            doc_id = f"{source_name}_chunk_{i}"

            # 벡터 생성
            vector = self.embedder.embed_text(chunk)

            # 메타데이터 생성
            metadata = {
                "source": source_name,
                "chunk_index": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks),
                "db_type": self.db_type
            }

            # 벡터 DB에 저장
            if self.vector_db.store_document(doc_id, chunk, vector, metadata):
                stored_count += 1

        return {
            "success": True,
            "total_chunks": len(chunks),
            "stored_chunks": stored_count,
            "db_type": self.db_type,
            "source": source_name
        }

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """쿼리로 문서 검색"""
        query_vector = self.embedder.embed_text(query)
        return self.vector_db.search_similar(query_vector, top_k)

    def get_stats(self) -> Dict[str, Any]:
        """벡터 DB 통계"""
        return {
            "total_documents": self.vector_db.count_documents(),
            "db_type": self.db_type,
            "supported_dbs": VectorDBFactory.get_supported_types()
        }

    def switch_database(self, new_db_type: str) -> bool:
        """벡터 DB 변경 (팩토리 패턴 활용)"""
        try:
            self.vector_db = VectorDBFactory.create_vector_db(new_db_type)
            self.db_type = new_db_type
            return True
        except ValueError:
            return False