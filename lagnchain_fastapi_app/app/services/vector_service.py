#!/usr/bin/env python3
"""
벡터 데이터베이스 서비스 (팩토리 패턴)
"""
import hashlib
import numpy as np
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid
from datetime import datetime


class VectorDatabase(ABC):
    """벡터 데이터베이스 인터페이스"""

    @abstractmethod
    def store_document(self, doc_id: str, text: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """문서 저장"""
        pass

    @abstractmethod
    def search_similar(self, query_vector: List[float], top_k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """유사 문서 검색"""
        pass

    @abstractmethod
    def count_documents(self) -> int:
        """저장된 문서 수"""
        pass

    @abstractmethod
    def get_documents_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """특정 소스(파일)의 모든 문서 조회"""
        pass

    @abstractmethod
    def list_document_sources(self) -> List[Dict[str, Any]]:
        """업로드된 문서 소스 목록 조회"""
        pass

    @abstractmethod
    def save_to_disk(self) -> bool:
        """디스크에 저장"""
        pass

    @abstractmethod
    def load_from_disk(self) -> bool:
        """디스크에서 로드"""
        pass


class WeaviateDB(VectorDatabase):
    """Weaviate 벡터 데이터베이스 (파일 지속성 포함)"""

    def __init__(self, data_dir: str = "./vector_data"):
        self.documents = {}
        self.name = "weaviate"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.data_file = self.data_dir / "weaviate_documents.json"

        # 시작시 기존 데이터 로드
        self.load_from_disk()

    def store_document(self, doc_id: str, text: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        try:
            self.documents[doc_id] = {
                "text": text,
                "vector": vector,
                "metadata": metadata
            }
            # 매번 저장하면 성능이 떨어지므로, 일정 간격이나 종료시에만 저장
            return True
        except Exception:
            return False

    def save_to_disk(self) -> bool:
        """데이터를 디스크에 저장"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"벡터 DB 저장 실패: {e}")
            return False

    def load_from_disk(self) -> bool:
        """디스크에서 데이터 로드"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                print(f"벡터 DB 로드 완료: {len(self.documents)}개 문서")
            return True
        except Exception as e:
            print(f"벡터 DB 로드 실패: {e}")
            self.documents = {}
            return False

    def search_similar(self, query_vector: List[float], top_k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []

        for doc_id, doc_data in self.documents.items():
            # 특정 문서 ID로 필터링
            if document_id and not doc_data["metadata"].get("document_id") == document_id:
                continue

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

    def get_documents_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """특정 소스 파일의 모든 문서 조회"""
        docs = []
        for doc_id, doc_data in self.documents.items():
            if doc_data["metadata"].get("source") == source_name:
                docs.append({
                    "doc_id": doc_id,
                    "text": doc_data["text"],
                    "metadata": doc_data["metadata"]
                })
        return docs

    def list_document_sources(self) -> List[Dict[str, Any]]:
        """업로드된 문서 소스 목록 조회"""
        sources = {}
        for doc_id, doc_data in self.documents.items():
            source = doc_data["metadata"].get("source")
            document_id = doc_data["metadata"].get("document_id")

            if source and document_id:
                if document_id not in sources:
                    sources[document_id] = {
                        "document_id": document_id,
                        "source_filename": source,
                        "chunk_count": 0,
                        "upload_timestamp": doc_data["metadata"].get("upload_timestamp", ""),
                        "total_chars": 0
                    }
                sources[document_id]["chunk_count"] += 1
                sources[document_id]["total_chars"] += len(doc_data["text"])

        return list(sources.values())


class ChromaDB(VectorDatabase):
    """Chroma 벡터 데이터베이스 (파일 지속성 포함)"""

    def __init__(self, data_dir: str = "./vector_data"):
        self.documents = {}
        self.name = "chroma"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.data_file = self.data_dir / "chroma_documents.json"

        # 시작시 기존 데이터 로드
        self.load_from_disk()

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

    def save_to_disk(self) -> bool:
        """데이터를 디스크에 저장"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"벡터 DB 저장 실패: {e}")
            return False

    def load_from_disk(self) -> bool:
        """디스크에서 데이터 로드"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                print(f"벡터 DB 로드 완료: {len(self.documents)}개 문서")
            return True
        except Exception as e:
            print(f"벡터 DB 로드 실패: {e}")
            self.documents = {}
            return False

    def search_similar(self, query_vector: List[float], top_k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []

        for doc_id, doc_data in self.documents.items():
            # 특정 문서 ID로 필터링
            if document_id and not doc_data["metadata"].get("document_id") == document_id:
                continue

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

    def get_documents_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """특정 소스 파일의 모든 문서 조회"""
        docs = []
        for doc_id, doc_data in self.documents.items():
            if doc_data["metadata"].get("source") == source_name:
                docs.append({
                    "doc_id": doc_id,
                    "text": doc_data["text"],
                    "metadata": doc_data["metadata"]
                })
        return docs

    def list_document_sources(self) -> List[Dict[str, Any]]:
        """업로드된 문서 소스 목록 조회"""
        sources = {}
        for doc_id, doc_data in self.documents.items():
            source = doc_data["metadata"].get("source")
            document_id = doc_data["metadata"].get("document_id")

            if source and document_id:
                if document_id not in sources:
                    sources[document_id] = {
                        "document_id": document_id,
                        "source_filename": source,
                        "chunk_count": 0,
                        "upload_timestamp": doc_data["metadata"].get("upload_timestamp", ""),
                        "total_chars": 0
                    }
                sources[document_id]["chunk_count"] += 1
                sources[document_id]["total_chars"] += len(doc_data["text"])

        return list(sources.values())


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
    """PDF → 벡터 저장 메인 서비스 (싱글톤 패턴)"""

    _instance = None
    _initialized = False

    def __new__(cls, db_type: str = "weaviate"):
        if cls._instance is None:
            cls._instance = super(PDFVectorService, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_type: str = "weaviate"):
        # 이미 초기화되었다면 다시 초기화하지 않음
        if PDFVectorService._initialized:
            return

        self.vector_db = VectorDBFactory.create_vector_db(db_type)
        self.embedder = TextEmbedder()
        self.chunker = TextChunker()
        self.db_type = db_type
        PDFVectorService._initialized = True
        print(f"벡터 서비스 초기화: {db_type} DB, 기존 {self.vector_db.count_documents()}개 문서 로드됨")

    def process_pdf_text(self, pdf_text: str, source_name: str) -> Dict[str, Any]:
        """PDF 텍스트를 처리하여 벡터 DB에 저장"""

        # 고유 문서 ID 생성
        document_id = str(uuid.uuid4())

        # 텍스트 청킹
        chunks = self.chunker.chunk_text(pdf_text)

        stored_count = 0
        failed_count = 0

        # 각 청크를 벡터화하여 저장
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 10:  # 너무 짧은 청크는 건너뛰기
                continue

            # 청크 ID 생성
            chunk_id = f"{document_id}_chunk_{i}"

            # 벡터 임베딩
            vector = self.embedder.embed_text(chunk)

            # 메타데이터 생성
            metadata = {
                "document_id": document_id,
                "source": source_name,
                "chunk_index": i,
                "chunk_id": chunk_id,
                "upload_timestamp": datetime.now().isoformat(),
                "text_length": len(chunk)
            }

            # 벡터 DB에 저장
            if self.vector_db.store_document(chunk_id, chunk, vector, metadata):
                stored_count += 1
            else:
                failed_count += 1

        # 🔥 중요: 저장 후 디스크에 영구 저장
        self.vector_db.save_to_disk()

        return {
            "document_id": document_id,
            "source_filename": source_name,
            "total_chunks": len(chunks),
            "stored_chunks": stored_count,
            "failed_chunks": failed_count,
            "text_length": len(pdf_text)
        }

    def search_documents(self, query: str, top_k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """쿼리로 문서 검색 (특정 문서 ID로 필터링 가능)"""
        query_vector = self.embedder.embed_text(query)
        return self.vector_db.search_similar(query_vector, top_k, document_id)

    def search_in_document(self, query: str, document_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """특정 문서 내에서만 검색"""
        return self.search_documents(query, top_k, document_id)

    def get_document_list(self) -> List[Dict[str, Any]]:
        """업로드된 문서 목록 조회"""
        return self.vector_db.list_document_sources()

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """특정 문서 정보 조회"""
        sources = self.get_document_list()
        for source in sources:
            if source["document_id"] == document_id:
                return source
        return None

    def get_stats(self) -> Dict[str, Any]:
        """벡터 DB 통계"""
        document_sources = self.get_document_list()
        return {
            "total_documents": self.vector_db.count_documents(),
            "total_uploaded_files": len(document_sources),
            "db_type": self.db_type,
            "supported_dbs": VectorDBFactory.get_supported_types(),
            "uploaded_files": document_sources
        }

    def switch_database(self, new_db_type: str) -> bool:
        """벡터 DB 변경 (팩토리 패턴 활용)"""
        try:
            # 기존 데이터 저장
            self.vector_db.save_to_disk()

            # 새 DB로 전환
            self.vector_db = VectorDBFactory.create_vector_db(new_db_type)
            self.db_type = new_db_type
            return True
        except ValueError:
            return False

    def force_save(self) -> bool:
        """강제로 디스크에 저장"""
        return self.vector_db.save_to_disk()


# 전역 싱글톤 인스턴스
_global_vector_service = None

def get_global_vector_service() -> PDFVectorService:
    """전역 벡터 서비스 반환 (싱글톤)"""
    global _global_vector_service

    if _global_vector_service is None:
        _global_vector_service = PDFVectorService()
        print("✅ 전역 벡터 서비스 생성 완료")

    return _global_vector_service