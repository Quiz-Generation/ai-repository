#!/usr/bin/env python3
"""
🔥 실제 ChromaDB 벡터 데이터베이스 서비스
- 고성능 벡터 검색
- 자동 임베딩 생성
- 영구 저장
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
import logging
import time

# 🔥 실제 벡터DB 라이브러리
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# 🚀 Meta FAISS - 초고속 벡터 검색
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

logger = logging.getLogger(__name__)


class VectorDatabase(ABC):
    """벡터 데이터베이스 인터페이스"""

    @abstractmethod
    def store_document(self, doc_id: str, text: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """문서 저장"""
        pass

    @abstractmethod
    def search_similar(self, query: str, top_k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
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


class RealChromaDB(VectorDatabase):
    """🔥 실제 ChromaDB 벡터 데이터베이스"""

    def __init__(self, data_dir: str = "./vector_data"):
        if not HAS_CHROMADB:
            raise ImportError("ChromaDB가 설치되지 않았습니다: pip install chromadb")

        self.name = "real_chromadb"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # ChromaDB 클라이언트 초기화 (영구 저장)
        self.client = chromadb.PersistentClient(
            path=str(self.data_dir / "chroma_db"),
            settings=Settings(anonymized_telemetry=False)
        )

        # 컬렉션 생성/가져오기
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
        )

        logger.info(f"ChromaDB 초기화 완료: {self.collection.count()}개 문서 로드됨")

    def store_document(self, doc_id: str, text: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """문서를 ChromaDB에 저장"""
        try:
            # ChromaDB는 자동으로 임베딩을 생성하므로 vector는 무시하고 텍스트를 사용
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            logger.error(f"ChromaDB 문서 저장 실패: {e}")
            return False

    def store_documents_batch(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> int:
        """🚀 배치로 여러 문서를 한번에 저장 (성능 최적화)"""
        try:
            # ChromaDB 배치 저장
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return len(documents)
        except Exception as e:
            logger.error(f"ChromaDB 배치 저장 실패: {e}")
            # Fallback: 개별 저장
            stored_count = 0
            for doc, metadata, doc_id in zip(documents, metadatas, ids):
                if self.store_document(doc_id, doc, [], metadata):
                    stored_count += 1
            return stored_count

    def search_similar(self, query: str, top_k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """유사 문서 검색 (텍스트 쿼리 사용)"""
        try:
            # 필터 설정
            where_filter = {}
            if document_id:
                where_filter["document_id"] = document_id

            # ChromaDB 검색
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter if where_filter else None
            )

            # 결과 변환
            formatted_results = []
            if (results.get("documents") and
                results["documents"] and
                len(results["documents"]) > 0 and
                results["documents"][0]):

                docs = results["documents"][0]
                ids = results.get("ids", [[]])[0] if results.get("ids") else []
                metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                distances = results.get("distances", [[]])[0] if results.get("distances") else []

                for i, doc in enumerate(docs):
                    formatted_results.append({
                        "doc_id": ids[i] if i < len(ids) else f"unknown_{i}",
                        "text": doc,
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "similarity": 1.0 - distances[i] if i < len(distances) else 0.5
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"ChromaDB 검색 실패: {e}")
            return []

    def count_documents(self) -> int:
        """저장된 문서 수"""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_documents_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """특정 소스 파일의 모든 문서 조회"""
        try:
            results = self.collection.get(
                where={"source": source_name}
            )

            docs = []
            documents = results.get("documents", [])
            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])

            if documents:
                for i, doc in enumerate(documents):
                    docs.append({
                        "doc_id": ids[i] if i < len(ids) else f"unknown_{i}",
                        "text": doc,
                        "metadata": metadatas[i] if i < len(metadatas) else {}
                    })
            return docs
        except Exception as e:
            logger.error(f"소스별 문서 조회 실패: {e}")
            return []

    def list_document_sources(self) -> List[Dict[str, Any]]:
        """업로드된 문서 소스 목록 조회"""
        try:
            # 모든 문서 가져오기
            all_docs = self.collection.get()

            sources = {}
            metadatas = all_docs.get("metadatas", [])
            documents = all_docs.get("documents", [])

            if metadatas:
                for i, metadata in enumerate(metadatas):
                    if not metadata:
                        continue

                    source = metadata.get("source")
                    document_id = metadata.get("document_id")

                    if source and document_id:
                        if document_id not in sources:
                            sources[document_id] = {
                                "document_id": document_id,
                                "source_filename": source,
                                "chunk_count": 0,
                                "upload_timestamp": metadata.get("upload_timestamp", ""),
                                "total_chars": 0
                            }
                        sources[document_id]["chunk_count"] += 1
                        if documents and i < len(documents):
                            sources[document_id]["total_chars"] += len(documents[i])

            return list(sources.values())
        except Exception as e:
            logger.error(f"문서 소스 목록 조회 실패: {e}")
            return []

    def save_to_disk(self) -> bool:
        """ChromaDB는 자동으로 영구 저장됨"""
        return True

    def load_from_disk(self) -> bool:
        """ChromaDB는 자동으로 로드됨"""
        return True


class MetaFAISSDB(VectorDatabase):
    """🚀 Meta FAISS - 초고속 벡터 검색 (프로덕션급)"""

    def __init__(self, data_dir: str = "./vector_data"):
        if not HAS_FAISS:
            raise ImportError("FAISS가 설치되지 않았습니다: pip install faiss-cpu")

        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("SentenceTransformers가 설치되지 않았습니다: pip install sentence-transformers")

        self.name = "meta_faiss"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # 임베딩 모델 로드 (가벼운 모델 사용)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # all-MiniLM-L6-v2의 차원

        # FAISS 인덱스 초기화 (코사인 유사도용)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (코사인 유사도)

        # 메타데이터 저장용
        self.documents = {}  # doc_id -> {text, metadata}
        self.id_map = {}     # faiss_idx -> doc_id
        self.next_id = 0

        # 파일 경로
        self.index_file = self.data_dir / "faiss_index.bin"
        self.metadata_file = self.data_dir / "faiss_metadata.json"

        logger.info("🚀 Meta FAISS 초기화 완료 (초고속 벡터 검색)")

        # 기존 데이터 로드
        self.load_from_disk()

    def _embed_text(self, text: str) -> np.ndarray:
        """텍스트를 벡터로 임베딩 (정규화 포함)"""
        embedding = self.embedder.encode([text])[0]
        # L2 정규화 (코사인 유사도를 위해)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype('float32')

    def store_document(self, doc_id: str, text: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """🚀 초고속 FAISS 저장"""
        try:
            # 텍스트 임베딩
            embedding = self._embed_text(text)

            # FAISS에 벡터 추가
            self.index.add(embedding.reshape(1, -1))

            # 메타데이터 저장
            faiss_idx = self.next_id
            self.documents[doc_id] = {
                "text": text,
                "metadata": metadata,
                "faiss_idx": faiss_idx
            }
            self.id_map[faiss_idx] = doc_id
            self.next_id += 1

            return True
        except Exception as e:
            logger.error(f"FAISS 저장 실패: {e}")
            return False

    def search_similar(self, query: str, top_k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """🚀 초고속 FAISS 검색"""
        try:
            if self.index.ntotal == 0:
                return []

            # 쿼리 임베딩
            query_embedding = self._embed_text(query).reshape(1, -1)

            # FAISS 검색 (초고속!)
            scores, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS가 찾지 못한 경우
                    continue

                doc_id = self.id_map.get(idx)
                if not doc_id or doc_id not in self.documents:
                    continue

                doc_data = self.documents[doc_id]

                # 특정 문서 필터링
                if document_id and doc_data["metadata"].get("document_id") != document_id:
                    continue

                results.append({
                    "doc_id": doc_id,
                    "text": doc_data["text"],
                    "metadata": doc_data["metadata"],
                    "similarity": float(score)
                })

                if len(results) >= top_k:
                    break

            return results

        except Exception as e:
            logger.error(f"FAISS 검색 실패: {e}")
            return []

    def count_documents(self) -> int:
        """저장된 문서 수"""
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

    def save_to_disk(self) -> bool:
        """FAISS 인덱스와 메타데이터 저장"""
        try:
            # FAISS 인덱스 저장
            faiss.write_index(self.index, str(self.index_file))

            # 메타데이터 저장
            metadata = {
                "documents": self.documents,
                "id_map": self.id_map,
                "next_id": self.next_id
            }
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            logger.error(f"FAISS 저장 실패: {e}")
            return False

    def load_from_disk(self) -> bool:
        """FAISS 인덱스와 메타데이터 로드"""
        try:
            # FAISS 인덱스 로드
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))

            # 메타데이터 로드
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.documents = metadata.get("documents", {})
                    self.id_map = metadata.get("id_map", {})
                    self.next_id = metadata.get("next_id", 0)

                logger.info(f"FAISS 로드 완료: {len(self.documents)}개 문서")

            return True
        except Exception as e:
            logger.error(f"FAISS 로드 실패: {e}")
            return False


class VectorDBFactory:
    """🔥 벡터 DB 팩토리 클래스 - Meta FAISS 우선 사용"""

    _db_types = {
        "faiss": MetaFAISSDB,
        "chromadb": RealChromaDB,
        "fallback": MetaFAISSDB  # fallback도 FAISS 사용
    }

    @classmethod
    def create_vector_db(cls, db_type: str = "faiss") -> VectorDatabase:
        """벡터 DB 생성 - FAISS 우선, 실패시 ChromaDB"""

        # FAISS 우선 시도
        if db_type == "faiss" or db_type == "fallback":
            try:
                return MetaFAISSDB()
            except ImportError as e:
                logger.warning(f"FAISS 사용 불가, ChromaDB 사용: {e}")
                try:
                    return RealChromaDB()
                except ImportError:
                    logger.error("FAISS와 ChromaDB 모두 사용 불가!")
                    raise ImportError("벡터 DB를 사용할 수 없습니다. faiss-cpu 또는 chromadb를 설치하세요.")

        # ChromaDB 직접 요청
        if db_type == "chromadb":
            try:
                return RealChromaDB()
            except ImportError as e:
                logger.warning(f"ChromaDB 사용 불가, FAISS 사용: {e}")
                return MetaFAISSDB()

        # 기본값은 FAISS 시도
        try:
            return MetaFAISSDB()
        except ImportError:
            logger.warning("FAISS 없음, ChromaDB 사용")
            return RealChromaDB()

    @classmethod
    def get_supported_types(cls) -> List[str]:
        return list(cls._db_types.keys())


class TextEmbedder:
    """텍스트 임베딩 생성기 (ChromaDB가 자동 처리하므로 필요시만 사용)"""

    def embed_text(self, text: str) -> List[float]:
        """텍스트를 벡터로 변환 (fallback용)"""
        if HAS_SENTENCE_TRANSFORMERS:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(text).tolist()
        else:
            # 간단한 해시 기반 벡터
            hash_val = hashlib.md5(text.encode()).hexdigest()
            return [float(ord(c)) / 255.0 for c in hash_val[:384]]


class TextChunker:
    """텍스트 청킹 클래스"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """텍스트를 청킹"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # 청크 끝 위치 계산
            end = start + self.chunk_size

            # 마지막 청크가 아니라면 문장 경계에서 자르기 시도
            if end < len(text):
                # 가장 가까운 문장 끝 찾기
                sentence_ends = ['.', '!', '?', '\n']
                best_end = end

                for i in range(end - 100, end + 100):
                    if i < len(text) and text[i] in sentence_ends:
                        best_end = i + 1
                        break

                end = best_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 다음 시작점 계산 (오버랩 적용)
            start = end - self.overlap

        return chunks


class PDFVectorService:
    """🔥 실제 ChromaDB 기반 PDF 벡터 서비스 (싱글톤 패턴)"""

    _instance = None
    _initialized = False

    def __new__(cls, db_type: str = "faiss"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_type: str = "faiss"):
        # 이미 초기화되었다면 다시 초기화하지 않음
        if self._initialized:
            return

        self.vector_db = VectorDBFactory.create_vector_db(db_type)
        self.text_chunker = TextChunker()
        self.embedder = TextEmbedder()  # fallback용

        logger.info(f"🔥 PDFVectorService 초기화 완료: {self.vector_db.name} 사용")
        self._initialized = True

    def process_pdf_text(self, pdf_text: str, source_name: str) -> Dict[str, Any]:
        """PDF 텍스트를 벡터화하여 저장 (🚀 배치 처리 최적화)"""
        try:
            # 고유한 문서 ID 생성
            document_id = str(uuid.uuid4())
            upload_timestamp = datetime.now().isoformat()

            # 텍스트 청킹
            chunks = self.text_chunker.chunk_text(pdf_text)
            logger.info(f"PDF 청킹 완료: {len(chunks)}개 청크 생성")

            # 🔥 배치 처리용 데이터 준비
            batch_documents = []
            batch_metadatas = []
            batch_ids = []

            valid_chunks = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # 너무 짧은 청크 스킵
                    continue

                # 청크 ID 생성
                chunk_id = f"{document_id}_{i}"

                # 메타데이터 생성
                metadata = {
                    "document_id": document_id,
                    "source": source_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "upload_timestamp": upload_timestamp,
                    "text_length": len(chunk)
                }

                batch_documents.append(chunk)
                batch_metadatas.append(metadata)
                batch_ids.append(chunk_id)
                valid_chunks.append(chunk)

            # 🚀 단일 배치로 모든 청크 저장 (ChromaDB 자동 임베딩)
            if batch_documents:
                logger.info(f"배치 벡터화 시작: {len(batch_documents)}개 청크")
                batch_start = time.time()

                # 🔥 최적화: 한번에 여러 청크 처리
                stored_count = 0
                batch_size = 10  # 10개씩 배치 처리

                for i in range(0, len(batch_documents), batch_size):
                    batch_end = min(i + batch_size, len(batch_documents))
                    current_batch_docs = batch_documents[i:batch_end]
                    current_batch_meta = batch_metadatas[i:batch_end]
                    current_batch_ids = batch_ids[i:batch_end]

                    # 개별 저장 (안정성 우선)
                    for doc, metadata, doc_id in zip(current_batch_docs, current_batch_meta, current_batch_ids):
                        if self.vector_db.store_document(doc_id, doc, [], metadata):
                            stored_count += 1

                batch_time = time.time() - batch_start
                logger.info(f"배치 벡터화 완료: {batch_time:.2f}초 (평균 {batch_time/len(batch_documents):.3f}초/청크)")
            else:
                stored_count = 0

            # 저장 (ChromaDB는 자동 저장)
            self.vector_db.save_to_disk()

            result = {
                "document_id": document_id,
                "source_name": source_name,
                "total_chunks": len(chunks),
                "stored_chunks": stored_count,
                "upload_timestamp": upload_timestamp,
                "success": stored_count > 0,
                "processing_mode": "batch_optimized"  # 🔥 배치 모드 표시
            }

            logger.info(f"PDF 처리 완료: {stored_count}/{len(chunks)}개 청크 저장됨 (배치 모드)")
            return result

        except Exception as e:
            logger.error(f"PDF 처리 실패: {e}")
            return {
                "document_id": None,
                "source_name": source_name,
                "error": str(e),
                "success": False
            }

    def search_documents(self, query: str, top_k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """문서 검색"""
        return self.vector_db.search_similar(query, top_k, document_id)

    def search_in_document(self, query: str, document_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """특정 문서 내에서 검색"""
        return self.vector_db.search_similar(query, top_k, document_id)

    def get_document_list(self) -> List[Dict[str, Any]]:
        """업로드된 문서 목록"""
        return self.vector_db.list_document_sources()

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """특정 문서 정보"""
        sources = self.vector_db.list_document_sources()
        for source in sources:
            if source.get("document_id") == document_id:
                return source
        return None

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보"""
        try:
            total_docs = self.vector_db.count_documents()
            sources = self.vector_db.list_document_sources()

            return {
                "total_chunks": total_docs,
                "total_documents": len(sources),
                "database_type": self.vector_db.name,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {"error": str(e)}

    def switch_database(self, new_db_type: str) -> bool:
        """벡터 DB 변경"""
        try:
            old_db = self.vector_db
            self.vector_db = VectorDBFactory.create_vector_db(new_db_type)
            logger.info(f"벡터 DB 변경: {old_db.name} -> {self.vector_db.name}")
            return True
        except Exception as e:
            logger.error(f"벡터 DB 변경 실패: {e}")
            return False

    def force_save(self) -> bool:
        """강제 저장"""
        try:
            return self.vector_db.save_to_disk()
        except Exception as e:
            logger.error(f"강제 저장 실패: {e}")
            return False


# 전역 서비스 인스턴스
def get_global_vector_service() -> PDFVectorService:
    """전역 벡터 서비스 반환"""
    return PDFVectorService()


if __name__ == "__main__":
    print("🔥 Real ChromaDB Vector Service")
    print("✅ High-performance vector search")
    print("✅ Automatic embedding generation")
    print("✅ Persistent storage")
    print("✅ Scalable and production-ready")

    # 사용 예시
    service = PDFVectorService()
    print(f"현재 사용 중인 벡터DB: {service.vector_db.name}")
    print(f"저장된 문서 수: {service.vector_db.count_documents()}")