"""
🗄️ Vector Database Service
"""
import logging
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

from ..core.vector_db.factory import VectorDBFactory
from ..core.vector_db.base import VectorDatabase, VectorDocument, SearchResult
from ..helper.text_helper import TextHelper

logger = logging.getLogger(__name__)


class VectorDBService:
    """벡터 DB 서비스 - 임베딩 생성 및 저장 관리"""

    def __init__(self):
        self.embedding_model = None
        self.model_name = "all-MiniLM-L6-v2"  # 경량화된 다국어 지원 모델
        self.vector_db = None
        self.current_db_type = None
        self.fallback_order = ["milvus", "faiss"]

    async def initialize_embedding_model(self) -> None:
        """임베딩 모델 초기화"""
        try:
            logger.info("STEP_VECTOR 임베딩 모델 로드 시작")
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"SUCCESS 임베딩 모델 로드 완료: {self.model_name}")
        except Exception as e:
            logger.error(f"ERROR 임베딩 모델 로드 실패: {e}")
            raise

    async def initialize_vector_db(self, preferred_db: Optional[str] = None) -> str:
        """벡터 DB 초기화 (우선순위에 따른 폴백)"""
        db_types_to_try = [preferred_db] if preferred_db else self.fallback_order

        for db_type in db_types_to_try:
            if db_type is None:
                continue

            try:
                logger.info(f"STEP_VECTOR {db_type.upper()} 초기화 시도")

                # 벡터 DB 인스턴스 생성 (올바른 경로 사용)
                db_path = f"data/vector_storage/{db_type}"
                temp_db = VectorDBFactory.create(db_type, db_path)

                # 헬스체크로 먼저 확인
                health_status = await temp_db.health_check()
                if health_status.get("status") != "healthy":
                    logger.warning(f"WARNING {db_type.upper()} 헬스체크 실패: {health_status.get('error', 'Unknown error')}")
                    continue

                # 초기화
                await temp_db.initialize()

                self.vector_db = temp_db
                self.current_db_type = db_type
                logger.info(f"SUCCESS {db_type.upper()} 초기화 및 활성화 완료")
                return db_type

            except Exception as e:
                logger.warning(f"WARNING {db_type.upper()} 초기화 실패: {e}")
                continue

        # 모든 DB 실패 시 마지막으로 FAISS 강제 시도
        if not self.current_db_type:
            try:
                logger.info("STEP_VECTOR 마지막 시도: FAISS 강제 초기화")
                db_path = f"data/vector_storage/faiss"
                self.vector_db = VectorDBFactory.create("faiss", db_path)
                await self.vector_db.initialize()
                self.current_db_type = "faiss"
                logger.info("SUCCESS FAISS 강제 초기화 완료")
                return "faiss"
            except Exception as e:
                logger.error(f"ERROR FAISS 강제 초기화도 실패: {e}")

        raise RuntimeError("모든 벡터 DB 초기화 실패")

    async def store_pdf_content(
        self,
        pdf_content: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Dict[str, Any]:
        """PDF 내용을 벡터 DB에 저장"""
        try:
            # 모델이 초기화되지 않은 경우 초기화
            if not self.embedding_model:
                await self.initialize_embedding_model()

            # 벡터 DB가 초기화되지 않은 경우 초기화
            if not self.vector_db:
                await self.initialize_vector_db()

            logger.info("STEP_VECTOR PDF 내용 청킹 시작")

            # 텍스트 청킹
            chunks = TextHelper.create_text_chunks(
                pdf_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            logger.info(f"STEP_VECTOR {len(chunks)}개 청크 생성 완료")

            # 임베딩 생성
            logger.info("STEP_VECTOR 임베딩 생성 시작")
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)

            # VectorDocument 객체들 생성
            vector_documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc_id = self._generate_document_id(chunk, metadata)

                # 청크별 메타데이터 추가
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "embedding_model": self.model_name,
                    "vector_db_type": self.current_db_type
                })

                vector_doc = VectorDocument(
                    id=doc_id,
                    content=chunk,
                    embedding=embedding.tolist(),
                    metadata=chunk_metadata
                )
                vector_documents.append(vector_doc)

            # 벡터 DB에 저장
            logger.info(f"STEP_VECTOR {self.current_db_type.upper()}에 저장 시작")
            stored_ids = await self.vector_db.add_documents(vector_documents)

            result = {
                "success": True,
                "vector_db_type": self.current_db_type,
                "stored_document_count": len(stored_ids),
                "chunk_count": len(chunks),
                "embedding_dimension": len(embeddings[0]),
                "model_name": self.model_name,
                "stored_ids": stored_ids[:5]  # 처음 5개 ID만 반환
            }

            logger.info(f"SUCCESS PDF 벡터화 저장 완료: {len(stored_ids)}개 문서")
            return result

        except Exception as e:
            logger.error(f"ERROR PDF 벡터화 저장 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "vector_db_type": self.current_db_type
            }

    async def search_similar_content(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """유사한 내용 검색"""
        try:
            # 모델과 DB 초기화 확인
            if not self.embedding_model:
                await self.initialize_embedding_model()
            if not self.vector_db:
                await self.initialize_vector_db()

            logger.info(f"STEP_VECTOR 검색 쿼리: '{query[:50]}...'")

            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query])[0]

            # 벡터 DB에서 검색
            results = await self.vector_db.search(
                query_embedding=query_embedding.tolist(),
                top_k=top_k,
                filters=filters
            )

            logger.info(f"SUCCESS 유사도 검색 완료: {len(results)}개 결과")
            return results

        except Exception as e:
            logger.error(f"ERROR 유사도 검색 실패: {e}")
            return []

    async def get_vector_db_status(self) -> Dict[str, Any]:
        """벡터 DB 상태 정보 조회"""
        try:
            status = {
                "current_db_type": self.current_db_type,
                "embedding_model": self.model_name,
                "supported_db_types": VectorDBFactory.get_supported_types(),
                "priority_order": VectorDBFactory.get_priority_order()
            }

            # 현재 DB 헬스체크
            if self.vector_db:
                health_info = await self.vector_db.health_check()
                status["current_db_health"] = health_info
                status["document_count"] = await self.vector_db.get_document_count()

            # 모든 DB 타입 헬스체크
            db_health_status = {}
            for db_type in self.fallback_order:
                try:
                    temp_db = VectorDBFactory.create(db_type, f"data/vector_storage/{db_type}")
                    health = await temp_db.health_check()
                    db_health_status[db_type] = health
                except Exception as e:
                    db_health_status[db_type] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "db_type": db_type
                    }

            status["all_db_health"] = db_health_status
            return status

        except Exception as e:
            logger.error(f"ERROR 벡터 DB 상태 조회 실패: {e}")
            return {
                "error": str(e),
                "current_db_type": self.current_db_type
            }

    async def switch_vector_db(self, new_db_type: str) -> bool:
        """벡터 DB 타입 변경"""
        try:
            if new_db_type not in VectorDBFactory.get_supported_types():
                raise ValueError(f"지원하지 않는 DB 타입: {new_db_type}")

            logger.info(f"STEP_VECTOR {new_db_type.upper()}로 전환 시도")

            # 새 DB 초기화 (올바른 경로 사용)
            db_path = f"data/vector_storage/{new_db_type}"
            new_db = VectorDBFactory.create(new_db_type, db_path)
            await new_db.initialize()

            # 성공 시 교체
            self.vector_db = new_db
            self.current_db_type = new_db_type

            logger.info(f"SUCCESS {new_db_type.upper()}로 전환 완료")
            return True

        except Exception as e:
            logger.error(f"ERROR 벡터 DB 전환 실패: {e}")
            return False

    def _generate_document_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """문서 ID 생성 (내용 기반 해시)"""
        # 내용과 주요 메타데이터로 고유 ID 생성
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        filename = metadata.get("filename", "unknown")
        chunk_info = f"{metadata.get('chunk_index', 0)}"

        return f"{filename}_{content_hash}_{chunk_info}_{uuid.uuid4().hex[:8]}"

    async def delete_documents_by_filename(self, filename: str) -> Dict[str, Any]:
        """파일명으로 문서들 삭제"""
        try:
            if not self.vector_db:
                await self.initialize_vector_db()

            # 필터로 해당 파일의 문서들 검색
            filter_condition = {"filename": filename}
            documents = await self.vector_db.search(
                query_embedding=[0.0] * 384,  # 더미 임베딩
                top_k=1000,  # 충분히 큰 수
                filters=filter_condition
            )

            # 찾은 문서들 삭제
            deleted_count = 0
            for result in documents:
                success = await self.vector_db.delete_document(result.document.id)
                if success:
                    deleted_count += 1

            logger.info(f"SUCCESS {filename} 관련 {deleted_count}개 문서 삭제 완료")
            return {
                "success": True,
                "deleted_count": deleted_count,
                "filename": filename
            }

        except Exception as e:
            logger.error(f"ERROR 문서 삭제 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }

    async def get_all_documents(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """벡터 DB의 모든 문서 조회"""
        try:
            # 벡터 DB 초기화 확인
            if not self.vector_db:
                await self.initialize_vector_db()

            logger.info(f"STEP_VECTOR 모든 문서 조회 시작 (제한: {limit or '없음'})")

            # 벡터 DB에서 모든 문서 조회
            documents = await self.vector_db.get_all_documents(limit)

            # 파일별 문서 그룹화
            files_info = {}
            for doc in documents:
                filename = doc.metadata.get("filename", "unknown")
                if filename not in files_info:
                    files_info[filename] = {
                        "filename": filename,
                        "document_count": 0,
                        "total_chunks": 0,
                        "file_size": doc.metadata.get("file_size", 0),
                        "upload_timestamp": doc.metadata.get("upload_timestamp", "unknown"),
                        "pdf_loader": doc.metadata.get("pdf_loader", "unknown"),
                        "language": doc.metadata.get("language", "unknown"),
                        "vector_db_type": doc.metadata.get("vector_db_type", self.current_db_type),
                        "first_chunk_content": ""
                    }

                files_info[filename]["document_count"] += 1
                files_info[filename]["total_chunks"] = doc.metadata.get("total_chunks", 0)

                # 첫 번째 청크의 내용 저장 (미리보기용)
                if files_info[filename]["first_chunk_content"] == "":
                    content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                    files_info[filename]["first_chunk_content"] = content_preview

            result = {
                "success": True,
                "vector_db_type": self.current_db_type,
                "total_documents": len(documents),
                "total_files": len(files_info),
                "limit_applied": limit,
                "files": list(files_info.values()),
                "embedding_model": self.model_name
            }

            logger.info(f"SUCCESS 모든 문서 조회 완료: {len(documents)}개 문서, {len(files_info)}개 파일")
            return result

        except Exception as e:
            logger.error(f"ERROR 모든 문서 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "vector_db_type": self.current_db_type,
                "total_documents": 0,
                "total_files": 0
            }