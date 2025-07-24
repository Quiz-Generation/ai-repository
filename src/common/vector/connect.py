"""
🗄️ Vector Database Service
"""
import logging
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from src.common.vector.factory import VectorDBFactory
from src.common.vector.base import  VectorDocument, SearchResult
from src.app.func.text_helper import TextHelper

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
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"임베딩 모델 로드 완료: {self.model_name}")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            raise

    async def initialize_vector_db(self, preferred_db: Optional[str] = None) -> str:
        if self.vector_db and self.current_db_type:
            try:
                health_status = await self.vector_db.health_check()
                if health_status.get("status") == "healthy":
                    logger.info(f"기존 {self.current_db_type} DB 재사용")
                    return self.current_db_type
            except Exception as e:
                logger.warning(f"기존 DB 헬스체크 실패, 재초기화: {e} {self.current_db_type}")

        db_types_to_try = [preferred_db] if preferred_db else self.fallback_order

        for db_type in db_types_to_try:
            if db_type is None:
                continue

            try:
                logger.info(f"{db_type.upper()} 초기화 시도")

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
                logger.info(f"{db_type.upper()} 초기화 및 활성화 완료")
                return db_type

            except Exception as e:
                logger.warning(f"{db_type.upper()} 초기화 실패: {e}")
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

            # 🔥 파일별 고유 ID 생성 (한 번만)
            file_id = self._generate_file_id(metadata.get("filename", "unknown"))

            # 텍스트 청킹
            chunks = TextHelper.create_text_chunks(
                pdf_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            # 임베딩 생성
            logger.info("STEP_VECTOR 임베딩 생성 시작")
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)

            # VectorDocument 객체들 생성
            vector_documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # 각 청크별 고유 ID (기존 방식 유지)
                doc_id = self._generate_document_id(chunk, metadata)

                # 청크별 메타데이터 추가 (+ file_id 포함)
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "file_id": file_id,  # 🎯 파일별 공통 ID 추가
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
                "file_id": file_id,  # 🎯 파일별 단일 ID 반환
                "vector_db_type": self.current_db_type,
                "stored_document_count": len(stored_ids),
                "chunk_count": len(chunks),
                "embedding_dimension": len(embeddings[0]),
                "model_name": self.model_name,
                "stored_ids": stored_ids[:5]  # 처음 5개 ID만 반환
            }

            return result

        except Exception as e:
            logger.error(f"ERROR PDF 벡터화 저장 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "vector_db_type": self.current_db_type
            }

    def _generate_file_id(self, filename: str) -> str:
        """파일별 고유 ID 생성 (퀴즈 생성용)"""
        # 🎯 파일명 기반 + 현재시간 + 짧은 UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        unique_id = uuid.uuid4().hex[:6]

        return f"file_{timestamp}_{file_hash}_{unique_id}"

    def _generate_document_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """문서 ID 생성 (현재시간 + UUID)"""
        # 🔥 현재시간 + UUID 기반 ID 생성 (파일명 노출 방지)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:12]  # 12자리 UUID

        return f"{timestamp}_{unique_id}"

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
            # 🔥 현재 DB가 없으면 자동으로 Milvus 초기화
            if not self.current_db_type or not self.vector_db:
                logger.info("STEP_AUTO Milvus 자동 초기화 시작 (기본값)")
                try:
                    await self.initialize_vector_db("milvus")
                except Exception as e:
                    logger.warning(f"WARNING Milvus 자동 초기화 실패, FAISS로 폴백: {e}")
                    await self.initialize_vector_db("faiss")

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

    async def force_switch_to_milvus(self) -> None:
        """강제로 Milvus DB로 전환 (기존 상태 무시)"""
        try:
            logger.info("🔥 FORCE Milvus 강제 전환 시작")

            # 기존 연결 정리
            self.vector_db = None
            self.current_db_type = None

            # Milvus 강제 초기화
            db_path = f"data/vector_storage/milvus"
            milvus_db = VectorDBFactory.create("milvus", db_path)

            # 헬스체크 먼저 확인
            health_status = await milvus_db.health_check()
            if health_status.get("status") != "healthy":
                raise Exception(f"Milvus 연결 실패: {health_status.get('error')}")

            # 초기화 및 활성화
            await milvus_db.initialize()
            self.vector_db = milvus_db
            self.current_db_type = "milvus"

            logger.info("강제 전환 완료")

        except Exception as e:
            logger.error(f"ERROR Milvus 강제 전환 실패: {e}")
            # 폴백으로 FAISS 시도
            logger.info("WARNING Milvus 실패, FAISS로 폴백")
            await self.switch_vector_db("faiss")
            raise Exception(f"Milvus 전환 실패, FAISS로 폴백됨: {e}")

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
        """벡터 DB의 문서 조회 (파일 단위)"""
        try:
            # 벡터 DB 초기화 확인
            if not self.vector_db:
                await self.initialize_vector_db()

            # 🔥 파일 개수 제한 (기본 100개 파일)
            file_limit = limit if limit else 100
            logger.info(f"STEP_VECTOR 파일 조회 시작 (제한: {file_limit}개 파일)")

            # 🔥 충분히 많은 청크를 조회해서 모든 파일을 찾기 위해
            chunk_limit = 10000  # 충분히 큰 수로 설정
            documents = await self.vector_db.get_all_documents(chunk_limit)

            # 🔥 파일별로 그룹화 (file_id 기준)
            files_info = {}
            for doc in documents:
                filename = doc.metadata.get("filename", "unknown")
                file_id = doc.metadata.get("file_id", "unknown")

                if file_id not in files_info:
                    files_info[file_id] = {
                        "filename": filename,
                        "file_id": file_id,
                        "document_count": 0,
                        "total_chunks": 0,
                        "file_size": doc.metadata.get("file_size", 0),
                        "upload_timestamp": doc.metadata.get("upload_timestamp", "unknown"),
                        "pdf_loader": doc.metadata.get("pdf_loader", "unknown"),
                        "language": doc.metadata.get("language", "unknown"),
                        "vector_db_type": doc.metadata.get("vector_db_type", self.current_db_type),
                        "first_chunk_content": ""
                    }

                files_info[file_id]["document_count"] += 1
                files_info[file_id]["total_chunks"] = doc.metadata.get("total_chunks", 0)

                # 첫 번째 청크의 내용 저장 (미리보기용)
                if files_info[file_id]["first_chunk_content"] == "":
                    content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                    files_info[file_id]["first_chunk_content"] = content_preview

            # 🔥 파일 리스트를 최신순으로 정렬 (upload_timestamp 기준)
            sorted_files = sorted(
                files_info.values(),
                key=lambda x: x.get("upload_timestamp", ""),
                reverse=True
            )

            # 🔥 파일 개수만큼 제한
            limited_files = sorted_files[:file_limit]

            # 전체 청크 수 계산
            total_chunks = sum(doc.metadata.get("total_chunks", 1) for doc in documents)

            result = {
                "success": True,
                "vector_db_type": self.current_db_type,
                "total_documents": total_chunks,  # 전체 청크 수
                "total_files": len(limited_files),  # 실제 반환된 파일 수
                "all_files_count": len(files_info),  # 전체 파일 수
                "limit_applied": file_limit,
                "files": limited_files,
                "embedding_model": self.model_name
            }

            logger.info(f"SUCCESS 파일 조회 완료: {len(limited_files)}개 파일 (전체 {len(files_info)}개 중)")
            return result

        except Exception as e:
            logger.error(f"ERROR 파일 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "vector_db_type": self.current_db_type,
                "total_documents": 0,
                "total_files": 0
            }

    async def clear_all_documents(self, confirm_token: Optional[str] = None) -> Dict[str, Any]:
        """벡터 DB의 모든 데이터 삭제 (위험한 작업)"""
        try:
            # 안전 확인 토큰 체크
            if confirm_token != "CLEAR_ALL_CONFIRM":
                return {
                    "success": False,
                    "error": "삭제 확인 토큰이 필요합니다: CLEAR_ALL_CONFIRM",
                    "vector_db_type": self.current_db_type or "unknown"
                }

            # 벡터 DB 초기화 확인
            if not self.vector_db:
                await self.initialize_vector_db()

            # 벡터 DB와 타입이 초기화되었는지 재확인
            if not self.vector_db or not self.current_db_type:
                return {
                    "success": False,
                    "error": "벡터 DB 초기화 실패",
                    "vector_db_type": "unknown"
                }

            logger.info("🚨 DANGER 모든 벡터 데이터 삭제 시작")

            # 삭제 전 현재 상태 확인
            current_count = await self.vector_db.get_document_count()
            logger.info(f"STEP_DELETE 삭제 예정 문서 수: {current_count}개")

            # 벡터 DB 타입별 전체 삭제 처리
            if hasattr(self.vector_db, 'clear_all'):
                # 전용 메서드가 있는 경우
                success = await self.vector_db.clear_all()
            else:
                # 전용 메서드가 없는 경우 - 모든 문서 개별 삭제
                logger.info("STEP_DELETE 개별 문서 삭제 방식으로 처리")

                # 모든 문서 조회 (제한 없이)
                all_documents = await self.vector_db.get_all_documents(limit=None)

                deleted_count = 0
                for doc in all_documents:
                    try:
                        delete_success = await self.vector_db.delete_document(doc.id)
                        if delete_success:
                            deleted_count += 1
                    except Exception as e:
                        logger.warning(f"WARNING 문서 삭제 실패 (ID: {doc.id}): {e}")
                        continue

                success = deleted_count > 0
                logger.info(f"STEP_DELETE 개별 삭제 완료: {deleted_count}개 문서")

            # 삭제 후 상태 확인
            final_count = await self.vector_db.get_document_count()

            if success:
                logger.info("🎉 SUCCESS 모든 벡터 데이터 삭제 완료")
                return {
                    "success": True,
                    "message": "모든 벡터 데이터 삭제 완료",
                    "vector_db_type": self.current_db_type,
                    "deleted_count": current_count - final_count,
                    "remaining_count": final_count
                }
            else:
                return {
                    "success": False,
                    "error": "데이터 삭제 중 오류 발생",
                    "vector_db_type": self.current_db_type,
                    "remaining_count": final_count
                }

        except Exception as e:
            logger.error(f"ERROR 벡터 데이터 삭제 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "vector_db_type": self.current_db_type or "unknown"
            }