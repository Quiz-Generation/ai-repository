import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict

from src.app.document.func.text import TextFunc
from src.common.vector.base import VectorDocument
from src.common.vector.connect import VectorDBService


class Vector:
    def __init__(
        self,
        logger,
        vector_db: VectorDBService
    ):
        self.logger = logger
        self.vector_db = vector_db
        self.text_func = TextFunc(logger)

    async def generate_file_id(
            self,
            filename: str
        ) -> str:
        """파일별 고유 ID 생성 (퀴즈 생성용)"""
        # 🎯 파일명 기반 + 현재시간 + 짧은 UUID
        self.logger.info(
            f"""
                [파일별 고유 ID 생성 시작]
                "파일명": {filename}
            """
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        unique_id = uuid.uuid4().hex[:6]

        return f"file_{timestamp}_{file_hash}_{unique_id}"

    async def generate_document_id(
            self,
            content: str,
            metadata: Dict[str, Any]
        ) -> str:
        # 🔥 현재시간 + UUID 기반 ID 생성 (파일명 노출 방지)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:12]  # 12자리 UUID

        return f"{timestamp}_{unique_id}"

    async def store_pdf_content(
            self,
            pdf_content: str,
            metadata: Dict[str, Any],
            chunk_size: int = 1000,
            chunk_overlap: int = 200
        ) -> Dict[str, Any]:
        """PDF 내용을 벡터 DB에 저장"""
        try:

            # 🔥 파일별 고유 ID 생성 (한 번만)
            file_id = await self.generate_file_id(
                filename=metadata.get("filename", "unknown")
            )

            # 텍스트 청킹
            chunks = await self.text_func.create_text_chunks(
                text=pdf_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            # 임베딩 생성

            embeddings = self.vector_db.embedding_model.encode(chunks, show_progress_bar=True)

            # VectorDocument 객체들 생성
            vector_documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # 각 청크별 고유 ID (기존 방식 유지)
                doc_id = await self.generate_document_id(
                    content=chunk,
                    metadata=metadata
                )

                # 청크별 메타데이터 추가 (+ file_id 포함)
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "file_id": file_id,  # 🎯 파일별 공통 ID 추가
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "embedding_model": self.vector_db.model_name,
                    "vector_db_type": self.vector_db.current_db_type
                })

                vector_doc = VectorDocument(
                    id=doc_id,
                    content=chunk,
                    embedding=embedding.tolist(),
                    metadata=chunk_metadata
                )
                vector_documents.append(vector_doc)

            # 벡터 DB에 저장

            stored_ids = await self.vector_db.vector_db.add_documents(vector_documents)

            result = {
                "success": True,
                "file_id": file_id,  # 🎯 파일별 단일 ID 반환
                "vector_db_type": self.vector_db.current_db_type,
                "stored_document_count": len(stored_ids),
                "chunk_count": len(chunks),
                "embedding_dimension": len(embeddings[0]),
                "model_name": self.vector_db.model_name,
                "stored_ids": stored_ids[:5]  # 처음 5개 ID만 반환
            }

            self.logger.info(f"벡터화 저장 완료: {len(stored_ids)}개 문서")
            return result

        except Exception as e:
            self.logger.error(
                f"""
                    [ERROR] PDF 벡터화 저장 실패
                    "error_class": {e.__class__.__name__}
                    "error_message": {str(e)}
                """
            )
            return {
                "success": False,
                "error": str(e),
                "vector_db_type": self.vector_db.current_db_type
            }