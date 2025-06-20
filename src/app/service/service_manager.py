"""
🏗️ Service Manager - 전역 서비스 관리
"""
import logging
from typing import Optional
from .vector_db_service import VectorDBService
from .document_service import DocumentService

logger = logging.getLogger(__name__)


class ServiceManager:
    """전역 서비스 매니저 - 싱글톤 패턴"""

    _instance = None
    _vector_service: Optional[VectorDBService] = None
    _document_service: Optional[DocumentService] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def initialize(cls) -> None:
        """앱 시작 시 한 번만 초기화"""
        if cls._initialized:
            logger.info("✅ ServiceManager 이미 초기화됨")
            return

        logger.info("🚀 ServiceManager 초기화 시작")

        try:
            # 벡터 DB 서비스 초기화
            cls._vector_service = VectorDBService()
            await cls._vector_service.initialize_embedding_model()
            await cls._vector_service.initialize_vector_db("milvus")

            # 문서 서비스 초기화
            cls._document_service = DocumentService()

            cls._initialized = True
            logger.info("🎉 ServiceManager 초기화 완료")

        except Exception as e:
            logger.error(f"ERROR ServiceManager 초기화 실패: {e}")
            raise

    @classmethod
    def get_vector_service(cls) -> VectorDBService:
        """벡터 DB 서비스 반환"""
        if not cls._initialized:
            raise RuntimeError("ServiceManager가 초기화되지 않았습니다")
        return cls._vector_service

    @classmethod
    def get_document_service(cls) -> DocumentService:
        """문서 서비스 반환"""
        if not cls._initialized:
            raise RuntimeError("ServiceManager가 초기화되지 않았습니다")
        return cls._document_service

    @classmethod
    def is_initialized(cls) -> bool:
        """초기화 상태 확인"""
        return cls._initialized