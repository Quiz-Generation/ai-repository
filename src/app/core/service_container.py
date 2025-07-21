"""
🔧 Service Container - 간단한 의존성 관리
"""
import logging
from typing import Dict, Any, Optional, Type
from src.app.service.quiz_service import QuizService
from src.app.service.document import DocumentService
from src.app.service.vector_db_service import VectorDBService
from src.app.service.test import TestService

logger = logging.getLogger(__name__)


class ServiceContainer:
    """간단한 서비스 컨테이너 - 싱글톤 패턴"""

    _instance = None
    _services: Dict[str, Any] = {}
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls) -> None:
        """서비스 컨테이너 초기화"""
        if cls._initialized:
            logger.info("✅ ServiceContainer 이미 초기화됨")
            return

        logger.info("🚀 ServiceContainer 초기화 시작")

        # 서비스 인스턴스들을 미리 생성하지 않고, 필요할 때 생성하는 lazy loading 방식
        cls._initialized = True
        logger.info("🎉 ServiceContainer 초기화 완료")

    @classmethod
    def get_quiz_service(cls) -> QuizService:
        """퀴즈 서비스 반환 (lazy loading)"""
        service_key = "quiz_service"

        if service_key not in cls._services:
            import os
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise RuntimeError("OpenAI API 키가 설정되지 않았습니다")

            cls._services[service_key] = QuizService(openai_api_key)
            logger.info("✅ QuizService 인스턴스 생성 완료")

        return cls._services[service_key]

    @classmethod
    def get_document_service(cls) -> DocumentService:
        """문서 서비스 반환 (lazy loading)"""
        service_key = "document_service"

        if service_key not in cls._services:
            cls._services[service_key] = DocumentService()
            logger.info("✅ DocumentService 인스턴스 생성 완료")

        return cls._services[service_key]

    @classmethod
    def get_vector_service(cls) -> VectorDBService:
        """벡터 DB 서비스 반환 (전역 서비스 사용)"""
        service_key = "vector_service"

        if service_key not in cls._services:
            # 전역 서비스 사용
            from ..main import global_vector_service
            if global_vector_service is None:
                raise RuntimeError("전역 벡터 DB 서비스가 초기화되지 않았습니다")

            cls._services[service_key] = global_vector_service
            logger.info("✅ VectorDBService 인스턴스 연결 완료")

        return cls._services[service_key]

    @classmethod
    def get_test_service(cls) -> TestService:
        """테스트 서비스 반환 (lazy loading)"""
        service_key = "test_service"

        if service_key not in cls._services:
            cls._services[service_key] = TestService()
            logger.info("✅ TestService 인스턴스 생성 완료")

        return cls._services[service_key]

    @classmethod
    def clear_all_services(cls) -> None:
        """모든 서비스 인스턴스 정리 (테스트용)"""
        cls._services.clear()
        cls._initialized = False
        logger.info("🧹 모든 서비스 인스턴스 정리 완료")

    @classmethod
    def get_service_status(cls) -> Dict[str, Any]:
        """서비스 상태 정보 반환"""
        return {
            "initialized": cls._initialized,
            "services_count": len(cls._services),
            "services": list(cls._services.keys())
        }