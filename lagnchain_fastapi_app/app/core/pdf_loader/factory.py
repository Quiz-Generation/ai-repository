"""
🏭 PDF Loader Factory
"""
import logging
from typing import Type, Dict, Any
from .base import PDFLoader, PDFLoaderInfo
from .pymupdf_loader import PyMuPDFLoader
from .pypdf_loader import PyPDFLoader
from .pdfplumber_loader import PDFPlumberLoader
from .pdfminer_loader import PDFMinerLoader
from .unstructured_loader import UnstructuredLoader

logger = logging.getLogger(__name__)


class PDFLoaderFactory:
    """PDF 로더 팩토리"""

    _loaders = {
        "pymupdf": PyMuPDFLoader,        # 1순위 - 최고 성능
        "pdfplumber": PDFPlumberLoader,  # 2순위 - 테이블 특화
        "pypdf": PyPDFLoader,            # 3순위 - 가벼움
        "pdfminer": PDFMinerLoader,      # 4순위 - 정확도
        "unstructured": UnstructuredLoader, # 5순위 - AI 기반
    }

    @classmethod
    def create(cls, loader_type: str) -> PDFLoader:
        """PDF 로더 인스턴스 생성"""
        if loader_type not in cls._loaders:
            logger.warning(f"지원하지 않는 PDF 로더: {loader_type}, pymupdf 사용")
            loader_type = "pymupdf"  # 기본값

        loader_class = cls._loaders[loader_type]
        return loader_class()

    @classmethod
    def get_supported_loaders(cls) -> list[str]:
        """지원하는 로더 목록 (우선순위 순)"""
        return ["pymupdf", "pdfplumber", "pypdf", "pdfminer", "unstructured"]

    @classmethod
    def get_priority_order(cls) -> Dict[str, int]:
        """PDF 로더 우선순위 반환"""
        return {
            "pymupdf": 1,      # 최고 성능
            "pdfplumber": 2,   # 테이블 특화
            "pypdf": 3,        # 가벼움
            "pdfminer": 4,     # 정확도
            "unstructured": 5  # AI 기반
        }

    @classmethod
    def get_recommended_loader(cls) -> str:
        """권장 PDF 로더 반환"""
        return "pymupdf"

    @classmethod
    def get_loader_info(cls, loader_type: str) -> Dict[str, Any]:
        """특정 로더 정보 반환"""
        if loader_type not in cls._loaders:
            return {"error": f"지원하지 않는 로더: {loader_type}"}

        loader = cls.create(loader_type)
        info = loader.get_loader_info()

        return {
            "name": info.name,
            "description": info.description,
            "priority": info.priority,
            "pros": info.pros,
            "cons": info.cons,
            "best_for": info.best_for,
            "supported_features": info.supported_features
        }

    @classmethod
    def get_all_loaders_info(cls) -> Dict[str, Dict[str, Any]]:
        """모든 로더 정보 반환"""
        all_info = {}
        for loader_type in cls.get_supported_loaders():
            all_info[loader_type] = cls.get_loader_info(loader_type)
        return all_info

    @classmethod
    def auto_select_loader(cls, file_characteristics: Dict[str, Any]) -> str:
        """파일 특성에 따라 최적 로더 자동 선택"""
        file_size = file_characteristics.get("file_size", 0)
        has_tables = file_characteristics.get("has_tables", False)
        complexity = file_characteristics.get("complexity", "simple")

        # 테이블이 많은 경우
        if has_tables:
            return "pdfplumber"

        # 복잡한 레이아웃인 경우
        if complexity == "complex":
            return "pdfminer"

        # 매우 큰 파일인 경우 (50MB 이상)
        if file_size > 50 * 1024 * 1024:
            return "pymupdf"

        # 작은 파일인 경우 (1MB 이하)
        if file_size < 1024 * 1024:
            return "pypdf"

        # 기본값: 최고 성능
        return "pymupdf"