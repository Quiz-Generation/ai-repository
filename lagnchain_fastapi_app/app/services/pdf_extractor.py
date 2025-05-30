import os
import logging
from typing import Optional
from app.services.pdf_loader_factory import PDFLoaderFactory

logger = logging.getLogger(__name__)


class PDFExtractor:
    """PDF 텍스트 추출기 - 다양한 로더 지원"""

    def __init__(self, loader_type: str = "pypdf"):
        """
        PDF 추출기 초기화

        Args:
            loader_type: 사용할 로더 타입 ("pypdf", "pymupdf", "unstructured")
        """
        self.loader_type = loader_type
        self.factory = PDFLoaderFactory()
        self.loader = self.factory.create_loader(loader_type)

    def extract_text(self, pdf_path: str) -> str:
        """PDF에서 텍스트를 추출합니다"""
        try:
            return self.loader.extract_text(pdf_path)
        except Exception as e:
            logger.error(f"PDF 텍스트 추출 실패 (로더: {self.loader_type}): {str(e)}")
            raise

    def get_loader_info(self) -> dict:
        """현재 사용 중인 로더 정보를 반환합니다"""
        return {
            "loader_type": self.loader_type,
            "loader_class": self.loader.__class__.__name__,
            "available_loaders": self.factory.get_available_loaders()
        }