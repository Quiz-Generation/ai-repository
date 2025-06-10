"""
📄 PyPDF Loader Implementation (3순위)
"""
import logging
from typing import Dict, Any
from fastapi import UploadFile
from .base import PDFLoader, PDFContent, PDFLoaderInfo

logger = logging.getLogger(__name__)


class PyPDFLoader(PDFLoader):
    """PyPDF 로더 구현체 (3순위 - 가벼움)"""

    def __init__(self):
        super().__init__("pypdf")

    async def extract_text_from_file(self, file: UploadFile) -> PDFContent:
        """업로드된 파일에서 텍스트 추출"""
        try:
            # TODO: 실제 PyPDF 구현
            return PDFContent(
                text="PyPDF로 추출된 텍스트 (TODO: 실제 구현 필요)",
                metadata={"loader": "pypdf"},
                page_count=1,
                file_size=file.size or 0
            )
        except Exception as e:
            logger.error(f"❌ PyPDF 텍스트 추출 실패: {e}")
            raise

    async def extract_text_from_path(self, file_path: str) -> PDFContent:
        """파일 경로에서 텍스트 추출"""
        return PDFContent(text="PyPDF 경로 추출 (TODO)", metadata={"loader": "pypdf"}, page_count=1, file_size=0)

    def validate_file(self, file: UploadFile) -> bool:
        """파일 유효성 검증"""
        if not file.filename:
            return False
        return file.filename.lower().endswith('.pdf')

    def get_supported_features(self) -> list[str]:
        """지원하는 기능 목록"""
        return ["기본 텍스트 추출", "메타데이터 추출", "가벼운 처리"]

    def get_loader_info(self) -> PDFLoaderInfo:
        """로더 정보 반환"""
        return PDFLoaderInfo(
            name="PyPDF",
            description="가벼운 Python PDF 처리 라이브러리",
            priority=3,
            pros=["🪶 가벼움", "⚡ 빠른 설치", "🔧 간단한 사용"],
            cons=["📄 제한적 기능", "🔍 정확도 낮음"],
            best_for="간단한 PDF 처리, 작은 파일",
            supported_features=self.get_supported_features()
        )

    async def health_check(self) -> Dict[str, Any]:
        """PyPDF 헬스체크"""
        return {"status": "healthy", "loader": "pypdf", "priority": 3}