"""
📄 PDFPlumber Loader Implementation (2순위)
"""
import logging
from typing import Dict, Any
from fastapi import UploadFile
from .base import PDFLoader, PDFContent, PDFLoaderInfo

logger = logging.getLogger(__name__)


class PDFPlumberLoader(PDFLoader):
    """PDFPlumber 로더 구현체 (2순위 - 테이블 특화)"""

    def __init__(self):
        super().__init__("pdfplumber")

    async def extract_text_from_file(self, file: UploadFile) -> PDFContent:
        """업로드된 파일에서 텍스트 추출"""
        try:
            # TODO: 실제 PDFPlumber 구현
            return PDFContent(
                text="PDFPlumber로 추출된 텍스트 (TODO: 실제 구현 필요)",
                metadata={"loader": "pdfplumber"},
                page_count=1,
                file_size=file.size or 0
            )
        except Exception as e:
            logger.error(f"❌ PDFPlumber 텍스트 추출 실패: {e}")
            raise

    async def extract_text_from_path(self, file_path: str) -> PDFContent:
        """파일 경로에서 텍스트 추출"""
        return PDFContent(text="PDFPlumber 경로 추출 (TODO)", metadata={"loader": "pdfplumber"}, page_count=1, file_size=0)

    def validate_file(self, file: UploadFile) -> bool:
        """파일 유효성 검증"""
        if not file.filename:
            return False
        return file.filename.lower().endswith('.pdf')

    def get_supported_features(self) -> list[str]:
        """지원하는 기능 목록"""
        return ["테이블 추출", "텍스트 추출", "레이아웃 분석", "좌표 정보"]

    def get_loader_info(self) -> PDFLoaderInfo:
        """로더 정보 반환"""
        return PDFLoaderInfo(
            name="PDFPlumber",
            description="테이블 추출에 특화된 PDF 처리 라이브러리",
            priority=2,
            pros=["📊 뛰어난 테이블 추출", "📐 정확한 레이아웃", "🔍 세밀한 제어"],
            cons=["🐌 느린 속도", "💾 메모리 사용량 높음"],
            best_for="테이블이 많은 PDF, 정확한 레이아웃 분석이 필요한 경우",
            supported_features=self.get_supported_features()
        )

    async def health_check(self) -> Dict[str, Any]:
        """PDFPlumber 헬스체크"""
        return {"status": "healthy", "loader": "pdfplumber", "priority": 2}