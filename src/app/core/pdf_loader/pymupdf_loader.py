"""
📄 PyMuPDF Loader Implementation (1순위)
"""
import logging
import io
from typing import Dict, Any
from fastapi import UploadFile
from .base import PDFLoader, PDFContent, PDFLoaderInfo

logger = logging.getLogger(__name__)


class PyMuPDFLoader(PDFLoader):
    """PyMuPDF (fitz) 로더 구현체 (1순위 - 최고 성능)"""

    def __init__(self):
        super().__init__("pymupdf")

    async def extract_text_from_file(self, file: UploadFile) -> PDFContent:
        """업로드된 파일에서 텍스트 추출"""
        try:
            import fitz  # PyMuPDF

            # 파일 내용 읽기
            file_content = await file.read()

            # PDF 문서 열기
            doc = fitz.open(stream=file_content, filetype="pdf")

            # 🔥 문서 닫기 전에 페이지 수 저장
            page_count = len(doc)

            # 텍스트 추출
            text = ""
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                text += page.get_text()
                text += "\n\n"  # 페이지 구분

            # 메타데이터 추출
            metadata = {
                "title": doc.metadata.get("title", "") if doc.metadata else "",
                "author": doc.metadata.get("author", "") if doc.metadata else "",
                "subject": doc.metadata.get("subject", "") if doc.metadata else "",
                "creator": doc.metadata.get("creator", "") if doc.metadata else "",
                "producer": doc.metadata.get("producer", "") if doc.metadata else "",
                "creation_date": doc.metadata.get("creationDate") if doc.metadata else None,
                "modification_date": doc.metadata.get("modDate") if doc.metadata else None,
                "total_pages": page_count,
                "loader": "pymupdf"
            }

            doc.close()



            return PDFContent(
                text=text.strip(),
                metadata=metadata,
                page_count=page_count,
                file_size=file.size or len(file_content)
            )

        except ImportError:
            logger.error("ERROR PyMuPDF 라이브러리가 설치되지 않았습니다. 'pip install PyMuPDF' 실행하세요.")
            raise ImportError("PyMuPDF 라이브러리가 필요합니다")
        except Exception as e:
            logger.error(f"ERROR PyMuPDF 텍스트 추출 실패: {e}")
            raise

    async def extract_text_from_path(self, file_path: str) -> PDFContent:
        """파일 경로에서 텍스트 추출"""
        try:
            import fitz

            # PDF 문서 열기
            doc = fitz.open(file_path)

            # 🔥 문서 닫기 전에 페이지 수 저장
            page_count = len(doc)

            # 텍스트 추출
            text = ""
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                text += page.get_text()
                text += "\n\n"

            # 메타데이터 추출
            metadata = {
                "title": doc.metadata.get("title", "") if doc.metadata else "",
                "author": doc.metadata.get("author", "") if doc.metadata else "",
                "subject": doc.metadata.get("subject", "") if doc.metadata else "",
                "creator": doc.metadata.get("creator", "") if doc.metadata else "",
                "producer": doc.metadata.get("producer", "") if doc.metadata else "",
                "creation_date": doc.metadata.get("creationDate") if doc.metadata else None,
                "modification_date": doc.metadata.get("modDate") if doc.metadata else None,
                "total_pages": page_count,
                "loader": "pymupdf",
                "file_path": file_path
            }

            file_size = 0
            try:
                import os
                file_size = os.path.getsize(file_path)
            except:
                pass

            doc.close()

            return PDFContent(
                text=text.strip(),
                metadata=metadata,
                page_count=page_count,
                file_size=file_size
            )

        except ImportError:
            logger.error("ERROR PyMuPDF 라이브러리가 설치되지 않았습니다")
            raise
        except Exception as e:
            logger.error("ERROR PyMuPDF 파일 경로 추출 실패: {e}")
            raise

    def validate_file(self, file: UploadFile) -> bool:
        """PDF 파일 유효성 검증"""
        if not file.filename:
            return False

        if not file.filename.lower().endswith('.pdf'):
            return False

        # 파일 크기 검사 (50MB 제한)
        if file.size and file.size > 50 * 1024 * 1024:
            return False

        return True

    def get_supported_features(self) -> list[str]:
        """지원하는 기능 목록"""
        return [
            "고속 텍스트 추출",
            "이미지 추출",
            "메타데이터 추출",
            "폰트 정보",
            "페이지 레이아웃",
            "링크 추출",
            "북마크 추출",
            "페이지별 처리"
        ]

    def get_loader_info(self) -> PDFLoaderInfo:
        """로더 정보 반환"""
        return PDFLoaderInfo(
            name="PyMuPDF (fitz)",
            description="Meta에서 개발한 고성능 PDF 처리 라이브러리",
            priority=1,
            pros=[
                "🚀 최고 속도 성능",
                "📄 정확한 텍스트 추출",
                "🖼️ 이미지 처리 지원",
                "📋 풍부한 메타데이터",
                "💾 메모리 효율적",
                "🔧 안정적인 라이브러리",
                "📖 페이지별 처리 가능"
            ],
            cons=[
                "📦 큰 라이브러리 크기",
                "💰 상업용 라이선스 고려사항",
                "🔧 복잡한 설치 (일부 환경)"
            ],
            best_for="고성능이 필요한 대용량 PDF 처리, 프로덕션 환경",
            supported_features=self.get_supported_features()
        )

    async def health_check(self) -> Dict[str, Any]:
        """PyMuPDF 헬스체크"""
        try:
            import fitz
            version = fitz.version

            return {
                "status": "healthy",
                "loader": "pymupdf",
                "priority": 1,
                "version": version,
                "features": self.get_supported_features(),
                "library_available": True
            }

        except ImportError:
            return {
                "status": "unhealthy",
                "loader": "pymupdf",
                "error": "PyMuPDF 라이브러리가 설치되지 않음",
                "library_available": False
            }
        except Exception as e:
            logger.error(f"❌ PyMuPDF 헬스체크 실패: {e}")
            return {
                "status": "unhealthy",
                "loader": "pymupdf",
                "error": str(e),
                "library_available": False
            }