"""
📄 PDFPlumber Loader Implementation (2순위)
"""
import logging
import io
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
            import pdfplumber

            # 파일 내용 읽기
            file_content = await file.read()

            # PDF 열기
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                text = ""
                tables_found = 0

                for page_num, page in enumerate(pdf.pages):
                    # 텍스트 추출
                    page_text = page.extract_text()
                    if page_text:
                        text += f"=== 페이지 {page_num + 1} ===\n"
                        text += page_text + "\n\n"

                    # 테이블 추출
                    tables = page.extract_tables()
                    if tables:
                        tables_found += len(tables)
                        text += f"=== 페이지 {page_num + 1} 테이블 ===\n"
                        for table_idx, table in enumerate(tables):
                            text += f"테이블 {table_idx + 1}:\n"
                            for row in table:
                                if row:
                                    text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                            text += "\n"

                # 메타데이터
                metadata = {
                    "total_pages": len(pdf.pages),
                    "tables_found": tables_found,
                    "loader": "pdfplumber",
                    "has_tables": tables_found > 0
                }

                # PDF 메타데이터 추가 시도
                try:
                    if hasattr(pdf, 'metadata') and pdf.metadata:
                        metadata.update({
                            "title": pdf.metadata.get("Title", ""),
                            "author": pdf.metadata.get("Author", ""),
                            "subject": pdf.metadata.get("Subject", ""),
                            "creator": pdf.metadata.get("Creator", ""),
                            "producer": pdf.metadata.get("Producer", "")
                        })
                except:
                    pass

                logger.info(f"{len(pdf.pages)}페이지, {tables_found}개 테이블 처리 완료")

                return PDFContent(
                    text=text.strip(),
                    metadata=metadata,
                    page_count=len(pdf.pages),
                    file_size=file.size or len(file_content)
                )

        except ImportError:
            logger.error("ERROR pdfplumber 라이브러리가 설치되지 않았습니다. 'pip install pdfplumber' 실행하세요.")
            raise ImportError("pdfplumber 라이브러리가 필요합니다")
        except Exception as e:
            logger.error(f"ERROR PDFPlumber 텍스트 추출 실패: {e}")
            raise

    async def extract_text_from_path(self, file_path: str) -> PDFContent:
        """파일 경로에서 텍스트 추출"""
        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                text = ""
                tables_found = 0

                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"=== 페이지 {page_num + 1} ===\n"
                        text += page_text + "\n\n"

                    tables = page.extract_tables()
                    if tables:
                        tables_found += len(tables)
                        text += f"=== 페이지 {page_num + 1} 테이블 ===\n"
                        for table_idx, table in enumerate(tables):
                            text += f"테이블 {table_idx + 1}:\n"
                            for row in table:
                                if row:
                                    text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                            text += "\n"

                metadata = {
                    "total_pages": len(pdf.pages),
                    "tables_found": tables_found,
                    "loader": "pdfplumber",
                    "has_tables": tables_found > 0,
                    "file_path": file_path
                }

                file_size = 0
                try:
                    import os
                    file_size = os.path.getsize(file_path)
                except:
                    pass

                return PDFContent(
                    text=text.strip(),
                    metadata=metadata,
                    page_count=len(pdf.pages),
                    file_size=file_size
                )

        except ImportError:
            logger.error("❌ pdfplumber 라이브러리가 설치되지 않았습니다")
            raise
        except Exception as e:
            logger.error(f"❌ PDFPlumber 파일 경로 추출 실패: {e}")
            raise

    def validate_file(self, file: UploadFile) -> bool:
        """파일 유효성 검증"""
        if not file.filename:
            return False
        return file.filename.lower().endswith('.pdf')

    def get_supported_features(self) -> list[str]:
        """지원하는 기능 목록"""
        return [
            "테이블 추출 (뛰어남)",
            "텍스트 추출",
            "레이아웃 분석",
            "좌표 정보",
            "페이지별 처리",
            "정확한 테이블 구조 보존"
        ]

    def get_loader_info(self) -> PDFLoaderInfo:
        """로더 정보 반환"""
        return PDFLoaderInfo(
            name="PDFPlumber",
            description="테이블 추출에 특화된 PDF 처리 라이브러리",
            priority=2,
            pros=[
                "📊 뛰어난 테이블 추출",
                "📐 정확한 레이아웃 분석",
                "🔍 세밀한 제어",
                "📋 테이블 구조 보존",
                "📖 페이지별 상세 분석"
            ],
            cons=[
                "🐌 느린 속도",
                "💾 메모리 사용량 높음",
                "🔧 복잡한 PDF에서 성능 저하"
            ],
            best_for="테이블이 많은 PDF, 정확한 레이아웃 분석이 필요한 경우, 데이터 추출",
            supported_features=self.get_supported_features()
        )

    async def health_check(self) -> Dict[str, Any]:
        """PDFPlumber 헬스체크"""
        try:
            import pdfplumber

            return {
                "status": "healthy",
                "loader": "pdfplumber",
                "priority": 2,
                "features": self.get_supported_features(),
                "library_available": True,
                "specialization": "table_extraction"
            }

        except ImportError:
            return {
                "status": "unhealthy",
                "loader": "pdfplumber",
                "error": "pdfplumber 라이브러리가 설치되지 않음",
                "library_available": False
            }
        except Exception as e:
            logger.error(f"❌ PDFPlumber 헬스체크 실패: {e}")
            return {
                "status": "unhealthy",
                "loader": "pdfplumber",
                "error": str(e),
                "library_available": False
            }