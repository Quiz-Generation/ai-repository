"""
📄 PyPDF Loader Implementation (3순위)
"""
import logging
import io
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
            import PyPDF2

            # 파일 내용 읽기
            file_content = await file.read()

            # PDF 리더 생성
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))

            # 텍스트 추출
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"=== 페이지 {page_num + 1} ===\n"
                    text += page_text + "\n\n"

            # 메타데이터 추출
            metadata = {
                "total_pages": len(pdf_reader.pages),
                "loader": "pypdf"
            }

            # PDF 메타데이터 추가
            try:
                if pdf_reader.metadata:
                    metadata.update({
                        "title": pdf_reader.metadata.get("/Title", ""),
                        "author": pdf_reader.metadata.get("/Author", ""),
                        "subject": pdf_reader.metadata.get("/Subject", ""),
                        "creator": pdf_reader.metadata.get("/Creator", ""),
                        "producer": pdf_reader.metadata.get("/Producer", ""),
                        "creation_date": str(pdf_reader.metadata.get("/CreationDate", "")),
                        "modification_date": str(pdf_reader.metadata.get("/ModDate", ""))
                    })
            except Exception as e:
                logger.warning(f"메타데이터 추출 실패: {e}")

            logger.info(f"SUCCESS PyPDF로 {len(pdf_reader.pages)}페이지 PDF 처리 완료")

            return PDFContent(
                text=text.strip(),
                metadata=metadata,
                page_count=len(pdf_reader.pages),
                file_size=file.size or len(file_content)
            )

        except ImportError:
            logger.error("ERROR PyPDF2 라이브러리가 설치되지 않았습니다. 'pip install PyPDF2' 실행하세요.")
            raise ImportError("PyPDF2 라이브러리가 필요합니다")
        except Exception as e:
            logger.error(f"ERROR PyPDF 텍스트 추출 실패: {e}")
            raise

    async def extract_text_from_path(self, file_path: str) -> PDFContent:
        """파일 경로에서 텍스트 추출"""
        try:
            import PyPDF2

            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # 텍스트 추출
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"=== 페이지 {page_num + 1} ===\n"
                        text += page_text + "\n\n"

                # 메타데이터
                metadata = {
                    "total_pages": len(pdf_reader.pages),
                    "loader": "pypdf",
                    "file_path": file_path
                }

                # PDF 메타데이터 추가
                try:
                    if pdf_reader.metadata:
                        metadata.update({
                            "title": pdf_reader.metadata.get("/Title", ""),
                            "author": pdf_reader.metadata.get("/Author", ""),
                            "subject": pdf_reader.metadata.get("/Subject", ""),
                            "creator": pdf_reader.metadata.get("/Creator", ""),
                            "producer": pdf_reader.metadata.get("/Producer", "")
                        })
                except:
                    pass

                file_size = 0
                try:
                    import os
                    file_size = os.path.getsize(file_path)
                except:
                    pass

                return PDFContent(
                    text=text.strip(),
                    metadata=metadata,
                    page_count=len(pdf_reader.pages),
                    file_size=file_size
                )

        except ImportError:
            logger.error("❌ PyPDF2 라이브러리가 설치되지 않았습니다")
            raise
        except Exception as e:
            logger.error(f"❌ PyPDF 파일 경로 추출 실패: {e}")
            raise

    def validate_file(self, file: UploadFile) -> bool:
        """파일 유효성 검증"""
        if not file.filename:
            return False
        return file.filename.lower().endswith('.pdf')

    def get_supported_features(self) -> list[str]:
        """지원하는 기능 목록"""
        return [
            "기본 텍스트 추출",
            "메타데이터 추출",
            "가벼운 처리",
            "빠른 설치",
            "페이지별 처리"
        ]

    def get_loader_info(self) -> PDFLoaderInfo:
        """로더 정보 반환"""
        return PDFLoaderInfo(
            name="PyPDF2",
            description="가벼운 Python PDF 처리 라이브러리",
            priority=3,
            pros=[
                "🪶 가벼움",
                "⚡ 빠른 설치",
                "🔧 간단한 사용",
                "💾 적은 메모리 사용",
                "📦 의존성 최소"
            ],
            cons=[
                "📄 제한적 기능",
                "🔍 정확도 낮음",
                "📊 테이블 처리 불가",
                "🖼️ 이미지 처리 제한적"
            ],
            best_for="간단한 PDF 처리, 작은 파일, 빠른 프로토타이핑",
            supported_features=self.get_supported_features()
        )

    async def health_check(self) -> Dict[str, Any]:
        """PyPDF 헬스체크"""
        try:
            import PyPDF2

            return {
                "status": "healthy",
                "loader": "pypdf",
                "priority": 3,
                "version": PyPDF2.__version__ if hasattr(PyPDF2, '__version__') else "unknown",
                "features": self.get_supported_features(),
                "library_available": True,
                "specialization": "lightweight_processing"
            }

        except ImportError:
            return {
                "status": "unhealthy",
                "loader": "pypdf",
                "error": "PyPDF2 라이브러리가 설치되지 않음",
                "library_available": False
            }
        except Exception as e:
            logger.error(f"❌ PyPDF 헬스체크 실패: {e}")
            return {
                "status": "unhealthy",
                "loader": "pypdf",
                "error": str(e),
                "library_available": False
            }