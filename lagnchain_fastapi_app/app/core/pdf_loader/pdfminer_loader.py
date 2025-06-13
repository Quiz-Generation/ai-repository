"""
📄 PDFMiner Loader Implementation (4순위)
"""
import logging
import io
from typing import Dict, Any
from fastapi import UploadFile
from .base import PDFLoader, PDFContent, PDFLoaderInfo

logger = logging.getLogger(__name__)


class PDFMinerLoader(PDFLoader):
    """PDFMiner 로더 구현체 (4순위 - 정확도)"""

    def __init__(self):
        super().__init__("pdfminer")

    async def extract_text_from_file(self, file: UploadFile) -> PDFContent:
        """업로드된 파일에서 텍스트 추출"""
        try:
            from pdfminer.high_level import extract_text, extract_pages
            from pdfminer.layout import LTTextBox, LTTextLine, LTChar

            # 파일 내용 읽기
            file_content = await file.read()

            # 전체 텍스트 추출
            text = extract_text(io.BytesIO(file_content))

            # 페이지별 상세 분석
            pages = list(extract_pages(io.BytesIO(file_content)))
            page_count = len(pages)

            # 폰트 정보 수집
            fonts_used = set()
            char_count = 0

            formatted_text = ""
            for page_num, page in enumerate(pages):
                formatted_text += f"=== 페이지 {page_num + 1} ===\n"
                page_text = ""

                for element in page:
                    if isinstance(element, LTTextBox):
                        for line in element:
                            if isinstance(line, LTTextLine):
                                line_text = ""
                                for char in line:
                                    if isinstance(char, LTChar):
                                        fonts_used.add(char.fontname)
                                        char_count += 1
                                        line_text += char.get_text()
                                page_text += line_text

                formatted_text += page_text + "\n\n"

            # 메타데이터
            metadata = {
                "total_pages": page_count,
                "loader": "pdfminer",
                "fonts_detected": list(fonts_used),
                "character_count": char_count,
                "text_extraction_method": "detailed_layout_analysis",
                "supports_korean": True
            }

            logger.info(f"SUCCESS PDFMiner로 {page_count}페이지, {len(fonts_used)}개 폰트, {char_count}자 처리 완료")

            # 포맷된 텍스트가 있으면 사용, 없으면 기본 텍스트 사용
            final_text = formatted_text.strip() if formatted_text.strip() else text.strip()

            return PDFContent(
                text=final_text,
                metadata=metadata,
                page_count=page_count,
                file_size=file.size or len(file_content)
            )

        except ImportError:
            logger.error("ERROR pdfminer.six 라이브러리가 설치되지 않았습니다. 'pip install pdfminer.six' 실행하세요.")
            raise ImportError("pdfminer.six 라이브러리가 필요합니다")
        except Exception as e:
            logger.error(f"ERROR PDFMiner 텍스트 추출 실패: {e}")
            raise

    async def extract_text_from_path(self, file_path: str) -> PDFContent:
        """파일 경로에서 텍스트 추출"""
        try:
            from pdfminer.high_level import extract_text, extract_pages
            from pdfminer.layout import LTTextBox, LTTextLine, LTChar

            # 전체 텍스트 추출
            text = extract_text(file_path)

            # 페이지별 상세 분석
            pages = list(extract_pages(file_path))
            page_count = len(pages)

            # 폰트 정보 수집
            fonts_used = set()
            char_count = 0

            formatted_text = ""
            for page_num, page in enumerate(pages):
                formatted_text += f"=== 페이지 {page_num + 1} ===\n"
                page_text = ""

                for element in page:
                    if isinstance(element, LTTextBox):
                        for line in element:
                            if isinstance(line, LTTextLine):
                                line_text = ""
                                for char in line:
                                    if isinstance(char, LTChar):
                                        fonts_used.add(char.fontname)
                                        char_count += 1
                                        line_text += char.get_text()
                                page_text += line_text

                formatted_text += page_text + "\n\n"

            # 메타데이터
            metadata = {
                "total_pages": page_count,
                "loader": "pdfminer",
                "fonts_detected": list(fonts_used),
                "character_count": char_count,
                "text_extraction_method": "detailed_layout_analysis",
                "supports_korean": True,
                "file_path": file_path
            }

            file_size = 0
            try:
                import os
                file_size = os.path.getsize(file_path)
            except:
                pass

            final_text = formatted_text.strip() if formatted_text.strip() else text.strip()

            return PDFContent(
                text=final_text,
                metadata=metadata,
                page_count=page_count,
                file_size=file_size
            )

        except ImportError:
            logger.error("❌ pdfminer.six 라이브러리가 설치되지 않았습니다")
            raise
        except Exception as e:
            logger.error(f"❌ PDFMiner 파일 경로 추출 실패: {e}")
            raise

    def validate_file(self, file: UploadFile) -> bool:
        """파일 유효성 검증"""
        if not file.filename:
            return False
        return file.filename.lower().endswith('.pdf')

    def get_supported_features(self) -> list[str]:
        """지원하는 기능 목록"""
        return [
            "정확한 텍스트 추출",
            "복잡한 레이아웃 처리",
            "한글 처리 (뛰어남)",
            "폰트 정보 분석",
            "문자 단위 분석",
            "레이아웃 구조 분석",
            "다국어 지원"
        ]

    def get_loader_info(self) -> PDFLoaderInfo:
        """로더 정보 반환"""
        return PDFLoaderInfo(
            name="PDFMiner.six",
            description="정확도에 특화된 PDF 텍스트 추출 라이브러리",
            priority=4,
            pros=[
                "🎯 최고 정확도",
                "🔧 복잡한 레이아웃 처리",
                "🌏 뛰어난 다국어 지원",
                "🇰🇷 한글 처리 특화",
                "🔍 상세한 레이아웃 분석",
                "📝 폰트 정보 추출"
            ],
            cons=[
                "🐌 느린 속도",
                "🔧 복잡한 설정",
                "💾 높은 메모리 사용",
                "⚙️ 학습 곡선 존재"
            ],
            best_for="복잡한 레이아웃 PDF, 높은 정확도가 필요한 경우, 한글 PDF, 학술 문서",
            supported_features=self.get_supported_features()
        )

    async def health_check(self) -> Dict[str, Any]:
        """PDFMiner 헬스체크"""
        try:
            import pdfminer

            return {
                "status": "healthy",
                "loader": "pdfminer",
                "priority": 4,
                "version": pdfminer.__version__ if hasattr(pdfminer, '__version__') else "unknown",
                "features": self.get_supported_features(),
                "library_available": True,
                "specialization": "high_accuracy_korean"
            }

        except ImportError:
            return {
                "status": "unhealthy",
                "loader": "pdfminer",
                "error": "pdfminer.six 라이브러리가 설치되지 않음",
                "library_available": False
            }
        except Exception as e:
            logger.error(f"❌ PDFMiner 헬스체크 실패: {e}")
            return {
                "status": "unhealthy",
                "loader": "pdfminer",
                "error": str(e),
                "library_available": False
            }