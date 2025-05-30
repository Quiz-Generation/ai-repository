import logging
from abc import ABC, abstractmethod
from typing import Protocol

logger = logging.getLogger(__name__)


class PDFLoaderProtocol(Protocol):
    """PDF 로더 프로토콜"""

    def extract_text(self, pdf_path: str) -> str:
        """PDF에서 텍스트를 추출합니다"""
        ...


class PyPDFLoader:
    """PyPDF2를 사용하는 PDF 로더"""

    def extract_text(self, pdf_path: str) -> str:
        """PyPDF2로 텍스트를 추출합니다"""
        import os
        from PyPDF2 import PdfReader

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {pdf_path}")

        try:
            reader = PdfReader(pdf_path)
            extracted_text = ""

            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += f"\n--- 페이지 {page_num + 1} ---\n"
                        extracted_text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"페이지 {page_num + 1} 텍스트 추출 실패: {str(e)}")
                    continue

            return self._clean_text(extracted_text)

        except Exception as e:
            logger.error(f"PyPDF2 텍스트 추출 실패: {str(e)}")
            raise ValueError(f"PDF 처리 중 오류가 발생했습니다: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """텍스트를 정리합니다"""
        if not text:
            return ""

        import re
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()


class PyMuPDFLoader:
    """PyMuPDF(fitz)를 사용하는 PDF 로더 - 이미지, 테이블 등 더 나은 추출"""

    def extract_text(self, pdf_path: str) -> str:
        """PyMuPDF로 텍스트를 추출합니다"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF가 설치되지 않았습니다. 'pip install PyMuPDF' 실행하세요.")

        import os

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            extracted_text = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # 텍스트 추출
                page_text = page.get_text()

                if page_text.strip():
                    extracted_text += f"\n--- 페이지 {page_num + 1} ---\n"
                    extracted_text += page_text + "\n"

                # 테이블 추출 (추가 기능)
                try:
                    tables = page.find_tables()
                    for table_num, table in enumerate(tables):
                        table_data = table.extract()
                        if table_data:
                            extracted_text += f"\n--- 페이지 {page_num + 1} 테이블 {table_num + 1} ---\n"
                            for row in table_data:
                                extracted_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                except Exception as e:
                    logger.warning(f"페이지 {page_num + 1} 테이블 추출 실패: {str(e)}")

            doc.close()
            return self._clean_text(extracted_text)

        except Exception as e:
            logger.error(f"PyMuPDF 텍스트 추출 실패: {str(e)}")
            raise ValueError(f"PDF 처리 중 오류가 발생했습니다: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """텍스트를 정리합니다"""
        if not text:
            return ""

        import re
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()


class UnstructuredLoader:
    """Unstructured를 사용하는 PDF 로더 - 고급 문서 구조 분석"""

    def extract_text(self, pdf_path: str) -> str:
        """Unstructured로 텍스트를 추출합니다"""
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError:
            raise ImportError("unstructured가 설치되지 않았습니다. 'pip install unstructured[pdf]' 실행하세요.")

        import os

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {pdf_path}")

        try:
            # Unstructured로 PDF 파티셔닝
            elements = partition_pdf(pdf_path)

            extracted_text = ""
            for element in elements:
                element_type = type(element).__name__
                text = str(element)

                if text.strip():
                    extracted_text += f"\n--- {element_type} ---\n"
                    extracted_text += text + "\n"

            return self._clean_text(extracted_text)

        except Exception as e:
            logger.error(f"Unstructured 텍스트 추출 실패: {str(e)}")
            raise ValueError(f"PDF 처리 중 오류가 발생했습니다: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """텍스트를 정리합니다"""
        if not text:
            return ""

        import re
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()


class PDFLoaderFactory:
    """PDF 로더 팩토리"""

    def __init__(self):
        self._loaders = {
            "pypdf": PyPDFLoader,
            "pymupdf": PyMuPDFLoader,
            "unstructured": UnstructuredLoader,
        }

    def create_loader(self, loader_type: str) -> PDFLoaderProtocol:
        """지정된 타입의 PDF 로더를 생성합니다"""
        if loader_type not in self._loaders:
            available_types = ", ".join(self._loaders.keys())
            raise ValueError(f"지원하지 않는 로더 타입입니다: {loader_type}. 사용 가능한 타입: {available_types}")

        loader_class = self._loaders[loader_type]
        return loader_class()

    def get_available_loaders(self) -> list[str]:
        """사용 가능한 로더 타입 목록을 반환합니다"""
        return list(self._loaders.keys())

    def get_recommended_loader(self, use_case: str = "general") -> str:
        """사용 사례에 따른 추천 로더를 반환합니다"""
        recommendations = {
            "general": "pypdf",  # 일반적인 용도
            "tables": "pymupdf",  # 테이블이 많은 문서
            "complex": "unstructured",  # 복잡한 구조의 문서
            "fast": "pypdf",  # 빠른 처리가 필요한 경우
        }

        return recommendations.get(use_case, "pypdf")