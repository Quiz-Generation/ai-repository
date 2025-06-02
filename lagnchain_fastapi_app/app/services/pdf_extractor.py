import os
from abc import ABC, abstractmethod
from typing import Optional


class PDFExtractorInterface(ABC):
    """PDF 추출기 인터페이스"""

    @abstractmethod
    def extract_text(self, pdf_path: str) -> str:
        """PDF에서 텍스트를 추출합니다"""
        pass


class PDFMinerExtractor(PDFExtractorInterface):
    """PDFMiner 추출기 - AutoRAG 실험 1위 (한글 처리 최고 성능)"""

    def extract_text(self, pdf_path: str) -> str:
        """PDFMiner로 텍스트를 추출합니다"""
        # 파일 존재 여부 확인
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        # PDFMiner로 실제 텍스트 추출
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer

            text = ""
            for page_layout in extract_pages(pdf_path):
                page_text = ""
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        page_text += element.get_text()
                text += page_text + "\n"  # 페이지별 구분

            return text.strip()

        except Exception as e:
            raise Exception(f"PDFMiner 텍스트 추출 실패: {str(e)}")


class PDFPlumberExtractor(PDFExtractorInterface):
    """PDFPlumber 추출기 - AutoRAG 실험 2위 (줄바꿈 완벽 보존)"""

    def extract_text(self, pdf_path: str) -> str:
        """PDFPlumber로 텍스트를 추출합니다"""
        # 파일 존재 여부 확인
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        # PDFPlumber로 실제 텍스트 추출
        try:
            import pdfplumber

            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            return text.strip()

        except Exception as e:
            raise Exception(f"PDFPlumber 텍스트 추출 실패: {str(e)}")


class PyMuPDFExtractor(PDFExtractorInterface):
    """PyMuPDF 추출기 - 빠른 처리 속도 (AutoRAG 실험 4위)"""

    def extract_text(self, pdf_path: str) -> str:
        """PyMuPDF로 텍스트를 추출합니다"""
        # 파일 존재 여부 확인
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        # PyMuPDF로 실제 텍스트 추출
        try:
            import fitz  # PyMuPDF

            text = ""
            doc = fitz.open(pdf_path)

            for page in doc:
                text += page.get_text() + "\n"

            doc.close()
            return text.strip()

        except Exception as e:
            raise Exception(f"PyMuPDF 텍스트 추출 실패: {str(e)}")


class PDFExtractorFactory:
    """PDF 추출기 팩토리 - AutoRAG 실험 결과 기반"""

    _extractors = {
        "pdfminer": PDFMinerExtractor,      # 1위: 한글 처리 최고 성능
        "pdfplumber": PDFPlumberExtractor,  # 2위: 줄바꿈 완벽 보존
        "pymupdf": PyMuPDFExtractor,        # 3위: 빠른 처리 속도
    }

    _default = "pdfminer"  # AutoRAG 실험 1위를 기본값으로

    @classmethod
    def create(cls, extractor_type: Optional[str] = None) -> PDFExtractorInterface:
        """PDF 추출기를 생성합니다"""
        if extractor_type is None:
            extractor_type = cls._default

        if extractor_type not in cls._extractors:
            raise ValueError(f"지원하지 않는 추출기: {extractor_type}. 사용 가능: {list(cls._extractors.keys())}")

        return cls._extractors[extractor_type]()

    @classmethod
    def get_available_extractors(cls) -> list:
        """사용 가능한 추출기 목록"""
        return list(cls._extractors.keys())

    @classmethod
    def get_extractor_info(cls) -> dict:
        """추출기 정보"""
        return {
            "pdfminer": "🥇 1위: 한글 처리 최고 성능, 띄어쓰기 완벽",
            "pdfplumber": "🥈 2위: 줄바꿈과 문단 구조 완벽 보존",
            "pymupdf": "🥉 3위: 빠른 처리 속도"
        }


# 기존 호환성을 위한 PDFExtractor (PDFMiner 기본값)
class PDFExtractor(PDFMinerExtractor):
    """기존 호환성을 위한 PDFExtractor - PDFMiner 기반"""
    pass