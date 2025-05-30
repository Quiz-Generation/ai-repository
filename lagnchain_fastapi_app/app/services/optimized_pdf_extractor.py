"""
최적화된 PDF 추출기
PyMuPDF를 기본으로 하되, 필요시 다른 로더로 교체 가능한 팩토리 패턴
"""
import time
from typing import Dict, Any, Optional
from langchain_community.document_loaders import PyMuPDFLoader
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class PDFExtractorInterface(ABC):
    """PDF 추출기 인터페이스"""

    @abstractmethod
    def extract_text(self, pdf_path: str) -> str:
        """PDF에서 텍스트 추출"""
        pass

    @abstractmethod
    def get_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """PDF 메타데이터 추출"""
        pass


class PyMuPDFExtractor(PDFExtractorInterface):
    """PyMuPDF 기반 최적화된 추출기 (기본 추천)"""

    def extract_text(self, pdf_path: str) -> str:
        """PyMuPDF로 텍스트 추출 - 빠르고 품질 좋음"""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            text = ""

            for page in doc:
                text += page.get_text()

            doc.close()
            return text

        except Exception as e:
            logger.error(f"PyMuPDF 추출 실패: {e}")
            raise

    def get_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """PDF 메타데이터 추출"""
        try:
            import fitz
            import os

            doc = fitz.open(pdf_path)
            metadata = doc.metadata or {}  # None 체크
            doc.close()

            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "file_size": os.path.getsize(pdf_path),
                "page_count": len(fitz.open(pdf_path))
            }

        except Exception as e:
            logger.error(f"메타데이터 추출 실패: {e}")
            return {}


class PyPDF2Extractor(PDFExtractorInterface):
    """PyPDF2 기반 추출기 (호환성용)"""

    def extract_text(self, pdf_path: str) -> str:
        """PyPDF2로 텍스트 추출 - 호환성 좋음"""
        try:
            import PyPDF2

            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page in pdf_reader.pages:
                    text += page.extract_text()

            return text

        except Exception as e:
            logger.error(f"PyPDF2 추출 실패: {e}")
            raise

    def get_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """PDF 메타데이터 추출"""
        try:
            import PyPDF2
            import os

            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata

                return {
                    "title": metadata.get("/Title", "") if metadata else "",
                    "author": metadata.get("/Author", "") if metadata else "",
                    "subject": metadata.get("/Subject", "") if metadata else "",
                    "creator": metadata.get("/Creator", "") if metadata else "",
                    "producer": metadata.get("/Producer", "") if metadata else "",
                    "creation_date": str(metadata.get("/CreationDate", "")) if metadata else "",
                    "modification_date": str(metadata.get("/ModDate", "")) if metadata else "",
                    "file_size": os.path.getsize(pdf_path),
                    "page_count": len(pdf_reader.pages)
                }

        except Exception as e:
            logger.error(f"메타데이터 추출 실패: {e}")
            return {}


class LangChainPyMuPDFExtractor(PDFExtractorInterface):
    """LangChain PyMuPDF 추출기 (기존 호환성용)"""

    def extract_text(self, pdf_path: str) -> str:
        """LangChain PyMuPDF로 텍스트 추출"""
        try:
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
            except ImportError:
                logger.error("langchain_community가 설치되지 않았습니다.")
                raise ImportError("langchain_community 패키지가 필요합니다.")

            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

            return "\n".join([doc.page_content for doc in documents])

        except Exception as e:
            logger.error(f"LangChain PyMuPDF 추출 실패: {e}")
            raise

    def get_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """PDF 메타데이터 추출"""
        try:
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
            except ImportError:
                logger.error("langchain_community가 설치되지 않았습니다.")
                return {}

            import os

            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

            # 첫 번째 문서의 메타데이터 사용
            metadata = documents[0].metadata if documents else {}

            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "file_size": os.path.getsize(pdf_path),
                "page_count": len(documents)
            }

        except Exception as e:
            logger.error(f"메타데이터 추출 실패: {e}")
            return {}


class OptimizedPDFExtractorFactory:
    """최적화된 PDF 추출기 팩토리"""

    _extractors = {
        "pymupdf": PyMuPDFExtractor,
        "pypdf2": PyPDF2Extractor,
        "langchain_pymupdf": LangChainPyMuPDFExtractor,
    }

    # 기본 추출기: PyMuPDF (가장 빠르고 품질 좋음)
    _default_extractor = "pymupdf"

    @classmethod
    def create_extractor(cls, extractor_type: Optional[str] = None) -> PDFExtractorInterface:
        """PDF 추출기 생성"""
        if extractor_type is None:
            extractor_type = cls._default_extractor

        if extractor_type not in cls._extractors:
            logger.warning(f"알 수 없는 추출기 타입: {extractor_type}. 기본값 사용: {cls._default_extractor}")
            extractor_type = cls._default_extractor

        return cls._extractors[extractor_type]()

    @classmethod
    def get_available_extractors(cls) -> list:
        """사용 가능한 추출기 목록"""
        return list(cls._extractors.keys())

    @classmethod
    def get_default_extractor(cls) -> str:
        """기본 추출기 타입"""
        return cls._default_extractor

    @classmethod
    def set_default_extractor(cls, extractor_type: str):
        """기본 추출기 변경"""
        if extractor_type in cls._extractors:
            cls._default_extractor = extractor_type
            logger.info(f"기본 추출기가 {extractor_type}로 변경되었습니다.")
        else:
            logger.error(f"알 수 없는 추출기 타입: {extractor_type}")


class OptimizedPDFService:
    """최적화된 PDF 서비스"""

    def __init__(self, extractor_type: Optional[str] = None):
        """
        Args:
            extractor_type: 사용할 추출기 타입 (None이면 기본값 사용)
        """
        self.extractor = OptimizedPDFExtractorFactory.create_extractor(extractor_type)
        self.extractor_type = extractor_type or OptimizedPDFExtractorFactory.get_default_extractor()

    def extract_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """PDF 추출 (텍스트 + 메타데이터 + 성능 정보)"""
        start_time = time.time()

        try:
            # 텍스트 추출
            text = self.extractor.extract_text(pdf_path)

            # 메타데이터 추출
            metadata = self.extractor.get_metadata(pdf_path)

            processing_time = time.time() - start_time

            return {
                "success": True,
                "text": text,
                "metadata": metadata,
                "processing_info": {
                    "extractor_type": self.extractor_type,
                    "processing_time": processing_time,
                    "text_length": len(text),
                    "file_size_mb": metadata.get("file_size", 0) / (1024 * 1024),
                    "processing_speed_mb_per_sec": (metadata.get("file_size", 0) / (1024 * 1024)) / processing_time if processing_time > 0 else 0
                }
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PDF 추출 실패: {e}")

            return {
                "success": False,
                "error": str(e),
                "processing_info": {
                    "extractor_type": self.extractor_type,
                    "processing_time": processing_time
                }
            }

    def extract_text_only(self, pdf_path: str) -> str:
        """텍스트만 빠르게 추출"""
        return self.extractor.extract_text(pdf_path)


# 편의 함수들
def extract_pdf_optimized(pdf_path: str, extractor_type: Optional[str] = None) -> Dict[str, Any]:
    """최적화된 PDF 추출 (편의 함수)"""
    service = OptimizedPDFService(extractor_type)
    return service.extract_pdf(pdf_path)


def extract_text_fast(pdf_path: str) -> str:
    """빠른 텍스트 추출 (편의 함수) - PyMuPDF 사용"""
    service = OptimizedPDFService("pymupdf")
    return service.extract_text_only(pdf_path)


def get_extractor_recommendations() -> Dict[str, str]:
    """추출기별 추천 사용 사례"""
    return {
        "pymupdf": "🏆 기본 추천 - 빠르고 품질 좋음 (실시간 처리, 테이블 추출)",
        "pypdf2": "🔧 호환성 - 안정적이고 널리 사용됨 (레거시 시스템)",
        "langchain_pymupdf": "🔗 LangChain 연동 - 기존 LangChain 파이프라인과 호환"
    }