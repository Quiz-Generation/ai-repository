"""
📄 PDF Loader Abstract Interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from fastapi import UploadFile


@dataclass
class PDFContent:
    """PDF 내용 모델"""
    text: str
    metadata: Dict[str, Any]
    page_count: int
    file_size: int


@dataclass
class PDFLoaderInfo:
    """PDF 로더 정보"""
    name: str
    description: str
    priority: int
    pros: list[str]
    cons: list[str]
    best_for: str
    supported_features: list[str]


class PDFLoader(ABC):
    """PDF 로더 추상 인터페이스"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def extract_text_from_file(self, file: UploadFile) -> PDFContent:
        """업로드된 파일에서 텍스트 추출"""
        pass

    @abstractmethod
    async def extract_text_from_path(self, file_path: str) -> PDFContent:
        """파일 경로에서 텍스트 추출"""
        pass

    @abstractmethod
    def validate_file(self, file: UploadFile) -> bool:
        """파일 유효성 검증"""
        pass

    @abstractmethod
    def get_supported_features(self) -> list[str]:
        """지원하는 기능 목록"""
        pass

    @abstractmethod
    def get_loader_info(self) -> PDFLoaderInfo:
        """로더 정보 반환"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """헬스체크"""
        pass