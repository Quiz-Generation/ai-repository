"""
📄 PDF Loader Selection Helper
"""
import re
import logging
from typing import Dict, Any, Optional
from fastapi import UploadFile
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PDFAnalysisResult:
    """PDF 분석 결과"""
    language: str  # 'korean', 'english', 'mixed', 'unknown'
    has_tables: bool
    has_images: bool
    complexity: str  # 'simple', 'medium', 'complex'
    file_size: int
    estimated_pages: int
    text_density: str  # 'low', 'medium', 'high'
    font_complexity: str  # 'simple', 'complex'
    recommended_loader: str


class PDFLoaderHelper:
    """PDF 로더 선택을 위한 Helper 클래스"""

    @staticmethod
    async def analyze_pdf_characteristics(file: UploadFile) -> PDFAnalysisResult:
        """PDF 파일 특성 분석"""
        try:
            # 파일 크기 기반 분석
            file_size = file.size or 0
            estimated_pages = max(1, file_size // (50 * 1024))  # 대략적인 페이지 수 추정

            # 파일명 기반 언어 추정
            filename = file.filename or ""
            language = PDFLoaderHelper._detect_language_from_filename(filename)

            # 파일 크기 기반 복잡도 추정
            complexity = PDFLoaderHelper._estimate_complexity_from_size(file_size)

            # 테이블/이미지 존재 추정 (파일명/크기 기반)
            has_tables = PDFLoaderHelper._estimate_tables_from_filename(filename)
            has_images = PDFLoaderHelper._estimate_images_from_size(file_size)

            # 텍스트 밀도 추정
            text_density = PDFLoaderHelper._estimate_text_density(file_size, estimated_pages)

            # 폰트 복잡도 추정
            font_complexity = PDFLoaderHelper._estimate_font_complexity(language, complexity)

            analysis_result = PDFAnalysisResult(
                language=language,
                has_tables=has_tables,
                has_images=has_images,
                complexity=complexity,
                file_size=file_size,
                estimated_pages=estimated_pages,
                text_density=text_density,
                font_complexity=font_complexity,
                recommended_loader=""  # 나중에 설정
            )

            # 최적 로더 추천
            recommended_loader = PDFLoaderHelper._recommend_loader(analysis_result)
            analysis_result.recommended_loader = recommended_loader

            logger.info(f"✅ PDF 분석 완료: {filename} -> {recommended_loader}")
            return analysis_result

        except Exception as e:
            logger.error(f"❌ PDF 분석 실패: {e}")
            # 기본값 반환
            return PDFAnalysisResult(
                language="unknown",
                has_tables=False,
                has_images=False,
                complexity="simple",
                file_size=file_size,
                estimated_pages=1,
                text_density="medium",
                font_complexity="simple",
                recommended_loader="pymupdf"
            )

    @staticmethod
    def _detect_language_from_filename(filename: str) -> str:
        """파일명에서 언어 감지"""
        filename_lower = filename.lower()

        # 한글 관련 키워드
        korean_keywords = ['한글', '한국', 'korean', 'kr', '보고서', '문서', '계약서', '제안서']

        # 영어 관련 키워드
        english_keywords = ['english', 'en', 'report', 'document', 'contract', 'proposal']

        # 테이블 관련 키워드
        table_keywords = ['table', '표', 'chart', '차트', 'data', '데이터']

        korean_score = sum(1 for keyword in korean_keywords if keyword in filename_lower)
        english_score = sum(1 for keyword in english_keywords if keyword in filename_lower)

        if korean_score > english_score:
            return "korean"
        elif english_score > korean_score:
            return "english"
        elif korean_score > 0 and english_score > 0:
            return "mixed"
        else:
            return "unknown"

    @staticmethod
    def _estimate_complexity_from_size(file_size: int) -> str:
        """파일 크기로 복잡도 추정"""
        if file_size < 1024 * 1024:  # 1MB 미만
            return "simple"
        elif file_size < 10 * 1024 * 1024:  # 10MB 미만
            return "medium"
        else:
            return "complex"

    @staticmethod
    def _estimate_tables_from_filename(filename: str) -> bool:
        """파일명에서 테이블 존재 추정"""
        table_keywords = ['table', '표', 'chart', '차트', 'data', '데이터', 'excel', 'sheet']
        filename_lower = filename.lower()
        return any(keyword in filename_lower for keyword in table_keywords)

    @staticmethod
    def _estimate_images_from_size(file_size: int) -> bool:
        """파일 크기로 이미지 존재 추정"""
        # 5MB 이상이면 이미지가 있을 가능성 높음
        return file_size > 5 * 1024 * 1024

    @staticmethod
    def _estimate_text_density(file_size: int, pages: int) -> str:
        """텍스트 밀도 추정"""
        if pages == 0:
            return "medium"

        size_per_page = file_size / pages

        if size_per_page < 50 * 1024:  # 50KB per page
            return "low"
        elif size_per_page < 200 * 1024:  # 200KB per page
            return "medium"
        else:
            return "high"

    @staticmethod
    def _estimate_font_complexity(language: str, complexity: str) -> str:
        """폰트 복잡도 추정"""
        if language == "korean" or language == "mixed":
            return "complex"
        elif complexity == "complex":
            return "complex"
        else:
            return "simple"

    @staticmethod
    def _recommend_loader(analysis: PDFAnalysisResult) -> str:
        """분석 결과를 바탕으로 최적 로더 추천"""

        # 1. 한글 문서의 경우 PDFMiner 우선 (정확도)
        if analysis.language == "korean":
            if analysis.has_tables:
                return "pdfplumber"  # 한글 + 테이블
            else:
                return "pdfminer"    # 한글 텍스트

        # 2. 테이블이 많은 경우 PDFPlumber
        if analysis.has_tables:
            return "pdfplumber"

        # 3. 복잡한 레이아웃인 경우 PDFMiner
        if analysis.complexity == "complex" or analysis.font_complexity == "complex":
            return "pdfminer"

        # 4. 큰 파일인 경우 PyMuPDF (성능)
        if analysis.file_size > 20 * 1024 * 1024:  # 20MB 이상
            return "pymupdf"

        # 5. 작은 파일인 경우 PyPDF (가벼움)
        if analysis.file_size < 1024 * 1024:  # 1MB 이하
            return "pypdf"

        # 6. 기본값: PyMuPDF (최고 성능)
        return "pymupdf"

    @staticmethod
    def get_loader_selection_rules() -> Dict[str, Any]:
        """로더 선택 규칙 반환"""
        return {
            "rules": [
                {
                    "condition": "한글 문서",
                    "action": "PDFMiner 사용 (정확도 우선)",
                    "reason": "한글 폰트 처리에 특화"
                },
                {
                    "condition": "한글 + 테이블",
                    "action": "PDFPlumber 사용",
                    "reason": "테이블 추출 + 한글 지원"
                },
                {
                    "condition": "테이블 포함",
                    "action": "PDFPlumber 사용",
                    "reason": "테이블 추출에 특화"
                },
                {
                    "condition": "복잡한 레이아웃",
                    "action": "PDFMiner 사용",
                    "reason": "정확한 텍스트 추출"
                },
                {
                    "condition": "대용량 파일 (20MB+)",
                    "action": "PyMuPDF 사용",
                    "reason": "최고 성능"
                },
                {
                    "condition": "소용량 파일 (1MB-)",
                    "action": "PyPDF 사용",
                    "reason": "가벼운 처리"
                },
                {
                    "condition": "기본값",
                    "action": "PyMuPDF 사용",
                    "reason": "전반적으로 최고 성능"
                }
            ],
            "priority_order": ["한글", "테이블", "복잡도", "파일크기", "기본값"]
        }