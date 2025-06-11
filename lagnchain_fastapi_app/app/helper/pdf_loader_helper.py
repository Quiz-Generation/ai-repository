"""
🔍 PDF Loader Selection Helper
"""
import re
import logging
from typing import Dict, Any, Optional
from fastapi import UploadFile
from dataclasses import dataclass
from ..core.pdf_loader.factory import PDFLoaderFactory

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
            file_size = file.size or 0
            estimated_pages = max(1, file_size // (50 * 1024))  # 대략적인 페이지 수 추정

            # 파일명 기반 1차 언어 추정
            filename = file.filename or ""
            filename_language = PDFLoaderHelper._detect_language_from_filename(filename)

            # 실제 텍스트 기반 언어 감지
            text_language = await PDFLoaderHelper._detect_language_from_content(file)

            # 파일명과 텍스트 분석 결과 종합
            language = PDFLoaderHelper._combine_language_results(filename_language, text_language)
            logger.info(f"STEP3-1 언어 감지 완료: 파일명={filename_language}, 텍스트={text_language}, 최종={language}")

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

            logger.info(f"STEP3-2 PDF 분석 완료: {filename} -> {recommended_loader}")
            return analysis_result

        except Exception as e:
            logger.error(f"ERROR PDF 분석 실패: {e}")
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
    async def _detect_language_from_content(file: UploadFile) -> str:
        """PDF 텍스트 내용에서 언어 감지 - 단순화된 버전"""
        try:
            # 파일 내용 읽기
            file_content = await file.read()

            # 파일 포인터 원위치
            try:
                await file.seek(0)
            except:
                pass

            if not file_content:
                logger.warning("WARNING 파일 내용이 비어있습니다")
                return "unknown"

            # PyMuPDF로 빠른 텍스트 추출
            try:
                import fitz
                doc = fitz.open(stream=file_content, filetype="pdf")

                if len(doc) == 0:
                    logger.warning("WARNING PDF 페이지가 없습니다")
                    return "unknown"

                # 첫 페이지 텍스트 추출
                page = doc.load_page(0)
                sample_text = page.get_text()[:1000]  # 1000자만
                doc.close()

                if not sample_text.strip():
                    logger.warning("WARNING 추출된 텍스트가 비어있습니다")
                    return "unknown"

                # 한글/영어 문자 카운트
                korean_chars = len(re.findall(r'[가-힣]', sample_text))
                english_chars = len(re.findall(r'[a-zA-Z]', sample_text))

                logger.info(f"STEP3-1d 텍스트 분석: 한글={korean_chars}자, 영어={english_chars}자")

                # 간단한 규칙 기반 판단
                if korean_chars > 20:  # 한글이 20자 이상이면
                    return "korean"
                elif korean_chars > 5 and english_chars < korean_chars * 3:  # 한글이 조금이라도 있고 영어가 많지 않으면
                    return "korean"
                elif english_chars > 50:  # 영어가 50자 이상이면
                    return "english"
                else:
                    return "unknown"

            except Exception as e:
                logger.warning(f"WARNING 텍스트 추출 실패: {e}")
                return "unknown"

        except Exception as e:
            logger.error(f"ERROR 언어 감지 실패: {e}")
            return "unknown"

    @staticmethod
    def _detect_language_with_langdetect(text: str) -> str:
        """langdetect를 사용한 언어 감지"""
        try:
            from langdetect import detect, LangDetectException

            # 텍스트 정리
            clean_text = re.sub(r'[^\w\s가-힣]', ' ', text)
            clean_text = ' '.join(clean_text.split())

            if len(clean_text) < 20:  # 너무 짧으면 감지 어려움
                return "unknown"

            detected = detect(clean_text)

            # 언어 코드를 일반적인 이름으로 변환
            language_map = {
                'ko': 'korean',
                'en': 'english',
                'ja': 'japanese',
                'zh-cn': 'chinese',
                'zh-tw': 'chinese'
            }

            return language_map.get(detected, detected)

        except (ImportError, LangDetectException) as e:
            logger.warning(f"WARNING langdetect 실패: {e}")
            return "unknown"
        except Exception as e:
            logger.warning(f"WARNING 언어 감지 오류: {e}")
            return "unknown"

    @staticmethod
    def _combine_language_results(filename_lang: str, text_lang: str) -> str:
        """파일명과 텍스트 분석 결과 종합"""
        logger.info(f"STEP3-1c 언어 결합: filename={filename_lang}, text={text_lang}")

        # 파일명에서 korean이 감지되면 우선시 (한글 파일명은 확실함)
        if filename_lang == "korean":
            return "korean"
        # 텍스트 분석 결과를 우선시
        elif text_lang in ["korean", "english", "mixed"]:
            return text_lang
        elif filename_lang in ["english", "mixed"]:
            return filename_lang
        else:
            # 둘 다 unknown이면 기본값으로 english 설정
            return "english"

    @staticmethod
    def _detect_language_from_filename(filename: str) -> str:
        """파일명에서 언어 감지"""
        if not filename:
            return "unknown"

        logger.info(f"STEP3-1b 파일명 분석 시작: '{filename}'")

        # 1. 파일명에 한글 문자가 있는지 직접 체크
        korean_chars = len(re.findall(r'[가-힣]', filename))
        logger.info(f"STEP3-1b 파일명에서 한글 문자 {korean_chars}개 발견")

        if korean_chars > 0:
            logger.info(f"STEP3-1b 한글 문자 발견으로 korean 반환")
            return "korean"

        # 2. 한글 관련 키워드 체크
        filename_lower = filename.lower()
        korean_keywords = ['한글', '한국', 'korean', 'kr', '보고서', '문서', '계약서', '제안서', '강의', '자료']
        english_keywords = ['english', 'en', 'report', 'document', 'contract', 'proposal', 'lecture', 'material']

        korean_score = sum(1 for keyword in korean_keywords if keyword in filename_lower)
        english_score = sum(1 for keyword in english_keywords if keyword in filename_lower)

        logger.info(f"STEP3-1b 키워드 점수: korean={korean_score}, english={english_score}")

        if korean_score > 0:
            logger.info(f"STEP3-1b 한글 키워드 발견으로 korean 반환")
            return "korean"
        elif english_score > 0:
            logger.info(f"STEP3-1b 영어 키워드 발견으로 english 반환")
            return "english"
        else:
            logger.info(f"STEP3-1b 키워드 없음으로 unknown 반환")
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