#!/usr/bin/env python3
"""
동적 PDF 추출 서비스
파일 크기, 내용 유형, 성능 요구사항에 따라 최적의 추출기를 자동 선택하는 서비스
"""
import os
import time
import logging
from typing import Dict, Any

from ..schemas.dynamic_pdf import ExtractionResult, ContentType, Priority
from .pdf_extractor import PDFExtractorFactory

# 로거 설정
logger = logging.getLogger(__name__)

class DynamicPDFService:
    """동적 PDF 추출 서비스"""

    def __init__(self):
        self.extractor_profiles = {
            "pdfminer": {
                "speed_score": 2,        # 1-5 점수 (5가 가장 빠름)
                "quality_score": 5,      # 1-5 점수 (5가 가장 좋음)
                "korean_score": 5,       # 한글 처리 점수
                "structure_score": 5,    # 구조 보존 점수 (업데이트됨)
                "memory_usage": "medium",
                "best_for": ["korean", "quality", "academic", "structure"]
            },
            "pdfplumber": {
                "speed_score": 1,        # 가장 느림
                "quality_score": 4,      # 좋은 품질
                "korean_score": 5,       # 한글 처리 좋음
                "structure_score": 1,    # 구조 보존 최악 (문단 파괴)
                "memory_usage": "high",
                "best_for": ["clean_lines"]  # 용도 축소
            },
            "pymupdf": {
                "speed_score": 5,        # 가장 빠름
                "quality_score": 3,      # 보통 품질
                "korean_score": 1,       # 한글 처리 매우 약함 (53% 손실)
                "structure_score": 5,    # 구조 보존 좋음
                "memory_usage": "low",
                "best_for": ["speed", "large_files", "english_only"]
            }
        }
        logger.info("DynamicPDFService 초기화 완료")

    def detect_content_type(self, pdf_path: str, sample_size: int = 2000) -> ContentType:
        """PDF 내용 유형 감지 (robust한 샘플링)"""

        try:
            # PyMuPDF로 여러 페이지에서 샘플 추출 시도
            import fitz
            doc = fitz.open(pdf_path)
            sample_text = ""

            # 최대 3페이지까지 시도하여 텍스트 확보
            pages_to_try = min(3, len(doc))
            for page_num in range(pages_to_try):
                try:
                    page_text = doc[page_num].get_text()
                    if page_text.strip():  # 텍스트가 있으면 추가
                        sample_text += page_text
                        if len(sample_text) >= sample_size:
                            break
                except Exception as e:
                    logger.debug(f"페이지 {page_num} 읽기 실패: {e}")
                    continue

            doc.close()

            # 충분한 텍스트를 확보하지 못한 경우
            if len(sample_text.strip()) < 50:
                logger.warning(f"텍스트 추출 부족: {len(sample_text)}자 (파일명에서 추론 시도)")
                # 파일명에서 내용 유형 추론
                filename = os.path.basename(pdf_path).lower()
                if any(keyword in filename for keyword in ['aws', 'api', 'tech', 'architect', 'developer', 'guide']):
                    logger.info(f"파일명 기반 기술 문서 감지: {os.path.basename(pdf_path)}")
                    return ContentType.TECHNICAL
                elif any(keyword in filename for keyword in ['korean', 'hangul', '한글', '강의', 'lecture']):
                    logger.info(f"파일명 기반 한글 문서 감지: {os.path.basename(pdf_path)}")
                    return ContentType.KOREAN
                else:
                    return ContentType.UNKNOWN

            # 텍스트 분석
            sample_text = sample_text[:sample_size]  # 크기 제한

            # 한글 문자 비율 계산
            korean_chars = sum(1 for char in sample_text if 0xAC00 <= ord(char) <= 0xD7A3)
            english_chars = sum(1 for char in sample_text if char.isalpha() and ord(char) < 128)
            total_chars = len(sample_text)

            korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
            english_ratio = english_chars / total_chars if total_chars > 0 else 0

            # 기술 문서 키워드 체크 (더 많은 키워드 추가)
            tech_keywords = [
                "API", "AWS", "algorithm", "function", "class", "method", "HTTP", "JSON", "SDK", "framework",
                "service", "instance", "VPC", "EC2", "S3", "Lambda", "CloudFormation", "IAM", "RDS",
                "database", "server", "configuration", "deployment", "architecture", "security",
                "certificate", "certified", "solutions", "architect", "associate"
            ]
            tech_count = sum(1 for keyword in tech_keywords if keyword.lower() in sample_text.lower())

            # 내용 유형 결정 (더 적극적인 기술 문서 감지)
            if korean_ratio > 0.05:  # 5% 이상 한글
                if english_ratio > 0.3:  # 30% 이상 영문
                    content_type = ContentType.MIXED
                else:
                    content_type = ContentType.KOREAN
            elif tech_count >= 2:  # 기술 키워드 2개 이상 (기준 낮춤)
                content_type = ContentType.TECHNICAL
            elif english_ratio > 0.4:  # 40% 이상 영문 (기준 낮춤)
                content_type = ContentType.ENGLISH
            else:
                content_type = ContentType.UNKNOWN

            return content_type

        except Exception as e:
            logger.error(f"내용 유형 감지 실패: {e}")
            # 파일명 기반 fallback
            filename = os.path.basename(pdf_path).lower()
            if any(keyword in filename for keyword in ['aws', 'api', 'tech', 'architect', 'developer']):
                logger.info(f"오류 시 파일명 기반 기술 문서 추론: {os.path.basename(pdf_path)}")
                return ContentType.TECHNICAL
            return ContentType.UNKNOWN

    def select_optimal_extractor(self, pdf_path: str, priority: Priority = Priority.BALANCED) -> str:
        """최적의 추출기 선택 (실제 테스트 결과 기반)"""

        # 1. 파일 크기 확인
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

        # 2. 내용 유형 감지
        content_type = self.detect_content_type(pdf_path)

        # 3. 업데이트된 선택 로직 (상세 분석 결과 반영)
        selected_extractor = self._select_extractor_v2(file_size_mb, content_type, priority)

        return selected_extractor

    def _select_extractor_v2(self, file_size_mb: float, content_type: ContentType, priority: Priority) -> str:
        """개선된 선택 로직 (더 적극적인 속도 최적화)"""

        # 🚀 더 적극적인 대용량 파일 최적화: 5MB 이상이면 속도 우선
        if file_size_mb > 5:
            # 한글이 포함되어도 10MB 이상이면 PyMuPDF 고려
            if file_size_mb > 10:
                if content_type == ContentType.KOREAN:
                    return "pdfminer"
                else:
                    return "pymupdf"
            elif content_type in [ContentType.ENGLISH, ContentType.TECHNICAL, ContentType.UNKNOWN]:
                # 5MB 이상 영문/기술/불명 문서는 PyMuPDF 사용
                return "pymupdf"

        # 절대 규칙 1: 한글 문서는 무조건 PDFMiner (5MB 미만)
        if content_type in [ContentType.KOREAN, ContentType.MIXED] and file_size_mb <= 10:
            return "pdfminer"

        # 절대 규칙 2: 문단 구조가 중요한 경우도 5MB 이상이면 PyMuPDF 고려
        if priority == Priority.QUALITY:
            if file_size_mb > 8:  # 8MB 이상이면 속도 우선
                return "pymupdf"
            else:
                return "pdfminer"

        # 속도 우선 전략
        if priority == Priority.SPEED:
            if content_type == ContentType.ENGLISH or content_type == ContentType.TECHNICAL:
                return "pymupdf"
            elif file_size_mb > 3:  # 3MB 이상이면 PyMuPDF
                return "pymupdf"
            else:
                return "pdfminer"

        # 균형 전략 (파일 크기 기반) - 더 적극적으로 PyMuPDF 사용
        if file_size_mb > 15:
            # 대용량: 무조건 속도 우선
            return "pymupdf"
        elif file_size_mb > 8:
            # 중대용량: 영문/기술/불명이면 PyMuPDF
            if content_type in [ContentType.ENGLISH, ContentType.TECHNICAL, ContentType.UNKNOWN]:
                return "pymupdf"
            else:
                return "pdfminer"
        elif file_size_mb > 3:
            # 중용량: 균형
            return "pdfminer"
        else:
            # 소용량: 품질 우선
            return "pdfminer"

    def extract_text(self, pdf_path: str, priority: Priority = Priority.BALANCED) -> ExtractionResult:
        """최적 선택으로 PDF 텍스트 추출"""

        filename = os.path.basename(pdf_path)
        start_time = time.time()

        try:
            # 1. 파일 분석 및 추출기 선택
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            content_type = self.detect_content_type(pdf_path)

            # 내용 유형 로그
            if content_type in [ContentType.KOREAN, ContentType.MIXED]:
                logger.info(f"한글 문서 감지")
            elif content_type == ContentType.ENGLISH:
                logger.info(f"영문 문서 감지")
            elif content_type == ContentType.TECHNICAL:
                logger.info(f"기술 문서 감지")

            extractor_name = self._select_extractor_v2(file_size_mb, content_type, priority)
            selection_time = time.time() - start_time

            logger.info(f"STEP1: 추출기 선택: {extractor_name.upper()} ({file_size_mb:.1f}MB, {priority.value})")

            # 2. 텍스트 추출
            logger.info(f"STEP2: 텍스트 추출 시작...")
            extract_start = time.time()

            extractor = PDFExtractorFactory.create(extractor_name)
            text = extractor.extract_text(pdf_path)
            extract_time = time.time() - extract_start

            speed_mbps = round(file_size_mb / extract_time, 2) if extract_time > 0 else 0
            logger.info(f"STEP3: 추출 완료 - {len(text):,}자 ({extract_time:.2f}초, {speed_mbps} MB/초)")

            # 3. 결과 생성
            total_time = time.time() - start_time

            result = ExtractionResult(
                success=True,
                text=text,
                extractor_used=extractor_name,
                file_size_mb=round(file_size_mb, 2),
                content_type=content_type.value,
                priority=priority.value,
                selection_time=round(selection_time, 3),
                extraction_time=round(extract_time, 3),
                total_time=round(total_time, 3),
                text_length=len(text),
                speed_mbps=speed_mbps,
                metadata={
                    "extractor_profile": self.extractor_profiles.get(extractor_name, {}),
                    "auto_selected": True,
                    "selection_reason": self._get_selection_reason(file_size_mb, content_type, priority, extractor_name)
                }
            )

            return result

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"PDF 추출 실패: {str(e)} ({error_time:.2f}초)")

            return ExtractionResult(
                success=False,
                error=str(e),
                total_time=round(error_time, 3),
                metadata={"auto_selected": True, "error_stage": "extraction"}
            )

    def _get_selection_reason(self, file_size_mb: float, content_type: ContentType, priority: Priority, extractor_name: str) -> str:
        """선택 이유 설명"""
        if content_type in [ContentType.KOREAN, ContentType.MIXED]:
            return f"한글 문서 감지 → {extractor_name.upper()} (한글 처리 최적화)"
        elif priority == Priority.QUALITY:
            return f"품질 우선 → {extractor_name.upper()} (구조 보존 + 품질 최고)"
        elif priority == Priority.SPEED and file_size_mb > 5:
            return f"속도 우선 + 대용량({file_size_mb}MB) → {extractor_name.upper()} (고속 처리)"
        else:
            return f"균형 전략 ({file_size_mb}MB, {content_type.value}) → {extractor_name.upper()}"

    def extract_with_specific_extractor(self, pdf_path: str, extractor_name: str) -> ExtractionResult:
        """특정 추출기로 강제 추출 (비교/테스트 용도)"""

        start_time = time.time()

        try:
            # 파일 정보
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            content_type = self.detect_content_type(pdf_path)

            # 텍스트 추출
            extract_start = time.time()
            extractor = PDFExtractorFactory.create(extractor_name)
            text = extractor.extract_text(pdf_path)
            extract_time = time.time() - extract_start

            return ExtractionResult(
                success=True,
                text=text,
                extractor_used=extractor_name,
                file_size_mb=round(file_size_mb, 2),
                content_type=content_type.value,
                priority="manual",
                selection_time=0.0,
                extraction_time=round(extract_time, 3),
                total_time=round(time.time() - start_time, 3),
                text_length=len(text),
                speed_mbps=round(file_size_mb / extract_time, 2) if extract_time > 0 else 0,
                metadata={
                    "auto_selected": False,
                    "manual_choice": True,
                    "extractor_profile": self.extractor_profiles.get(extractor_name, {})
                }
            )

        except Exception as e:
            return ExtractionResult(
                success=False,
                error=str(e),
                extractor_used=extractor_name,
                total_time=round(time.time() - start_time, 3),
                metadata={"auto_selected": False, "manual_choice": True}
            )

    def get_extractor_recommendations(self, pdf_path: str) -> Dict[str, Any]:
        """파일에 대한 추출기 추천 정보"""

        if not os.path.exists(pdf_path):
            return {"error": "파일을 찾을 수 없습니다"}

        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        content_type = self.detect_content_type(pdf_path)

        recommendations = {}

        for priority in Priority:
            recommended = self._select_extractor_v2(file_size_mb, content_type, priority)
            recommendations[priority.value] = {
                "extractor": recommended,
                "reason": self._get_selection_reason(file_size_mb, content_type, priority, recommended)
            }

        return {
            "file_info": {
                "size_mb": round(file_size_mb, 2),
                "content_type": content_type.value,
                "filename": os.path.basename(pdf_path)
            },
            "recommendations": recommendations,
            "extractor_profiles": self.extractor_profiles
        }


# 서비스 인스턴스 (싱글톤 패턴)
dynamic_pdf_service = DynamicPDFService()