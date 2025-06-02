#!/usr/bin/env python3
"""
동적 PDF 추출기 선택 시스템
파일 크기, 내용 유형, 성능 요구사항에 따라 최적의 추출기를 자동 선택
"""
import os
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple
from lagnchain_fastapi_app.app.services.pdf_extractor import PDFExtractorFactory


class ContentType(Enum):
    """문서 내용 유형"""
    KOREAN = "korean"           # 한글 문서
    ENGLISH = "english"         # 영문 문서
    MIXED = "mixed"            # 한영 혼합
    TECHNICAL = "technical"     # 기술 문서
    UNKNOWN = "unknown"        # 알 수 없음


class Priority(Enum):
    """우선순위 유형"""
    SPEED = "speed"           # 속도 우선
    QUALITY = "quality"       # 품질 우선
    BALANCED = "balanced"     # 균형 잡힌


class DynamicPDFExtractor:
    """동적 PDF 추출기 선택기"""

    def __init__(self):
        # 추출기별 특성 정의 (실제 테스트 결과 기반)
        self.extractor_profiles = {
            "pdfminer": {
                "speed_score": 2,        # 1-5 점수 (5가 가장 빠름)
                "quality_score": 5,      # 1-5 점수 (5가 가장 좋음)
                "korean_score": 5,       # 한글 처리 점수
                "structure_score": 4,    # 구조 보존 점수
                "memory_usage": "medium",
                "best_for": ["korean", "quality", "academic"]
            },
            "pdfplumber": {
                "speed_score": 1,        # 가장 느림
                "quality_score": 4,      # 좋은 품질
                "korean_score": 5,       # 한글 처리 좋음
                "structure_score": 5,    # 구조 보존 최고
                "memory_usage": "high",
                "best_for": ["structure", "analysis", "clean_text"]
            },
            "pymupdf": {
                "speed_score": 5,        # 가장 빠름
                "quality_score": 3,      # 보통 품질
                "korean_score": 2,       # 한글 처리 약함
                "structure_score": 2,    # 구조 보존 약함
                "memory_usage": "low",
                "best_for": ["speed", "large_files", "english"]
            }
        }

    def detect_content_type(self, pdf_path: str, sample_size: int = 2000) -> ContentType:
        """PDF 내용 유형 감지 (빠른 샘플링)"""
        try:
            # PyMuPDF로 빠른 샘플 추출 (첫 페이지만)
            import fitz
            doc = fitz.open(pdf_path)
            if len(doc) > 0:
                sample_text = doc[0].get_text()[:sample_size]
            else:
                sample_text = ""
            doc.close()

            if not sample_text.strip():
                return ContentType.UNKNOWN

            # 한글 문자 비율 계산
            korean_chars = sum(1 for char in sample_text if 0xAC00 <= ord(char) <= 0xD7A3)
            english_chars = sum(1 for char in sample_text if char.isalpha() and ord(char) < 128)
            total_chars = len(sample_text)

            korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
            english_ratio = english_chars / total_chars if total_chars > 0 else 0

            # 기술 문서 키워드 체크
            tech_keywords = ["API", "AWS", "algorithm", "function", "class", "method", "HTTP", "JSON"]
            tech_count = sum(1 for keyword in tech_keywords if keyword.lower() in sample_text.lower())

            # 내용 유형 결정
            if korean_ratio > 0.05:  # 5% 이상 한글
                if english_ratio > 0.3:  # 30% 이상 영문
                    return ContentType.MIXED
                else:
                    return ContentType.KOREAN
            elif tech_count >= 3:  # 기술 키워드 3개 이상
                return ContentType.TECHNICAL
            elif english_ratio > 0.5:  # 50% 이상 영문
                return ContentType.ENGLISH
            else:
                return ContentType.UNKNOWN

        except Exception as e:
            print(f"내용 유형 감지 실패: {e}")
            return ContentType.UNKNOWN

    def get_optimal_extractor(self, pdf_path: str, priority: Priority = Priority.BALANCED) -> str:
        """최적의 추출기 선택"""

        # 1. 파일 크기 확인
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

        # 2. 내용 유형 감지
        content_type = self.detect_content_type(pdf_path)

        # 3. 선택 로직
        return self._select_extractor(file_size_mb, content_type, priority)

    def _select_extractor(self, file_size_mb: float, content_type: ContentType, priority: Priority) -> str:
        """선택 로직 구현"""

        # 규칙 기반 선택
        if content_type == ContentType.KOREAN:
            # 한글 문서는 무조건 PDFMiner (AutoRAG 1위)
            return "pdfminer"

        if file_size_mb > 20:
            # 대용량 파일은 속도 우선
            if priority == Priority.QUALITY and content_type != ContentType.ENGLISH:
                return "pdfminer"  # 품질 우선이면서 영문이 아닌 경우
            else:
                return "pymupdf"   # 기본적으로 빠른 처리

        if priority == Priority.SPEED:
            return "pymupdf"
        elif priority == Priority.QUALITY:
            if content_type in [ContentType.MIXED, ContentType.TECHNICAL]:
                return "pdfminer"
            else:
                return "pdfplumber"
        else:  # BALANCED
            if file_size_mb < 5:
                return "pdfminer"      # 소용량은 품질 우선
            elif file_size_mb < 15:
                return "pdfplumber"    # 중간 크기는 균형
            else:
                return "pymupdf"       # 큰 크기는 속도 우선

    def extract_with_optimal_choice(self, pdf_path: str, priority: Priority = Priority.BALANCED) -> Dict:
        """최적 선택으로 추출 수행"""

        start_time = time.time()

        # 최적 추출기 선택
        extractor_name = self.get_optimal_extractor(pdf_path, priority)

        selection_time = time.time() - start_time

        # 실제 추출 수행
        extract_start = time.time()
        try:
            extractor = PDFExtractorFactory.create(extractor_name)
            text = extractor.extract_text(pdf_path)
            extract_time = time.time() - extract_start

            # 결과 정보
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            content_type = self.detect_content_type(pdf_path)

            return {
                "success": True,
                "text": text,
                "extractor_used": extractor_name,
                "file_size_mb": round(file_size_mb, 2),
                "content_type": content_type.value,
                "priority": priority.value,
                "selection_time": round(selection_time, 3),
                "extraction_time": round(extract_time, 3),
                "total_time": round(time.time() - start_time, 3),
                "text_length": len(text),
                "speed_mbps": round(file_size_mb / extract_time, 2) if extract_time > 0 else 0
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "extractor_used": extractor_name,
                "selection_time": round(selection_time, 3),
                "total_time": round(time.time() - start_time, 3)
            }

    def benchmark_all_strategies(self, pdf_paths: List[str]) -> Dict:
        """모든 전략 벤치마크"""
        results = {}

        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                continue

            file_name = os.path.basename(pdf_path)
            results[file_name] = {}

            # 각 우선순위별 테스트
            for priority in Priority:
                print(f"\n🔧 {file_name} - {priority.value} 전략 테스트...")
                result = self.extract_with_optimal_choice(pdf_path, priority)
                results[file_name][priority.value] = result

                if result["success"]:
                    print(f"  ✅ {result['extractor_used'].upper()} 선택 - {result['extraction_time']}초 ({result['speed_mbps']} MB/초)")
                else:
                    print(f"  ❌ 실패: {result['error']}")

        return results


def test_dynamic_extractor():
    """동적 추출기 테스트"""
    print("🚀 동적 PDF 추출기 테스트")
    print("=" * 80)

    # 테스트 파일들
    pdf_files = [
        "static/temp/lecture-DynamicProgramming.pdf",
        "static/temp/AWS Certified Solutions Architect Associate SAA-C03.pdf"
    ]

    dynamic_extractor = DynamicPDFExtractor()

    # 1. 개별 파일 테스트
    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            continue

        print(f"\n📄 {os.path.basename(pdf_file)} 분석")
        print("-" * 50)

        # 내용 유형 감지
        content_type = dynamic_extractor.detect_content_type(pdf_file)
        file_size_mb = round(os.path.getsize(pdf_file) / (1024 * 1024), 2)

        print(f"파일 크기: {file_size_mb}MB")
        print(f"내용 유형: {content_type.value}")

        # 각 우선순위별 추천
        for priority in Priority:
            recommended = dynamic_extractor.get_optimal_extractor(pdf_file, priority)
            print(f"{priority.value:>10} 우선: {recommended.upper()}")

    # 2. 전체 벤치마크
    print(f"\n📊 전체 벤치마크 테스트")
    print("=" * 80)

    benchmark_results = dynamic_extractor.benchmark_all_strategies(pdf_files)

    # 결과 분석
    print("\n🏆 벤치마크 결과 요약")
    print("-" * 50)

    for file_name, file_results in benchmark_results.items():
        print(f"\n📄 {file_name}:")

        for strategy, result in file_results.items():
            if result["success"]:
                print(f"  {strategy:>8}: {result['extractor_used'].upper()} | {result['extraction_time']}초 | {result['speed_mbps']} MB/초")
            else:
                print(f"  {strategy:>8}: ❌ 실패")

    return benchmark_results


if __name__ == "__main__":
    test_dynamic_extractor()