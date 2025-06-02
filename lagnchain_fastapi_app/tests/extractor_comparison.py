#!/usr/bin/env python3
"""
3개 PDF 추출기의 데이터 추출 결과 비교
PDFMiner vs PDFPlumber vs PyMuPDF
"""
import time
import os
from lagnchain_fastapi_app.app.services.pdf_extractor import PDFExtractorFactory


def compare_extractors(pdf_file_name=None):
    """3개 PDF 추출기의 추출 결과 상세 비교"""
    print("🔍 3개 PDF 추출기 데이터 추출 비교")
    print("=" * 80)

    # 테스트할 PDF 파일들
    pdf_files = [
        "static/temp/lecture-DynamicProgramming.pdf",
        "static/temp/AWS Certified Solutions Architect Associate SAA-C03.pdf"
    ]

    # 특정 파일만 테스트하는 경우
    if pdf_file_name:
        pdf_files = [f"static/temp/{pdf_file_name}"]

    all_results = {}

    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            print(f"❌ 파일을 찾을 수 없습니다: {pdf_file}")
            continue

        file_size_mb = round(os.path.getsize(pdf_file) / (1024 * 1024), 2)
        print(f"\n📄 테스트 파일: {os.path.basename(pdf_file)} ({file_size_mb}MB)")
        print("=" * 80)

        # 3개 추출기로 텍스트 추출
        extractors = ["pdfminer", "pdfplumber", "pymupdf"]
        results = {}

        for extractor_name in extractors:
            print(f"\n🔧 {extractor_name.upper()} 추출 중...")

            try:
                start_time = time.time()

                extractor = PDFExtractorFactory.create(extractor_name)
                text = extractor.extract_text(pdf_file)

                processing_time = round(time.time() - start_time, 3)

                results[extractor_name] = {
                    "text": text,
                    "processing_time": processing_time,
                    "text_length": len(text),
                    "success": True
                }

                print(f"  ✅ 성공: {processing_time}초, {len(text):,}자")

            except Exception as e:
                print(f"  ❌ 실패: {str(e)}")
                results[extractor_name] = {"success": False, "error": str(e)}

        all_results[pdf_file] = results

        # 상세 비교 분석
        print("\n" + "=" * 80)
        print(f"📊 {os.path.basename(pdf_file)} 상세 분석")
        print("=" * 80)

        # 1. 기본 통계
        print("\n1️⃣ 기본 통계")
        print("-" * 40)
        for extractor_name, result in results.items():
            if result["success"]:
                speed = round(file_size_mb / result['processing_time'], 2) if result['processing_time'] > 0 else 0
                print(f"{extractor_name.upper():<12}: {result['processing_time']:<6}초 | {result['text_length']:>8,}자 | {speed:>5} MB/초")

        # 2. 텍스트 시작 부분 비교 (처음 300자)
        print("\n2️⃣ 텍스트 시작 부분 비교 (처음 300자)")
        print("-" * 40)
        for extractor_name, result in results.items():
            if result["success"]:
                text_preview = result["text"][:300].replace('\n', '\\n')
                print(f"\n{extractor_name.upper()}:")
                print(f"  {text_preview}...")

        # 3. 줄바꿈 처리 비교
        print("\n3️⃣ 줄바꿈 처리 비교")
        print("-" * 40)
        for extractor_name, result in results.items():
            if result["success"]:
                text = result["text"]
                newline_count = text.count('\n')
                double_newline_count = text.count('\n\n')
                print(f"{extractor_name.upper():<12}: 줄바꿈 {newline_count:,}개, 단락구분 {double_newline_count:,}개")

        # 4. 키워드별 검색 (파일에 따라 다른 키워드)
        print("\n4️⃣ 특정 키워드 검색")
        print("-" * 40)

        if "Dynamic" in pdf_file:
            keywords = ["Dynamic", "Programming", "algorithm", "recursion", "memoization"]
        else:  # AWS 파일
            keywords = ["AWS", "Cloud", "Architecture", "Solutions", "Security"]

        for keyword in keywords:
            print(f"\n'{keyword}' 검색 결과:")
            for extractor_name, result in results.items():
                if result["success"]:
                    text_lower = result["text"].lower()
                    count = text_lower.count(keyword.lower())
                    print(f"  {extractor_name.upper():<12}: {count}개")

        # 5. 텍스트 품질 분석
        print("\n5️⃣ 텍스트 품질 분석")
        print("-" * 40)
        for extractor_name, result in results.items():
            if result["success"]:
                text = result["text"]

                # 한글 문자 비율
                korean_chars = sum(1 for char in text if 0xAC00 <= ord(char) <= 0xD7A3)
                korean_ratio = round(korean_chars / len(text) * 100, 1) if len(text) > 0 else 0

                # 영문 문자 비율
                english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
                english_ratio = round(english_chars / len(text) * 100, 1) if len(text) > 0 else 0

                # 공백 문자 비율
                whitespace_chars = sum(1 for char in text if char.isspace())
                whitespace_ratio = round(whitespace_chars / len(text) * 100, 1) if len(text) > 0 else 0

                # 숫자 비율
                digit_chars = sum(1 for char in text if char.isdigit())
                digit_ratio = round(digit_chars / len(text) * 100, 1) if len(text) > 0 else 0

                print(f"{extractor_name.upper():<12}: 한글 {korean_ratio}% | 영문 {english_ratio}% | 공백 {whitespace_ratio}% | 숫자 {digit_ratio}%")

    # 전체 종합 결과
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("🏆 전체 종합 분석")
        print("=" * 80)

        # AutoRAG 실험 결과와 비교
        print("\n📊 AutoRAG 실험 결과 참고")
        print("-" * 40)
        extractor_info = PDFExtractorFactory.get_extractor_info()
        for extractor_name, info in extractor_info.items():
            print(f"{extractor_name.upper():<12}: {info}")

        # 파일별 성능 요약
        print("\n⚡ 성능 요약")
        print("-" * 40)
        for pdf_file, results in all_results.items():
            print(f"\n📄 {os.path.basename(pdf_file)}:")
            for extractor_name, result in results.items():
                if result["success"]:
                    file_size_mb = round(os.path.getsize(pdf_file) / (1024 * 1024), 2)
                    speed = round(file_size_mb / result['processing_time'], 2) if result['processing_time'] > 0 else 0
                    print(f"  {extractor_name.upper():<12}: {result['processing_time']}초 ({speed} MB/초)")

    return all_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 특정 파일만 테스트
        compare_extractors(sys.argv[1])
    else:
        # 모든 파일 테스트
        compare_extractors()