#!/usr/bin/env python3
"""
3개 PDF 추출기의 데이터 추출 결과 비교
PDFMiner vs PDFPlumber vs PyMuPDF
"""
import time
import os
from app.services.pdf_extractor import PDFExtractorFactory


def compare_extractors():
    """3개 PDF 추출기의 추출 결과 상세 비교"""
    print("🔍 3개 PDF 추출기 데이터 추출 비교")
    print("=" * 80)

    # 테스트할 PDF 파일 (작은 파일로 먼저 테스트)
    pdf_file = "static/temp/lecture-DynamicProgramming.pdf"

    if not os.path.exists(pdf_file):
        print(f"❌ 파일을 찾을 수 없습니다: {pdf_file}")
        return

    file_size_mb = round(os.path.getsize(pdf_file) / (1024 * 1024), 2)
    print(f"📄 테스트 파일: {os.path.basename(pdf_file)} ({file_size_mb}MB)")

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

    # 상세 비교 분석
    print("\n" + "=" * 80)
    print("📊 상세 비교 분석")
    print("=" * 80)

    # 1. 기본 통계
    print("\n1️⃣ 기본 통계")
    print("-" * 40)
    for extractor_name, result in results.items():
        if result["success"]:
            print(f"{extractor_name.upper():<12}: {result['processing_time']:<6}초 | {result['text_length']:>8,}자")

    # 2. 텍스트 시작 부분 비교 (처음 500자)
    print("\n2️⃣ 텍스트 시작 부분 비교 (처음 500자)")
    print("-" * 40)
    for extractor_name, result in results.items():
        if result["success"]:
            text_preview = result["text"][:500].replace('\n', '\\n')
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

    # 4. 특정 키워드 검색
    print("\n4️⃣ 특정 키워드 검색")
    print("-" * 40)
    keywords = ["Dynamic", "Programming", "algorithm", "recursion", "memoization"]

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

            # 공백 문자 비율
            whitespace_chars = sum(1 for char in text if char.isspace())
            whitespace_ratio = round(whitespace_chars / len(text) * 100, 1) if len(text) > 0 else 0

            # 숫자 비율
            digit_chars = sum(1 for char in text if char.isdigit())
            digit_ratio = round(digit_chars / len(text) * 100, 1) if len(text) > 0 else 0

            print(f"{extractor_name.upper():<12}: 한글 {korean_ratio}% | 공백 {whitespace_ratio}% | 숫자 {digit_ratio}%")

    # 6. AutoRAG 실험 결과와 비교
    print("\n6️⃣ AutoRAG 실험 결과 참고")
    print("-" * 40)
    extractor_info = PDFExtractorFactory.get_extractor_info()
    for extractor_name, info in extractor_info.items():
        print(f"{extractor_name.upper():<12}: {info}")

    return results


if __name__ == "__main__":
    compare_extractors()