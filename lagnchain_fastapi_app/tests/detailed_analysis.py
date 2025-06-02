#!/usr/bin/env python3
"""
PDF 추출기 상세 분석 스크립트
- 텍스트 품질 세부 분석
- 문단 구조 분석
- 문자 분포 분석
- 실제 텍스트 샘플 비교
"""
import time
import os
import re
from collections import Counter
from lagnchain_fastapi_app.app.services.pdf_extractor import PDFExtractorFactory


def detailed_analysis():
    """PDF 추출기 상세 분석"""
    print("🔬 PDF 추출기 상세 분석")
    print("=" * 80)

    # 테스트할 PDF 파일들
    pdf_files = [
        "static/temp/lecture-DynamicProgramming.pdf",
        "static/temp/AWS Certified Solutions Architect Associate SAA-C03.pdf"
    ]

    extractors = ["pdfminer", "pdfplumber", "pymupdf"]
    all_results = {}

    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            print(f"❌ 파일을 찾을 수 없습니다: {pdf_file}")
            continue

        file_size_mb = round(os.path.getsize(pdf_file) / (1024 * 1024), 2)
        print(f"\n📄 {os.path.basename(pdf_file)} ({file_size_mb}MB)")
        print("=" * 80)

        results = {}

        for extractor_name in extractors:
            print(f"\n🔧 {extractor_name.upper()} 분석 중...")

            try:
                start_time = time.time()
                extractor = PDFExtractorFactory.create(extractor_name)
                text = extractor.extract_text(pdf_file)
                processing_time = round(time.time() - start_time, 3)

                # 상세 분석 수행
                analysis = perform_detailed_analysis(text, extractor_name)
                analysis['processing_time'] = processing_time
                analysis['file_size_mb'] = file_size_mb
                analysis['success'] = True

                results[extractor_name] = analysis
                print(f"  ✅ 분석 완료: {processing_time}초")

            except Exception as e:
                print(f"  ❌ 실패: {str(e)}")
                results[extractor_name] = {"success": False, "error": str(e)}

        all_results[pdf_file] = results

        # 파일별 상세 비교
        print_detailed_comparison(pdf_file, results)

    # 전체 종합 분석
    print_comprehensive_analysis(all_results)

    return all_results


def perform_detailed_analysis(text: str, extractor_name: str) -> dict:
    """텍스트에 대한 상세 분석 수행"""

    # 기본 통계
    text_length = len(text)
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]

    # 문자 분포 분석
    char_analysis = analyze_characters(text)

    # 줄바꿈 및 공백 분석
    whitespace_analysis = analyze_whitespace(text)

    # 문단 구조 분석
    paragraph_analysis = analyze_paragraphs(text)

    # 언어별 분석
    language_analysis = analyze_languages(text)

    # 특수 문자 및 기호 분석
    symbol_analysis = analyze_symbols(text)

    return {
        'text': text,
        'text_length': text_length,
        'line_count': len(lines),
        'non_empty_line_count': len(non_empty_lines),
        'char_analysis': char_analysis,
        'whitespace_analysis': whitespace_analysis,
        'paragraph_analysis': paragraph_analysis,
        'language_analysis': language_analysis,
        'symbol_analysis': symbol_analysis,
        'text_preview': text[:500]
    }


def analyze_characters(text: str) -> dict:
    """문자 분포 분석"""
    korean_chars = sum(1 for char in text if 0xAC00 <= ord(char) <= 0xD7A3)
    english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
    digit_chars = sum(1 for char in text if char.isdigit())
    space_chars = sum(1 for char in text if char == ' ')
    newline_chars = text.count('\n')
    tab_chars = text.count('\t')

    total_chars = len(text)

    return {
        'korean_count': korean_chars,
        'korean_ratio': round(korean_chars / total_chars * 100, 2) if total_chars > 0 else 0,
        'english_count': english_chars,
        'english_ratio': round(english_chars / total_chars * 100, 2) if total_chars > 0 else 0,
        'digit_count': digit_chars,
        'digit_ratio': round(digit_chars / total_chars * 100, 2) if total_chars > 0 else 0,
        'space_count': space_chars,
        'space_ratio': round(space_chars / total_chars * 100, 2) if total_chars > 0 else 0,
        'newline_count': newline_chars,
        'newline_ratio': round(newline_chars / total_chars * 100, 2) if total_chars > 0 else 0,
        'tab_count': tab_chars
    }


def analyze_whitespace(text: str) -> dict:
    """공백 및 줄바꿈 분석"""
    single_newlines = text.count('\n') - text.count('\n\n') * 2
    double_newlines = text.count('\n\n')
    triple_newlines = text.count('\n\n\n')

    # 연속된 공백 분석
    multiple_spaces = len(re.findall(r'  +', text))  # 2개 이상 연속 공백

    # 줄 끝 공백
    lines_with_trailing_spaces = len([line for line in text.split('\n') if line.endswith(' ')])

    return {
        'single_newlines': single_newlines,
        'double_newlines': double_newlines,
        'triple_newlines': triple_newlines,
        'multiple_spaces': multiple_spaces,
        'lines_with_trailing_spaces': lines_with_trailing_spaces
    }


def analyze_paragraphs(text: str) -> dict:
    """문단 구조 분석"""
    # 빈 줄로 구분된 문단
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # 문단 길이 분석
    paragraph_lengths = [len(p) for p in paragraphs]
    avg_paragraph_length = sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0

    # 짧은 문단 (한 줄) vs 긴 문단
    short_paragraphs = len([p for p in paragraphs if '\n' not in p])
    long_paragraphs = len([p for p in paragraphs if '\n' in p])

    return {
        'total_paragraphs': len(paragraphs),
        'avg_paragraph_length': round(avg_paragraph_length, 1),
        'short_paragraphs': short_paragraphs,
        'long_paragraphs': long_paragraphs,
        'shortest_paragraph': min(paragraph_lengths) if paragraph_lengths else 0,
        'longest_paragraph': max(paragraph_lengths) if paragraph_lengths else 0
    }


def analyze_languages(text: str) -> dict:
    """언어별 분석"""
    # 한글 단어 수 (공백으로 구분)
    korean_words = len(re.findall(r'[가-힣]+', text))

    # 영문 단어 수
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))

    # 숫자 패턴
    numbers = len(re.findall(r'\d+', text))

    return {
        'korean_words': korean_words,
        'english_words': english_words,
        'numbers': numbers,
        'total_words': korean_words + english_words
    }


def analyze_symbols(text: str) -> dict:
    """특수 문자 및 기호 분석"""
    bullets = text.count('•') + text.count('·') + text.count('-')
    parentheses = text.count('(') + text.count(')')
    brackets = text.count('[') + text.count(']')
    quotes = text.count('"') + text.count("'")

    return {
        'bullets': bullets,
        'parentheses': parentheses,
        'brackets': brackets,
        'quotes': quotes
    }


def print_detailed_comparison(pdf_file: str, results: dict):
    """파일별 상세 비교 출력"""
    print(f"\n📊 {os.path.basename(pdf_file)} 상세 분석 결과")
    print("=" * 80)

    # 1. 기본 성능 비교
    print("\n1️⃣ 성능 비교")
    print("-" * 50)
    for extractor_name, result in results.items():
        if result["success"]:
            speed = round(result['file_size_mb'] / result['processing_time'], 2) if result['processing_time'] > 0 else 0
            print(f"{extractor_name.upper():<12}: {result['processing_time']:>6.3f}초 | {result['text_length']:>8,}자 | {speed:>6.2f} MB/초")

    # 2. 문자 분포 비교
    print("\n2️⃣ 문자 분포 비교")
    print("-" * 50)
    for extractor_name, result in results.items():
        if result["success"]:
            char = result['char_analysis']
            print(f"{extractor_name.upper():<12}: 한글 {char['korean_ratio']:>5.1f}% | 영문 {char['english_ratio']:>5.1f}% | 숫자 {char['digit_ratio']:>4.1f}% | 공백 {char['space_ratio']:>5.1f}%")

    # 3. 줄바꿈 세부 분석
    print("\n3️⃣ 줄바꿈 세부 분석")
    print("-" * 50)
    for extractor_name, result in results.items():
        if result["success"]:
            ws = result['whitespace_analysis']
            print(f"{extractor_name.upper():<12}: 단일 {ws['single_newlines']:>4}개 | 이중 {ws['double_newlines']:>3}개 | 삼중 {ws['triple_newlines']:>2}개 | 연속공백 {ws['multiple_spaces']:>3}개")

    # 4. 문단 구조 분석
    print("\n4️⃣ 문단 구조 분석")
    print("-" * 50)
    for extractor_name, result in results.items():
        if result["success"]:
            para = result['paragraph_analysis']
            print(f"{extractor_name.upper():<12}: 문단 {para['total_paragraphs']:>3}개 | 평균길이 {para['avg_paragraph_length']:>6.1f} | 짧은문단 {para['short_paragraphs']:>3}개 | 긴문단 {para['long_paragraphs']:>3}개")

    # 5. 언어별 단어 수
    print("\n5️⃣ 언어별 단어 수")
    print("-" * 50)
    for extractor_name, result in results.items():
        if result["success"]:
            lang = result['language_analysis']
            print(f"{extractor_name.upper():<12}: 한글단어 {lang['korean_words']:>4}개 | 영문단어 {lang['english_words']:>5}개 | 숫자 {lang['numbers']:>4}개")

    # 6. 텍스트 샘플 비교 (처음 200자)
    print("\n6️⃣ 텍스트 샘플 비교 (처음 200자)")
    print("-" * 50)
    for extractor_name, result in results.items():
        if result["success"]:
            preview = result['text_preview'][:200].replace('\n', '\\n')
            print(f"\n{extractor_name.upper()}:")
            print(f"  {preview}...")


def print_comprehensive_analysis(all_results: dict):
    """전체 종합 분석 출력"""
    print("\n" + "=" * 80)
    print("🏆 전체 종합 분석")
    print("=" * 80)

    # 파일별 성능 요약
    print("\n📈 파일별 성능 요약")
    print("-" * 50)
    for pdf_file, results in all_results.items():
        print(f"\n📄 {os.path.basename(pdf_file)}:")
        performance_ranking = []

        for extractor_name, result in results.items():
            if result["success"]:
                speed = round(result['file_size_mb'] / result['processing_time'], 2) if result['processing_time'] > 0 else 0
                performance_ranking.append((extractor_name, speed, result['processing_time']))

        # 속도순 정렬
        performance_ranking.sort(key=lambda x: x[1], reverse=True)

        for i, (extractor_name, speed, time) in enumerate(performance_ranking, 1):
            print(f"  {i}위. {extractor_name.upper():<12}: {time:>6.3f}초 ({speed:>6.2f} MB/초)")

    # 품질 분석 요약
    print("\n🎯 품질 분석 요약")
    print("-" * 50)

    for pdf_file, results in all_results.items():
        print(f"\n📄 {os.path.basename(pdf_file)}:")

        # 줄바꿈 품질 (적을수록 좋음)
        newline_quality = []
        for extractor_name, result in results.items():
            if result["success"]:
                ws = result['whitespace_analysis']
                total_newlines = ws['single_newlines'] + ws['double_newlines'] * 2 + ws['triple_newlines'] * 3
                newline_quality.append((extractor_name, total_newlines))

        newline_quality.sort(key=lambda x: x[1])

        print("  줄바꿈 품질 (적을수록 좋음):")
        for i, (extractor_name, count) in enumerate(newline_quality, 1):
            print(f"    {i}위. {extractor_name.upper():<12}: {count:>5}개")


if __name__ == "__main__":
    detailed_analysis()