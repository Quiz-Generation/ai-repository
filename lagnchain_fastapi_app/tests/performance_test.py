#!/usr/bin/env python3
"""
PDF 추출기 성능 비교 테스트
AutoRAG 실험 결과와 실제 성능을 비교해봅시다
"""
import time
import os
from lagnchain_fastapi_app.app.services.pdf_extractor import PDFExtractorFactory


def performance_test():
    """PDF 추출기 성능 비교"""
    print("🔍 PDF 추출기 성능 비교 테스트")
    print("=" * 60)

    # 테스트할 PDF 파일
    pdf_files = [
        "static/temp/lecture-DynamicProgramming.pdf",
        "static/temp/AWS Certified Solutions Architect Associate SAA-C03.pdf"
    ]

    # 테스트할 추출기들
    extractors = ["pdfminer", "pdfplumber"]

    results = {}

    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            print(f"❌ 파일을 찾을 수 없습니다: {pdf_file}")
            continue

        file_size_mb = round(os.path.getsize(pdf_file) / (1024 * 1024), 2)
        print(f"\n📄 테스트 파일: {os.path.basename(pdf_file)} ({file_size_mb}MB)")
        print("-" * 40)

        results[pdf_file] = {}

        for extractor_name in extractors:
            print(f"\n🔧 {extractor_name.upper()} 테스트 중...")

            try:
                # 성능 측정
                start_time = time.time()

                extractor = PDFExtractorFactory.create(extractor_name)
                text = extractor.extract_text(pdf_file)

                end_time = time.time()
                processing_time = round(end_time - start_time, 3)

                # 결과 저장
                results[pdf_file][extractor_name] = {
                    "processing_time": processing_time,
                    "text_length": len(text),
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
                    "speed_mb_per_sec": round(file_size_mb / processing_time, 2) if processing_time > 0 else 0,
                    "success": True
                }

                print(f"  ✅ 성공: {processing_time}초")
                print(f"  📏 텍스트 길이: {len(text):,}자")
                print(f"  ⚡ 처리 속도: {results[pdf_file][extractor_name]['speed_mb_per_sec']} MB/초")

            except Exception as e:
                print(f"  ❌ 실패: {str(e)}")
                results[pdf_file][extractor_name] = {
                    "success": False,
                    "error": str(e)
                }

    # 종합 결과 출력
    print("\n" + "=" * 60)
    print("📊 종합 성능 비교 결과")
    print("=" * 60)

    for pdf_file, file_results in results.items():
        print(f"\n📄 {os.path.basename(pdf_file)}")
        print("-" * 40)

        for extractor_name, result in file_results.items():
            if result["success"]:
                print(f"{extractor_name.upper():<12}: {result['processing_time']:<6}초 | "
                      f"{result['text_length']:>8,}자 | "
                      f"{result['speed_mb_per_sec']:>6} MB/초")
            else:
                print(f"{extractor_name.upper():<12}: ❌ 실패 - {result['error']}")

    # AutoRAG 실험 결과와 비교
    print("\n🏆 AutoRAG 실험 결과 순위")
    print("-" * 40)
    print("🥇 1위: PDFMiner  - 한글 처리 최고 성능, 띄어쓰기 완벽")
    print("🥈 2위: PDFPlumber - 줄바꿈과 문단 구조 완벽 보존")

    return results


if __name__ == "__main__":
    performance_test()