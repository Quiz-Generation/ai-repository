#!/usr/bin/env python3
"""
🚀 M1 성능 테스트 스크립트
"""
import asyncio
import time
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_m1_performance():
    """M1 MPS 가속 성능 테스트"""

    print("🍎 M1 성능 테스트 시작")
    print("=" * 50)

    # 1. PyTorch MPS 확인
    try:
        import torch
        print(f"✅ PyTorch 버전: {torch.__version__}")
        print(f"✅ MPS 사용 가능: {torch.backends.mps.is_available()}")

        if torch.backends.mps.is_available():
            print("🎉 M1 MPS 가속 활성화됨!")
        else:
            print("⚠️ MPS 사용 불가능 - CPU 사용")

    except ImportError:
        print("❌ PyTorch가 설치되지 않음")
        return

    # 2. SentenceTransformer 테스트
    try:
        from sentence_transformers import SentenceTransformer

        # 테스트 텍스트 생성
        test_texts = [
            "이것은 첫 번째 테스트 문장입니다.",
            "두 번째 테스트 문장입니다.",
            "세 번째 테스트 문장입니다.",
            "네 번째 테스트 문장입니다.",
            "다섯 번째 테스트 문장입니다.",
        ] * 20  # 100개 문장으로 확장

        print(f"\n📝 테스트 텍스트: {len(test_texts)}개 문장")

        # CPU vs MPS 성능 비교
        for device_name in ["cpu", "mps"]:
            if device_name == "mps" and not torch.backends.mps.is_available():
                continue

            print(f"\n🔧 {device_name.upper()} 테스트 시작...")

            # 모델 로드
            start_time = time.time()
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device_name)
            load_time = time.time() - start_time
            print(f"   모델 로드 시간: {load_time:.2f}초")

            # 임베딩 생성
            start_time = time.time()
            embeddings = model.encode(test_texts, show_progress_bar=False)
            encode_time = time.time() - start_time
            print(f"   임베딩 생성 시간: {encode_time:.2f}초")
            print(f"   처리 속도: {len(test_texts)/encode_time:.1f} 문장/초")

            # 배치 처리 테스트
            print(f"   배치 처리 테스트...")
            batch_size = 32 if device_name == "mps" else 8
            start_time = time.time()

            all_embeddings = []
            for i in range(0, len(test_texts), batch_size):
                batch = test_texts[i:i + batch_size]
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)

            batch_time = time.time() - start_time
            print(f"   배치 처리 시간: {batch_time:.2f}초")
            print(f"   배치 처리 속도: {len(test_texts)/batch_time:.1f} 문장/초")

            # 메모리 정리
            del model
            if device_name == "mps":
                torch.mps.empty_cache()
            elif device_name == "cuda":
                torch.cuda.empty_cache()

    except ImportError:
        print("❌ sentence-transformers가 설치되지 않음")
        return

    print("\n" + "=" * 50)
    print("🎯 성능 테스트 완료!")
    print("\n💡 결과 해석:")
    print("- MPS가 CPU보다 빠르면 성능 개선 성공")
    print("- 배치 처리가 단일 처리보다 효율적이면 최적화 성공")

if __name__ == "__main__":
    asyncio.run(test_m1_performance())