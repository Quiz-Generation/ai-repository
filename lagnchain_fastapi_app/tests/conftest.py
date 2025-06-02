#!/usr/bin/env python3
"""
pytest 설정 파일
- 테스트 환경 구성
- 공통 픽스처 정의
- 마커 설정
"""
import pytest
import os
import sys
import tempfile
from typing import Dict, Any

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """pytest 설정"""
    # 사용자 정의 마커 등록
    config.addinivalue_line(
        "markers", "performance: 성능 테스트 마커"
    )
    config.addinivalue_line(
        "markers", "integration: 통합 테스트 마커"
    )
    config.addinivalue_line(
        "markers", "slow: 느린 테스트 마커"
    )


@pytest.fixture(scope="session")
def test_config():
    """테스트 전역 설정"""
    return {
        "vector_db_types": ["chroma", "faiss", "pinecone", "weaviate", "qdrant"],
        "test_data_size": {
            "small": 10,
            "medium": 100,
            "large": 1000
        },
        "performance_thresholds": {
            "insert_time": 10.0,  # 초
            "search_time": 1.0,   # 초
            "accuracy": 0.7       # 70%
        },
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384
        }
    }


@pytest.fixture
def temp_pdf_file():
    """임시 PDF 파일 생성"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        # 간단한 PDF 내용 (실제로는 reportlab 등으로 생성)
        f.write(b'%PDF-1.4\n')
        f.write(b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n')
        f.write(b'2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n')
        f.write(b'3 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n')
        f.write(b'xref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n')
        f.write(b'0000000053 00000 n \n0000000105 00000 n \n')
        f.write(b'trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n149\n%%EOF\n')

        temp_path = f.name

    yield temp_path

    # 정리
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_korean_text():
    """한글 테스트 텍스트"""
    return """
    동적계획법(Dynamic Programming)은 복잡한 문제를 더 작은 하위 문제로 분할하여 해결하는 알고리즘 설계 기법입니다.
    이 기법의 핵심 아이디어는 중복되는 부분 문제의 해를 저장하여 재계산을 피하는 것입니다.
    동적계획법은 다음과 같은 특성을 가진 문제에 적용할 수 있습니다:

    1. 최적 부분 구조(Optimal Substructure): 문제의 최적 해가 부분 문제들의 최적 해로 구성될 수 있어야 합니다.
    2. 중복되는 부분 문제(Overlapping Subproblems): 같은 부분 문제가 여러 번 해결되어야 합니다.

    대표적인 동적계획법 문제로는 다음이 있습니다:
    - 피보나치 수열
    - 배낭 문제(Knapsack Problem)
    - 최장 공통 부분 수열(Longest Common Subsequence, LCS)
    - 최단 경로 문제

    동적계획법을 구현하는 방법에는 두 가지가 있습니다:
    1. 하향식(Top-down) 접근법: 메모이제이션(Memoization)을 사용합니다.
    2. 상향식(Bottom-up) 접근법: 표를 채워나가는 방식을 사용합니다.
    """


@pytest.fixture
def sample_english_text():
    """영문 테스트 텍스트"""
    return """
    FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+
    based on standard Python type hints. The key features of FastAPI include:

    1. Fast: Very high performance, on par with NodeJS and Go
    2. Fast to code: Increase the speed to develop features by about 200% to 300%
    3. Fewer bugs: Reduce about 40% of human (developer) induced errors
    4. Intuitive: Great editor support with completion everywhere
    5. Easy: Designed to be easy to use and learn
    6. Short: Minimize code duplication
    7. Robust: Get production-ready code with automatic interactive documentation
    8. Standards-based: Based on (and fully compatible with) the open standards for APIs

    FastAPI is built on top of Starlette for the web parts and Pydantic for the data parts.
    It provides automatic API documentation using Swagger UI and ReDoc.
    The framework supports async/await for handling concurrent requests efficiently.
    """


@pytest.fixture
def sample_mixed_text():
    """한영 혼합 테스트 텍스트"""
    return """
    Machine Learning(머신러닝)은 인공지능(AI)의 한 분야로, 컴퓨터가 명시적으로 프로그래밍되지 않고도
    데이터를 통해 학습하고 예측하는 기술입니다.

    주요 머신러닝 알고리즘:
    1. Supervised Learning(지도 학습): 라벨이 있는 데이터로 학습
       - Linear Regression(선형 회귀)
       - Decision Tree(의사결정 트리)
       - Random Forest(랜덤 포레스트)
       - Support Vector Machine(SVM)

    2. Unsupervised Learning(비지도 학습): 라벨이 없는 데이터로 학습
       - K-means Clustering(K-평균 클러스터링)
       - Hierarchical Clustering(계층적 클러스터링)
       - Principal Component Analysis(PCA)

    3. Reinforcement Learning(강화 학습): 환경과의 상호작용을 통해 학습
       - Q-Learning
       - Deep Q-Network(DQN)
       - Policy Gradient Methods

    Python에서 머신러닝을 위한 주요 라이브러리:
    - scikit-learn: 전통적인 머신러닝 알고리즘
    - TensorFlow/Keras: 딥러닝 프레임워크
    - PyTorch: 연구 친화적인 딥러닝 프레임워크
    - pandas: 데이터 처리 및 분석
    - numpy: 수치 계산
    """


@pytest.fixture
def quiz_test_data():
    """퀴즈 생성 테스트용 데이터"""
    return {
        "algorithms": [
            "동적계획법은 복잡한 문제를 작은 부분 문제로 나누어 해결하는 기법입니다.",
            "그래프 알고리즘 중 최단 경로를 찾는 대표적인 알고리즘은 다익스트라 알고리즘입니다.",
            "정렬 알고리즘 중 평균 시간복잡도가 O(n log n)인 것은 퀵 정렬, 병합 정렬, 힙 정렬 등이 있습니다.",
            "이진 탐색은 정렬된 배열에서 특정 값을 찾는 효율적인 알고리즘입니다."
        ],
        "web_development": [
            "RESTful API는 HTTP 메서드를 사용하여 자원에 대한 CRUD 연산을 수행합니다.",
            "FastAPI는 Python으로 고성능 API를 개발할 수 있는 현대적인 프레임워크입니다.",
            "데이터베이스 정규화는 데이터 중복을 줄이고 무결성을 보장하는 과정입니다.",
            "JWT(JSON Web Token)는 클라이언트와 서버 간 안전한 정보 전송을 위한 토큰 기반 인증 방식입니다."
        ],
        "machine_learning": [
            "지도 학습은 입력과 출력이 모두 주어진 데이터로 모델을 훈련시키는 방법입니다.",
            "과적합(Overfitting)은 모델이 훈련 데이터에만 특화되어 새로운 데이터에 대한 성능이 떨어지는 현상입니다.",
            "교차 검증(Cross Validation)은 모델의 성능을 객관적으로 평가하기 위한 기법입니다.",
            "특성 선택(Feature Selection)은 모델 성능에 중요한 특성만 선별하는 과정입니다."
        ]
    }


def pytest_collection_modifyitems(config, items):
    """테스트 수집 후 수정"""
    for item in items:
        # 성능 테스트 마커 자동 추가
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)

        # 통합 테스트 마커 자동 추가
        if "integration" in item.nodeid or "PDF" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # 느린 테스트 마커 자동 추가
        if "comparison" in item.nodeid or "large" in item.nodeid:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """각 테스트 실행 전 환경 설정"""
    # 임시 디렉토리 생성
    os.makedirs("temp_test_data", exist_ok=True)

    yield

    # 테스트 후 정리 (필요시)
    pass