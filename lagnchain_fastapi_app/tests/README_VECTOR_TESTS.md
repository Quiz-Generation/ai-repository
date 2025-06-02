# 벡터 데이터베이스 TDD 테스트 가이드

> **PDF 기반 퀴즈 생성 애플리케이션을 위한 벡터 데이터베이스 테스트 스위트**

## 📚 개요

이 테스트 스위트는 PDF 파일에서 추출한 텍스트를 벡터 데이터베이스에 저장하고 검색하는 기능을 TDD(Test-Driven Development) 방식으로 검증합니다. 특히 퀴즈 생성 애플리케이션의 요구사항에 맞춰 다양한 벡터 데이터베이스를 비교 테스트합니다.

## 🎯 지원하는 벡터 데이터베이스

| 벡터 DB | 속도 | 정확도 | 최적 용도 | 비고 |
|---------|------|--------|-----------|------|
| **Chroma** | 4/5 | 4/5 | 개발, 프로토타입, 로컬 | 설정 간단 |
| **FAISS** | 5/5 | 5/5 | 대규모, 성능최적화, 연구 | Meta 개발 |
| **Pinecone** | 5/5 | 5/5 | 프로덕션, 클라우드, 확장성 | 관리형 서비스 |
| **Weaviate** | 4/5 | 4/5 | 그래프, 멀티모달, 스키마 | 오픈소스 |
| **Qdrant** | 4/5 | 4/5 | 실시간, 필터링 | Rust 기반 |

## 🧪 테스트 구조

### 1. 기본 기능 테스트 (`TestVectorDBBasics`)
- **연결 테스트**: 각 벡터 DB 연결 확인
- **삽입 테스트**: 벡터 데이터 저장 기능
- **검색 테스트**: 유사도 기반 검색 기능

### 2. 성능 테스트 (`TestVectorDBPerformance`)
- **삽입 성능**: 100개 벡터 삽입 속도 측정
- **검색 성능**: 평균 검색 시간 측정 (10회 반복)
- **메모리 효율성**: 벡터 개수별 메모리 사용량 분석

### 3. PDF-벡터 통합 테스트 (`TestPDFVectorIntegration`)
- **파이프라인 테스트**: PDF → 텍스트 → 벡터 → 저장 → 검색
- **퀴즈 생성 워크플로우**: 퀴즈 생성에 특화된 검색 테스트
- **청킹 최적화**: 다양한 텍스트 분할 방식 비교

### 4. 벡터 DB 비교 테스트 (`TestVectorDBComparison`)
- **성능 비교**: DB별 삽입/검색 속도 비교
- **정확도 비교**: 동일 데이터에 대한 검색 정확도 측정
- **확장성 분석**: 최대 처리 용량 및 메모리 효율성

## 🚀 빠른 시작

### 1. 데모 실행
```bash
# 테스트 디렉토리로 이동
cd lagnchain_fastapi_app/tests

# 간단한 데모 실행
python demo_vector_test.py
```

### 2. 개별 테스트 실행
```bash
# 기본 기능 테스트만 실행
python -m pytest test_vector_databases.py::TestVectorDBBasics -v

# 성능 테스트 실행
python -m pytest test_vector_databases.py::TestVectorDBPerformance -v

# 특정 벡터 DB만 테스트
python -m pytest test_vector_databases.py -k "chroma" -v
```

### 3. 전체 테스트 스위트 실행
```bash
# 모든 테스트 실행
python run_vector_tests.py --test-type all

# 결과를 JSON으로 저장
python run_vector_tests.py --test-type all --save-results

# 특정 테스트만 실행
python run_vector_tests.py --test-type performance
```

## 📊 테스트 시나리오

### 시나리오 1: 개발 환경 설정
```python
# Chroma DB로 로컬 개발 환경 테스트
python -m pytest test_vector_databases.py -k "chroma" -v
```

### 시나리오 2: 성능 최적화
```python
# FAISS와 Pinecone 성능 비교
python run_vector_tests.py --test-type comparison
```

### 시나리오 3: 퀴즈 생성 검증
```python
# 퀴즈 생성 관련 테스트만 실행
python -m pytest test_vector_service.py -k "quiz" -v
```

### 시나리오 4: 프로덕션 배포 준비
```python
# 전체 테스트 + 상세 리포트 생성
python run_vector_tests.py --test-type all --save-results --output-file prod_test_results.json
```

## 🔧 고급 사용법

### 1. 커스텀 테스트 데이터 추가
```python
# conftest.py에서 테스트 데이터 커스터마이징
@pytest.fixture
def custom_pdf_texts():
    return [
        "사용자 정의 테스트 텍스트 1",
        "사용자 정의 테스트 텍스트 2"
    ]
```

### 2. 특정 마커로 테스트 필터링
```bash
# 성능 테스트만 실행
python -m pytest -m performance

# 통합 테스트만 실행
python -m pytest -m integration

# 느린 테스트 제외
python -m pytest -m "not slow"
```

### 3. 병렬 테스트 실행
```bash
# pytest-xdist 사용 (설치 필요)
pip install pytest-xdist
python -m pytest -n auto test_vector_databases.py
```

## 📈 테스트 결과 해석

### 성공 기준
- **삽입 시간**: 10초 이내 (100개 벡터)
- **검색 시간**: 1초 이내 (평균)
- **정확도**: 70% 이상
- **메모리**: 1GB 이내 (10,000개 벡터)

### 일반적인 결과 패턴
```
CHROMA 삽입 성능:
  - 벡터 수: 100
  - 삽입 시간: 0.045초
  - 초당 벡터: 2222.2개/초

FAISS 검색 성능:
  - 평균 검색 시간: 0.003초
  - 검색된 결과: 5개
```

## 🐛 문제 해결

### 일반적인 오류들

#### 1. Import 오류
```bash
ModuleNotFoundError: No module named 'lagnchain_fastapi_app'
```
**해결방법**:
- `conftest.py`가 제대로 설정되어 있는지 확인
- 프로젝트 루트에서 실행하고 있는지 확인

#### 2. 의존성 누락
```bash
ImportError: No module named 'numpy'
```
**해결방법**:
```bash
pip install numpy pytest
```

#### 3. 성능 테스트 실패
```bash
AssertionError: assert 15.2 < 10.0
```
**해결방법**:
- 시스템 리소스 확인
- 테스트 데이터 크기 조정
- 성능 임계값 조정

### 로그 활성화
```bash
# 상세 로그와 함께 실행
python -m pytest test_vector_databases.py -v -s --tb=long
```

## 📋 체크리스트

### 개발 환경 설정 완료
- [ ] 기본 연결 테스트 통과
- [ ] 벡터 삽입/검색 정상 동작
- [ ] PDF 텍스트 처리 파이프라인 동작

### 성능 최적화 완료
- [ ] 삽입 성능 기준 만족 (10초 이내)
- [ ] 검색 성능 기준 만족 (1초 이내)
- [ ] 메모리 사용량 기준 만족

### 프로덕션 배포 준비
- [ ] 모든 테스트 통과
- [ ] 선택한 벡터 DB 설정 완료
- [ ] 퀴즈 생성 워크플로우 검증

## 🤝 기여 가이드

### 새로운 벡터 DB 추가
1. `VectorDBFactory.SUPPORTED_DBS`에 DB 이름 추가
2. `get_db_profiles()`에 특성 정보 추가
3. `MockVectorDB` 클래스에 해당 DB 로직 구현
4. 테스트 케이스 추가

### 새로운 테스트 시나리오 추가
1. 적절한 테스트 클래스에 메서드 추가
2. 필요시 `conftest.py`에 새로운 fixture 추가
3. 마커가 필요한 경우 `pytest_configure`에 등록

## 📚 참고 자료

- [Chroma 공식 문서](https://docs.trychroma.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Pinecone 문서](https://docs.pinecone.io/)
- [Weaviate 문서](https://weaviate.io/developers/weaviate)
- [Qdrant 문서](https://qdrant.tech/documentation/)

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 먼저 위의 문제 해결 섹션을 확인하세요
2. 테스트 로그를 확인하여 구체적인 오류를 파악하세요
3. 관련 벡터 DB 공식 문서를 참조하세요

---

**📝 마지막 업데이트**: 2024년 12월 벡터 데이터베이스 테스트 스위트 v1.0