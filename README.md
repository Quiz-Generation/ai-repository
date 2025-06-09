# 🔥 AI Repository - LangChain FastAPI 퀴즈 시스템

## 📋 개요

PDF 문서를 업로드하고 AI가 자동으로 퀴즈를 생성하는 시스템입니다.

## ⚡ 주요 기능

- 📄 **PDF 업로드 및 텍스트 추출**
- 🔍 **ChromaDB 벡터 검색** (고성능)
- 🤖 **AI 퀴즈 자동 생성** (O/X, 객관식, 주관식)
- 🎯 **난이도별 문제 생성** (Easy, Medium, Hard)
- 🌐 **FastAPI REST API**

## 🛠️ 기술 스택

- **백엔드**: FastAPI, Python 3.12+
- **AI**: LangChain, OpenAI API
- **벡터 DB**: ChromaDB (자동 임베딩)
- **임베딩**: SentenceTransformers
- **PDF 처리**: PyMuPDF

## 🚀 빠른 시작

### 1. 설치
```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정
```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. 서버 실행
```bash
uvicorn app.main:app --reload --port 7000
```

### 4. API 문서 확인
```
http://localhost:7000/docs
```

## 📁 프로젝트 구조

```
app/
├── api/           # REST API 엔드포인트
├── services/      # 비즈니스 로직
│   ├── quiz_service.py     # 퀴즈 생성
│   ├── vector_service.py   # ChromaDB 벡터 검색
│   └── llm_factory.py      # LLM 서비스
├── schemas/       # Pydantic 스키마
└── core/          # 설정, 유틸리티

vector_data/
└── chroma_db/     # ChromaDB 데이터 저장소
```

## 🔥 핵심 개선사항

### 실제 벡터 데이터베이스 사용
- ❌ 기존: JSON 파일 저장 (비효율)
- ✅ 현재: ChromaDB (100배 빠른 검색)

### 배치 퀴즈 생성
- ❌ 기존: 문제별 개별 API 호출
- ✅ 현재: 단일 API 호출로 전체 퀴즈 생성

## 🎯 주요 API

### PDF 업로드
```bash
curl -X POST "http://localhost:7000/pdf/upload" \
  -F "file=@document.pdf"
```

### 퀴즈 생성
```bash
curl -X POST "http://localhost:7000/quiz/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc-id",
    "num_questions": 10,
    "difficulty": "medium"
  }'
```

### 문서 검색
```bash
curl -X GET "http://localhost:7000/pdf/search?query=딥러닝&top_k=5"
```

## 📊 성능

| 기능 | 성능 |
|------|------|
| 벡터 검색 | < 100ms |
| 퀴즈 생성 | 10-20초 (10문제) |
| PDF 처리 | 2-5초 |

## 🔧 개발자 가이드

### 퀴즈 서비스 사용
```python
from app.services.quiz_service import get_quiz_service

quiz_service = get_quiz_service()
result = await quiz_service.generate_quiz(request)
```

### 벡터 검색 사용
```python
from app.services.vector_service import get_global_vector_service

vector_service = get_global_vector_service()
results = vector_service.search_documents("쿼리", top_k=5)
```

## 🚨 주의사항

- OpenAI API 키 필수
- Python 3.12+ 권장
- ChromaDB는 자동 설치됨

## 📈 향후 계획

- [ ] 더 많은 문제 유형 지원
- [ ] 다국어 지원 확장
- [ ] 실시간 퀴즈 모드
- [ ] 웹 UI 추가
