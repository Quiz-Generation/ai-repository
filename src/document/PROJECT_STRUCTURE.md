# AI Repository - 프로젝트 구조

## 📁 전체 프로젝트 구조

```
ai-repository/
├── .git/                          # Git 버전 관리
├── .venv/                         # Python 가상환경
├── .vscode/                       # VS Code 설정
├── .pytest_cache/                 # pytest 캐시
├── src/                           # 소스 코드 메인 디렉토리
│   ├── app/                       # FastAPI 애플리케이션
│   ├── statics/                   # 정적 파일
│   ├── document/                  # 문서
│   └── docker-compose.yml         # Docker Compose 설정
├── setup.py                       # 프로젝트 설치 설정
├── .gitignore                     # Git 무시 파일 목록
├── README.md                      # 프로젝트 README
└── LICENSE                        # 라이선스 파일
```

## 📁 src/ 디렉토리 상세 구조

### 🏗️ app/ - FastAPI 애플리케이션
```
src/app/
├── __init__.py                    # 패키지 초기화
├── main.py                        # FastAPI 애플리케이션 진입점
├── api/                           # API 라우터
│   ├── __init__.py
│   ├── document_routes.py         # 문서 관련 API 엔드포인트
│   └── quiz_routes.py             # 퀴즈 관련 API 엔드포인트
├── core/                          # 핵심 설정 및 인프라
│   ├── __init__.py
│   ├── config.py                  # 애플리케이션 설정
│   ├── pdf_loader/                # PDF 로더 모듈
│   │   ├── __init__.py
│   │   ├── base.py                # PDF 로더 기본 클래스
│   │   ├── factory.py             # PDF 로더 팩토리
│   │   ├── pdfminer_loader.py     # PDFMiner 기반 로더
│   │   ├── pdfplumber_loader.py   # PDFPlumber 기반 로더
│   │   ├── pymupdf_loader.py      # PyMuPDF 기반 로더
│   │   └── pypdf_loader.py        # PyPDF 기반 로더
│   └── vector_db/                 # 벡터 데이터베이스
│       ├── __init__.py
│       ├── base.py                # 벡터 DB 기본 클래스
│       ├── factory.py             # 벡터 DB 팩토리
│       ├── faiss_db.py            # FAISS 벡터 DB 구현
│       └── milvus_db.py           # Milvus 벡터 DB 구현
├── service/                       # 비즈니스 로직 서비스
│   ├── __init__.py
│   ├── document_service.py        # 문서 처리 서비스
│   ├── quiz_service.py            # 퀴즈 생성 서비스
│   └── vector_db_service.py       # 벡터 DB 서비스
├── models/                        # 데이터 모델
│   ├── __init__.py
│   └── document_model.py          # 문서 모델
├── schemas/                       # Pydantic 스키마
│   └── __init__.py
├── repository/                    # 데이터 접근 계층
│   └── __init__.py
├── helper/                        # 유틸리티 헬퍼
│   ├── __init__.py
│   ├── pdf_helper.py              # PDF 관련 헬퍼
│   ├── pdf_loader_helper.py       # PDF 로더 헬퍼
│   └── text_helper.py             # 텍스트 처리 헬퍼
├── agent/                         # AI 에이전트
│   ├── __init__.py
│   ├── quiz_generator.py          # 퀴즈 생성 에이전트
│   └── prompt/                    # 프롬프트 관리
│       ├── __init__.py
│       └── quiz_prompt_manager.py # 퀴즈 프롬프트 관리자
└── docs/                          # API 문서
    └── __init__.py
```

### 📁 statics/ - 정적 파일
```
src/statics/
└── temp_pdf/                      # 임시 PDF 파일 저장소
```

### 📁 document/ - 프로젝트 문서
```
src/document/
└── PROJECT_STRUCTURE.md           # 프로젝트 구조 문서 (현재 파일)
```

## 🏗️ 아키텍처 개요

### 📋 계층 구조
1. **API Layer** (`api/`) - HTTP 요청/응답 처리
2. **Service Layer** (`service/`) - 비즈니스 로직
3. **Repository Layer** (`repository/`) - 데이터 접근
4. **Model Layer** (`models/`, `schemas/`) - 데이터 모델
5. **Core Layer** (`core/`) - 인프라 및 설정
6. **Agent Layer** (`agent/`) - AI 에이전트
7. **Helper Layer** (`helper/`) - 유틸리티

### 🔧 주요 컴포넌트

#### 📄 PDF 처리 시스템
- **다중 PDF 로더 지원**: PDFMiner, PDFPlumber, PyMuPDF, PyPDF
- **팩토리 패턴**: 로더 선택 및 초기화
- **기본 클래스**: 확장 가능한 로더 구조

#### 🗄️ 벡터 데이터베이스
- **다중 벡터 DB 지원**: FAISS, Milvus
- **팩토리 패턴**: DB 선택 및 초기화
- **기본 클래스**: 확장 가능한 벡터 DB 구조

#### 🤖 AI 에이전트
- **퀴즈 생성**: PDF 기반 문제 생성
- **프롬프트 관리**: 체계적인 프롬프트 관리
- **LangChain/LangGraph**: LLM 호출 최적화

#### 🔄 서비스 레이어
- **문서 서비스**: PDF 업로드, 처리, 저장
- **퀴즈 서비스**: 문제 생성, 검증, 관리
- **벡터 DB 서비스**: 임베딩, 검색, 저장

## 🚀 실행 환경

### 📦 의존성 관리
- **setup.py**: 프로젝트 설치 및 의존성 정의
- **.venv/**: Python 가상환경
- **docker-compose.yml**: Docker 컨테이너 설정

### 🔧 개발 도구
- **.vscode/**: VS Code 설정
- **.pytest_cache/**: 테스트 캐시
- **.gitignore**: Git 무시 파일

## 📝 주요 기능

### 📚 문서 처리
- PDF 파일 업로드 및 파싱
- 텍스트 추출 및 전처리
- 벡터 데이터베이스 저장

### 🎯 퀴즈 생성
- PDF 내용 기반 문제 생성
- 다양한 문제 유형 지원
- 중복 제거 및 품질 검증
- 병렬 처리로 성능 최적화

### 🔍 벡터 검색
- 의미론적 검색
- 유사도 기반 문서 검색
- 다중 벡터 DB 지원

## 🛠️ 개발 가이드

### 📁 파일 명명 규칙
- **Python 파일**: snake_case (예: `quiz_service.py`)
- **클래스**: PascalCase (예: `QuizGenerator`)
- **함수/변수**: snake_case (예: `generate_quiz`)
- **상수**: UPPER_SNAKE_CASE (예: `MAX_QUESTIONS`)

### 🔄 코드 구조 원칙
- **단일 책임 원칙**: 각 모듈은 하나의 책임만 가짐
- **의존성 주입**: 팩토리 패턴으로 의존성 관리
- **확장 가능성**: 기본 클래스로 확장 구조 제공
- **테스트 가능성**: 모듈화된 구조로 테스트 용이

### 📋 API 설계
- **RESTful API**: 표준 HTTP 메서드 사용
- **Pydantic 스키마**: 요청/응답 데이터 검증
- **에러 처리**: 일관된 에러 응답 형식
- **문서화**: 자동 API 문서 생성