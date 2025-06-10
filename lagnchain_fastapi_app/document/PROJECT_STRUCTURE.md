# 🚀 AI Repository 프로젝트 구조

## 📁 전체 구조

```
ai-repository/
├── 📄 README.md                    # 프로젝트 설명서
├── 📄 requirements.txt             # Python 의존성 패키지
├── 📄 LICENSE                      # 라이선스 파일
├── 📄 .gitignore                   # Git 무시 파일 목록
├── 📄 .python-version              # Python 버전 지정
├── 📁 .git/                        # Git 저장소 메타데이터
├── 📁 .vscode/                     # VS Code 설정 파일들
├── 📁 .venv/                       # Python 가상환경
├── 📁 .pytest_cache/               # Pytest 캐시 파일들
└── 📁 lagnchain_fastapi_app/       # 메인 애플리케이션 폴더
    ├── 📁 app/                     # FastAPI 애플리케이션
    │   ├── 📄 __init__.py          # Python 패키지 초기화
    │   ├── 📄 main.py              # FastAPI 메인 애플리케이션
    │   ├── 📁 __pycache__/         # Python 바이트코드 캐시
    │   ├── 📁 api/                 # API 라우터들
    │   │   └── 📄 __init__.py      # API 패키지 초기화
    │   ├── 📁 service/             # 비즈니스 로직 서비스들
    │   │   └── 📄 __init__.py      # 서비스 패키지 초기화
    │   ├── 📁 repository/          # 데이터 접근 계층
    │   │   └── 📄 __init__.py      # Repository 패키지 초기화
    │   └── 📁 helper/              # 유틸리티 헬퍼 함수들
    │       └── 📄 __init__.py      # 헬퍼 패키지 초기화
    ├── 📁 agent/                   # AI 에이전트 관련
    │   └── 📄 __init__.py          # 에이전트 패키지 초기화
    ├── 📁 data/                    # 데이터 저장소 (비어있음)
    └── 📁 document/                # 문서 저장소 (비어있음)
```

## 📋 폴더별 설명

### 🔧 **루트 레벨**
- **README.md**: 프로젝트 소개 및 사용법
- **requirements.txt**: 필요한 Python 패키지 목록
- **LICENSE**: 프로젝트 라이선스 정보
- **.gitignore**: Git에서 추적하지 않을 파일/폴더 목록
- **.python-version**: 사용할 Python 버전 지정

### 🚀 **lagnchain_fastapi_app/**
메인 애플리케이션 폴더

#### 📦 **app/**
FastAPI 핵심 애플리케이션
- **main.py**: FastAPI 서버 엔트리포인트
- **api/**: REST API 엔드포인트들
- **service/**: 비즈니스 로직 처리
- **repository/**: 데이터베이스 접근 계층
- **helper/**: 공통 유틸리티 함수들

#### 🤖 **agent/**
AI 에이전트 관련 코드

#### 💾 **data/**
데이터 파일 저장소 (현재 비어있음)

#### 📄 **document/**
문서 파일 저장소 (현재 비어있음)

## 🏗️ 아키텍처 패턴

현재 프로젝트는 **계층형 아키텍처(Layered Architecture)** 패턴을 따르고 있습니다:

```
📱 API Layer      (app/api/)      - HTTP 요청/응답 처리
🧠 Service Layer  (app/service/)  - 비즈니스 로직 처리
💾 Repository     (app/repository/) - 데이터 접근 추상화
🛠️ Helper        (app/helper/)    - 공통 유틸리티
```

## 📝 현재 상태

- ✅ **기본 구조**: 설정 완료
- ✅ **FastAPI 앱**: 기본 틀 구성
- ⏳ **API 엔드포인트**: 개발 대기 중
- ⏳ **서비스 로직**: 개발 대기 중
- ⏳ **데이터 계층**: 개발 대기 중

## 🎯 다음 단계

1. **API 라우터** 구현
2. **서비스 로직** 개발
3. **데이터베이스 연동**
4. **AI 에이전트** 통합
5. **테스트 코드** 작성

---

> 📅 **최종 업데이트**: 기본 FastAPI 구조 완성
> 🔧 **개발 환경**: Python, FastAPI, 가상환경 설정 완료