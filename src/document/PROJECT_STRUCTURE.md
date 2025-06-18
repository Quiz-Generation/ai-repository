# 🚀 PDF Processing with Vector DB 프로젝트 구조

## 📁 전체 구조

```
ai-repository/
├── 📄 README.md                           # 프로젝트 설명서
├── 📄 requirements.txt                    # Python 의존성 패키지 (루트)
├── 📄 LICENSE                             # 라이선스 파일
├── 📄 .gitignore                          # Git 무시 파일 목록
├── 📄 .python-version                     # Python 버전 지정
├── 📁 .git/                               # Git 저장소 메타데이터
├── 📁 .vscode/                            # VS Code 설정 파일들
├── 📁 .venv/                              # Python 가상환경
├── 📁 .pytest_cache/                      # Pytest 캐시 파일들
├── 📁 data/                               # 전역 데이터 저장소
└── 📁 src/              # 🎯 메인 애플리케이션 폴더
    ├── 📄 requirements.txt                # 앱별 Python 의존성
    ├── 📄 Dockerfile                      # Docker 컨테이너 설정
    ├── 📄 docker-compose.yml             # Multi-container 설정 (Milvus)
    ├── 📁 logs/                           # 애플리케이션 로그 저장소
    │   └── 📄 app.log                     # 메인 로그 파일
    ├── 📁 data/                           # 애플리케이션 데이터
    │   └── 📁 vector_storage/             # 벡터 DB 저장소
    │       ├── 📁 milvus/                 # Milvus DB 데이터 (1순위)
    │       └── 📁 faiss/                  # FAISS DB 데이터 (2순위)
    ├── 📁 statics/                        # 정적 파일 (UI 등)
    ├── 📁 document/                       # 프로젝트 문서
    │   └── 📄 PROJECT_STRUCTURE.md       # 현재 파일
    ├── 📁 agent/                          # AI 에이전트 관련 (미래 확장)
    │   └── 📄 __init__.py
    └── 📁 app/                            # 🚀 FastAPI 핵심 애플리케이션
        ├── 📄 __init__.py                 # Python 패키지 초기화
        ├── 📄 main.py                     # 🎯 FastAPI 메인 애플리케이션 + 서버 실행
        ├── 📁 api/                        # API 라우터들
        │   ├── 📄 __init__.py
        │   └── 📄 document_routes.py      # 📄🗄️ 통합 문서 처리 API (간소화)
        ├── 📁 service/                    # 🧠 비즈니스 로직 서비스들
        │   ├── 📄 __init__.py
        │   ├── 📄 document_service.py     # PDF 문서 처리 서비스
        │   └── 📄 vector_db_service.py    # 벡터 DB 관리 서비스
        ├── 📁 repository/                 # 💾 데이터 접근 계층 (예약됨)
        │   └── 📄 __init__.py
        ├── 📁 helper/                     # 🛠️ 유틸리티 헬퍼 함수들
        │   ├── 📄 __init__.py
        │   ├── 📄 pdf_loader_helper.py    # PDF 로더 선택 헬퍼
        │   └── 📄 text_helper.py          # 텍스트 처리 헬퍼
        ├── 📁 schemas/                    # 📋 API 스키마 정의 (예약됨)
        │   └── 📄 __init__.py
        └── 📁 core/                       # 🏗️ 핵심 구현체들
            ├── 📄 __init__.py
            ├── 📁 pdf_loader/             # PDF 로더 팩토리 패턴
            │   ├── 📄 __init__.py
            │   ├── 📄 factory.py          # PDF 로더 팩토리
            │   ├── 📄 base.py             # 추상 기본 클래스
            │   ├── 📄 pymupdf_loader.py   # PyMuPDF 구현체 (성능 1순위)
            │   ├── 📄 pdfplumber_loader.py # PDFPlumber 구현체 (테이블 특화)
            │   ├── 📄 pypdf_loader.py     # PyPDF2 구현체 (경량)
            │   └── 📄 pdfminer_loader.py  # PDFMiner 구현체 (정확도)
            └── 📁 vector_db/              # 벡터 DB 팩토리 패턴
                ├── 📄 __init__.py
                ├── 📄 factory.py          # 벡터 DB 팩토리
                ├── 📄 base.py             # 추상 기본 클래스
                ├── 📄 milvus_db.py        # Milvus 구현체 (1순위 - 고성능)
                └── 📄 faiss_db.py         # FAISS 구현체 (2순위 - 로컬)
```

## 📋 레이어별 상세 설명

### 🎯 **앱 레벨 (app/)**

#### 📱 **API Layer (api/)**
- **document_routes.py**: 간소화된 통합 문서 처리 API
  - `POST /documents/upload`: PDF 업로드 + 벡터 저장 (자동화) ⭐
  - `POST /documents/vector-switch`: 벡터 DB 타입 변경 ⭐
  - `GET /documents/all-documents`: 파일별 문서 조회 (file_id 포함) ⭐
  - `GET /documents/vector-status`: 벡터 DB 상태 조회 ⭐
  - `DELETE /documents/clear-all`: 전체 데이터 삭제 (안전장치) ⭐

#### 🧠 **Service Layer (service/)**
- **document_service.py**: PDF 처리 서비스
  - 동적 로더 선택 및 폴백
  - 다국어 지원 (한국어 특화)
  - 텍스트 품질 검증
- **vector_db_service.py**: 벡터 DB 관리 서비스
  - 임베딩 생성 및 저장
  - Milvus 우선 자동 전환
  - 파일별 고유 ID 관리

#### 🛠️ **Helper Layer (helper/)**
- **pdf_loader_helper.py**: PDF 특성 분석 및 로더 선택
- **text_helper.py**: 텍스트 청킹 및 전처리

#### 🏗️ **Core Layer (core/)**
- **pdf_loader/**: 4개 PDF 로더 팩토리 패턴
- **vector_db/**: 2개 벡터 DB 팩토리 패턴

## 📊 API 엔드포인트 (간소화됨)

| 메서드 | 엔드포인트 | 설명 | 파라미터 |
|--------|------------|------|----------|
| **POST** | `/documents/upload` | PDF 업로드 + 자동 처리 ⭐ | `file` (PDF만) |
| **POST** | `/documents/vector-switch` | 벡터 DB 타입 변경 ⭐ | `db_type` |
| **GET** | `/documents/all-documents` | 문서 목록 + file_id ⭐ | 없음 (자동 100건) |
| **GET** | `/documents/vector-status` | 벡터 DB 상태 조회 ⭐ | 없음 |
| **DELETE** | `/documents/clear-all` | 전체 데이터 삭제 ⭐ | `confirm_token` |

## 🏗️ 아키텍처 패턴

### **🎯 간소화된 레이어드 아키텍처**

```
📱 API Layer (간소화)  → 5개 핵심 엔드포인트만
    ↓
🧠 Service Layer      → PDF 처리 + 벡터 DB 관리
    ↓
🛠️ Helper Layer       → 분석 및 유틸리티
    ↓
🏗️ Core Layer         → 팩토리 패턴 (PDF 4개 + Vector 2개)
    ↓
💾 Data Layer         → Milvus(우선) + FAISS(폴백)
```

### **🔄 완전 자동화 플로우**

```
PDF 업로드 → 자동 분석 → 최적 로더 → 텍스트 추출 → 자동 청킹 → 임베딩 → Milvus 저장
     ↓          ↓         ↓          ↓           ↓        ↓         ↓
   파일만    언어/복잡도   동적 선택    폴백 지원    800자    384차원   file_id 생성
```

## 🎯 핵심 기능

### **📄 PDF 처리 (완전 자동화)**
- ✅ **파일만 업로드**: 추가 파라미터 불필요
- ✅ **동적 로더 선택**: 파일 특성 기반 자동 선택
- ✅ **자동 청킹**: 한국어 최적화 (800자/100자 오버랩)
- ✅ **폴백 메커니즘**: 실패 시 자동 대체 로더

### **🗄️ 벡터 DB 통합 (Milvus 우선)**
- ✅ **2개 DB 지원**: Milvus (1순위), FAISS (2순위)
- ✅ **강제 Milvus**: 업로드 시 Milvus 우선 사용
- ✅ **파일별 ID**: `file_id` 단일 구조로 퀴즈 생성 지원
- ✅ **자동 폴백**: Milvus 실패 시 FAISS로 전환

### **🔍 지능형 분석 (자동화)**
- ✅ **언어 감지**: 파일명 + 내용 기반 (한국어 특화)
- ✅ **복잡도 분석**: 파일 크기, 테이블 존재, 폰트 복잡도
- ✅ **품질 검증**: 내용 추출 후 자동 검증

## 🚀 실행 방법

### **로컬 실행**
```bash
# 1. 가상환경 활성화
source .venv/bin/activate

# 2. 의존성 설치
cd src
pip install -r requirements.txt

# 3. 서버 실행
python app/main.py
# → http://localhost:7000
```

### **Docker 실행 (Milvus 포함)**
```bash
# 1. Docker 컨테이너 실행
cd src
docker compose up -d

# 2. API 접속
# → http://localhost:7000
# → Milvus: localhost:19530
```

## 📊 기술 스택

### **Backend**
- **FastAPI**: 고성능 웹 프레임워크
- **Python 3.9+**: 코어 언어
- **Pydantic**: 데이터 검증

### **PDF 처리**
- **PyMuPDF**: 고성능 PDF 처리 (1순위)
- **PDFPlumber**: 테이블 특화 (2순위)
- **PyPDF2**: 경량 처리 (3순위)
- **PDFMiner**: 정확도 우선 (4순위)

### **벡터 DB**
- **Milvus**: 고성능 분산 벡터 DB (1순위)
- **FAISS**: 로컬 벡터 인덱스 (2순위)

### **임베딩**
- **sentence-transformers**: all-MiniLM-L6-v2
- **384차원**: 다국어 지원

## 🔧 주요 변경사항

### **v2.0 간소화 업데이트**
- ✅ **API 간소화**: 5개 핵심 엔드포인트만 유지
- ✅ **Weaviate 제거**: Milvus + FAISS 2개만 지원
- ✅ **완전 자동화**: 파일만 업로드하면 모든 처리 자동
- ✅ **file_id 구조**: 파일별 단일 ID로 퀴즈 생성 지원
- ✅ **불필요 코드 정리**: Repository, Schema 레이어 제거

### **개발자 친화적**
- 🎯 **단순한 API**: 복잡한 파라미터 없음
- 🛡️ **안전장치**: 전체 삭제 시 확인 토큰 필요
- 📊 **상태 조회**: 실시간 DB 상태 모니터링
- 🔄 **자동 폴백**: 실패 시 자동 대체 DB 사용

---

> 📅 **최종 업데이트**: 2025-06-11 (v2.0 간소화)
> 🔧 **개발 환경**: Python 3.9+, FastAPI, Milvus, FAISS
> 🎯 **포트**: 7000
> 🏗️ **패턴**: 팩토리 + 간소화된 레이어드 아키텍처
> 🚀 **특징**: 완전 자동화, 파일만 업로드