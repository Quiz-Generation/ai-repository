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
└── 📁 lagnchain_fastapi_app/              # 🎯 메인 애플리케이션 폴더
    ├── 📄 requirements.txt                # 앱별 Python 의존성
    ├── 📁 logs/                           # 애플리케이션 로그 저장소
    │   └── 📄 app.log                     # 메인 로그 파일
    ├── 📁 data/                           # 애플리케이션 데이터
    │   └── 📁 vector_storage/             # 벡터 DB 저장소
    │       ├── 📁 milvus/                 # Milvus DB 데이터 (1순위)
    │       ├── 📁 weaviate/               # Weaviate DB 데이터 (2순위)
    │       └── 📁 faiss/                  # FAISS DB 데이터 (3순위)
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
        │   └── 📄 document_routes.py      # 📄🗄️ 통합 문서 처리 API (PDF + Vector DB)
        ├── 📁 service/                    # 🧠 비즈니스 로직 서비스들
        │   ├── 📄 __init__.py
        │   ├── 📄 document_service.py     # PDF 문서 처리 서비스
        │   └── 📄 vector_db_service.py    # 벡터 DB 관리 서비스
        ├── 📁 repository/                 # 💾 데이터 접근 계층 (미래 확장)
        │   └── 📄 __init__.py
        ├── 📁 helper/                     # 🛠️ 유틸리티 헬퍼 함수들
        │   ├── 📄 __init__.py
        │   ├── 📄 pdf_loader_helper.py    # PDF 로더 선택 헬퍼
        │   └── 📄 text_helper.py          # 텍스트 처리 헬퍼
        ├── 📁 schemas/                    # 📋 API 스키마 정의
        │   ├── 📄 __init__.py
        │   └── 📄 document_schema.py      # 문서 관련 스키마
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
                ├── 📄 weaviate_db.py      # Weaviate 구현체 (2순위 - GraphQL)
                └── 📄 faiss_db.py         # FAISS 구현체 (3순위 - 로컬)
```

## 📋 레이어별 상세 설명

### 🎯 **앱 레벨 (app/)**

#### 📱 **API Layer (api/)**
- **document_routes.py**: 통합 문서 처리 API (PDF + Vector DB)
  - `POST /documents/upload`: 기존 PDF 업로드 (벡터 저장 없음)
  - `POST /documents/upload-and-store`: PDF 업로드 + 벡터 DB 저장 ⭐
  - `POST /documents/search`: 벡터 DB 유사도 검색 ⭐
  - `GET /documents/vector-status`: 벡터 DB 상태 조회 ⭐
  - `POST /documents/vector-switch`: 벡터 DB 타입 변경 ⭐
  - `DELETE /documents/vector-documents/{filename}`: 파일별 문서 삭제 ⭐
  - `GET /documents/loaders`: PDF 로더 정보 조회
  - `GET /documents/info`: 시스템 전체 정보 조회 ⭐

#### 📋 **Schema Layer (schemas/)**
- **document_schema.py**: API 요청/응답 스키마 정의
  - 타입 안전성 보장
  - 자동 문서화 지원

## 📊 API 엔드포인트

| 분류 | 메서드 | 엔드포인트 | 설명 |
|------|--------|------------|------|
| **기존** | POST | `/documents/upload` | PDF 업로드 (벡터 저장 없음) |
| **기존** | GET | `/documents/loaders` | PDF 로더 정보 조회 |
| **신규** | POST | `/documents/upload-and-store` | PDF 업로드 + 벡터 저장 ⭐ |
| **신규** | POST | `/documents/search` | 벡터 DB 유사도 검색 ⭐ |
| **신규** | GET | `/documents/vector-status` | 벡터 DB 상태 조회 ⭐ |
| **신규** | POST | `/documents/vector-switch` | 벡터 DB 타입 변경 ⭐ |
| **신규** | DELETE | `/documents/vector-documents/{filename}` | 파일별 문서 삭제 ⭐ |
| **신규** | GET | `/documents/info` | 시스템 전체 정보 ⭐ |
| **기본** | GET | `/` | API 정보 및 기능 목록 |
| **기본** | GET | `/health` | 헬스체크 |

## 🏗️ 아키텍처 패턴

### **🎯 레이어드 아키텍처 + 팩토리 패턴**

```
📱 API Layer      → HTTP 요청/응답 처리
    ↓
🧠 Service Layer  → 비즈니스 로직 + 팩토리 사용
    ↓
🛠️ Helper Layer   → 분석 및 유틸리티 함수
    ↓
🏗️ Core Layer     → 팩토리 패턴 구현체들
    ↓
💾 Data Layer     → 벡터 DB 영구 저장소
```

### **🔄 동적 선택 플로우**

```
PDF 업로드 → 특성 분석 → 로더 선택 → 텍스트 추출 → 청킹 → 임베딩 → 벡터 DB 저장
     ↓           ↓          ↓           ↓         ↓       ↓         ↓
   파일명     언어/복잡도   최적 로더    폴백 시도   1000자   384차원   우선순위 DB
```

## 🎯 핵심 기능

### **📄 PDF 처리**
- ✅ **동적 로더 선택**: 파일 특성 기반 자동 선택
- ✅ **다국어 지원**: 한국어 특화 처리
- ✅ **폴백 메커니즘**: 실패 시 자동 대체 로더
- ✅ **4개 로더**: PyMuPDF, PDFPlumber, PyPDF2, PDFMiner

### **🗄️ 벡터 DB 통합**
- ✅ **3개 DB 지원**: Milvus (1순위), Weaviate (2순위), FAISS (3순위)
- ✅ **자동 폴백**: DB 실패 시 차순위 DB로 자동 전환
- ✅ **임베딩 생성**: sentence-transformers (all-MiniLM-L6-v2)
- ✅ **유사도 검색**: 코사인 유사도 기반

### **🔍 지능형 분석**
- ✅ **언어 감지**: 파일명 + 내용 기반 (한국어 특화)
- ✅ **복잡도 분석**: 파일 크기, 테이블 존재, 폰트 복잡도
- ✅ **청킹 전략**: 텍스트 크기별 적응형 분할

## 🚀 실행 방법

```bash
# 1. 가상환경 활성화
source .venv/bin/activate

# 2. 의존성 설치
cd lagnchain_fastapi_app
pip install -r requirements.txt

# 3. 서버 실행
python app/main.py
# → http://localhost:7000
```

## 🔧 설정 및 확장

### **벡터 DB 설정**
- **Milvus**: `localhost:19530` (Docker 필요)
- **Weaviate**: `localhost:8080` (Docker 필요)
- **FAISS**: 로컬 파일 저장 (설치만 필요)

### **확장 포인트**
- 새로운 PDF 로더 추가 → `core/pdf_loader/`
- 새로운 벡터 DB 추가 → `core/vector_db/`
- 새로운 API 엔드포인트 → `api/routes/`
- 새로운 비즈니스 로직 → `service/`

---

> 📅 **최종 업데이트**: 벡터 DB 통합 완료
> 🔧 **개발 환경**: Python 3.9+, FastAPI, Vector DBs
> 🎯 **포트**: 7000
> 🏗️ **패턴**: 팩토리 + 레이어드 아키텍처