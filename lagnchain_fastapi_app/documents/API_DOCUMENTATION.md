# 📚 PDF 벡터 API 문서

## 🎯 API 개요

PDF 파일을 업로드하여 벡터 데이터베이스에 저장하고, 문서별 검색 및 RAG 컨텍스트 추출을 제공하는 API입니다.

**Base URL**: `http://localhost:8000`

---

## 📤 POST /pdf/upload

### Description
* PDF 파일을 업로드하여 텍스트를 추출하고 벡터 데이터베이스에 저장합니다.
* 업로드 완료 시 고유한 `document_id`를 반환하여 나중에 특정 문서 검색에 사용할 수 있습니다.

### Request
* **Content-Type**: `multipart/form-data`
* **body**
    * file: (File, required) 업로드할 PDF 파일 (.pdf 확장자만 허용)

### Response (Success)
* HTTP 200
    * JSON 타입 반환
        ```json
        {
            "message": "PDF 업로드 및 벡터 저장 성공",
            "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "filename": "algorithm_study.pdf",
            "file_size": 1048576,
            "text_length": 15420,
            "total_chunks": 18,
            "stored_chunks": 18,
            "db_type": "weaviate",
            "upload_timestamp": "2024-01-15T10:30:45.123456",
            "note": "document_id를 저장하여 나중에 RAG 퀴즈 생성 시 사용하세요"
        }
        ```

### Response (Fail)
* HTTP 400
    - PDF 파일만 지원됩니다
    - PDF에서 충분한 텍스트를 추출할 수 없습니다
* HTTP 500
    - 서버 오류: [상세 에러 메시지]

### 사용 예시
```bash
curl -X POST "http://localhost:8000/pdf/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@algorithm_study.pdf"
```

---

## 📋 GET /pdf/documents

### Description
* 업로드된 모든 PDF 문서의 목록을 조회합니다.
* 각 문서의 기본 정보와 RAG 준비 상태를 포함합니다.

### Request
* **method**: GET
* **parameters**: 없음

### Response (Success)
* HTTP 200
    * JSON 타입 반환
        ```json
        {
            "message": "문서 목록 조회 성공",
            "total_documents": 3,
            "db_type": "weaviate",
            "documents": [
                {
                    "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "source_filename": "algorithm_study.pdf",
                    "chunk_count": 18,
                    "upload_timestamp": "2024-01-15T10:30:45.123456",
                    "total_chars": 15420,
                    "available_for_rag": true,
                    "recommended_for_quiz": true
                }
            ],
            "note": "document_id를 사용하여 특정 문서로 RAG 퀴즈를 생성할 수 있습니다"
        }
        ```

### Response (Fail)
* HTTP 500
    - 문서 목록 조회 오류: [상세 에러 메시지]

### 사용 예시
```bash
curl -X GET "http://localhost:8000/pdf/documents"
```

---

## 📄 GET /pdf/documents/{document_id}

### Description
* 특정 문서의 상세 정보를 조회합니다.
* RAG 퀴즈 생성을 위한 추가 메타데이터를 포함합니다.

### Request
* **method**: GET
* **path parameters**
    * document_id: (str, required) 조회할 문서의 고유 ID

### Response (Success)
* HTTP 200
    * JSON 타입 반환
        ```json
        {
            "message": "문서 정보 조회 성공",
            "document": {
                "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "source_filename": "algorithm_study.pdf",
                "chunk_count": 18,
                "total_chars": 15420,
                "upload_timestamp": "2024-01-15T10:30:45.123456",
                "rag_ready": true,
                "chunk_size_avg": 856,
                "quiz_generation_score": 10
            },
            "db_type": "weaviate",
            "rag_info": {
                "can_generate_quiz": true,
                "recommended_questions": 9,
                "content_quality": "high"
            }
        }
        ```

### Response (Fail)
* HTTP 404
    - 문서를 찾을 수 없습니다: [document_id]
* HTTP 500
    - 문서 정보 조회 오류: [상세 에러 메시지]

### 사용 예시
```bash
curl -X GET "http://localhost:8000/pdf/documents/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

---

## 🔍 GET /pdf/search

### Description
* 업로드된 모든 문서에서 텍스트 검색을 수행합니다.
* 코사인 유사도 기반으로 결과를 랭킹하여 반환합니다.

### Request
* **method**: GET
* **query parameters**
    * query: (str, required) 검색할 텍스트 쿼리
    * top_k: (int, optional) 반환할 결과 개수 (기본값: 5, 최대: 20)

### Response (Success)
* HTTP 200
    * JSON 타입 반환
        ```json
        {
            "message": "전체 검색 완료",
            "query": "동적계획법",
            "total_results": 5,
            "db_type": "weaviate",
            "results": [
                {
                    "doc_id": "a1b2c3d4_chunk_0",
                    "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "source_filename": "algorithm_study.pdf",
                    "text_preview": "동적계획법은 복잡한 문제를 간단한 하위 문제로...",
                    "similarity": 0.8542,
                    "chunk_index": 0
                }
            ]
        }
        ```

### Response (Fail)
* HTTP 400
    - 검색 쿼리가 비어있습니다
* HTTP 500
    - 검색 오류: [상세 에러 메시지]

### 사용 예시
```bash
curl -X GET "http://localhost:8000/pdf/search?query=동적계획법&top_k=5"
```

---

## 🎯 GET /pdf/search/{document_id}

### Description
* 특정 문서 내에서만 텍스트 검색을 수행합니다.
* RAG 퀴즈 생성을 위한 컨텍스트 추출 기능을 포함합니다.

### Request
* **method**: GET
* **path parameters**
    * document_id: (str, required) 검색할 문서의 고유 ID
* **query parameters**
    * query: (str, required) 검색할 텍스트 쿼리
    * top_k: (int, optional) 반환할 결과 개수 (기본값: 5, 최대: 10)

### Response (Success)
* HTTP 200
    * JSON 타입 반환
        ```json
        {
            "message": "문서 내 검색 완료",
            "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "document_filename": "algorithm_study.pdf",
            "query": "동적계획법",
            "total_results": 5,
            "db_type": "weaviate",
            "results": [
                {
                    "doc_id": "a1b2c3d4_chunk_0",
                    "text_preview": "동적계획법은 복잡한 문제를 간단한 하위 문제로...",
                    "full_text": "동적계획법은 복잡한 문제를 간단한 하위 문제로 나누어 해결하는 알고리즘 기법입니다...",
                    "similarity": 0.8542,
                    "chunk_index": 0
                }
            ],
            "rag_context": {
                "combined_text": "동적계획법은 복잡한 문제를 간단한 하위 문제로 나누어 해결하는...",
                "context_length": 4280,
                "ready_for_rag": true
            }
        }
        ```

### Response (Fail)
* HTTP 400
    - 검색 쿼리가 비어있습니다
* HTTP 404
    - 문서를 찾을 수 없습니다: [document_id]
* HTTP 500
    - 문서 내 검색 오류: [상세 에러 메시지]

### 사용 예시
```bash
curl -X GET "http://localhost:8000/pdf/search/a1b2c3d4-e5f6-7890-abcd-ef1234567890?query=동적계획법&top_k=5"
```

---

## 🏥 GET /pdf/health

### Description
* PDF 벡터 서비스의 상태를 확인합니다.
* 현재 사용 중인 데이터베이스와 전체 통계를 포함합니다.

### Request
* **method**: GET
* **parameters**: 없음

### Response (Success)
* HTTP 200
    * JSON 타입 반환
        ```json
        {
            "status": "healthy",
            "service": "PDF Vector Service",
            "vector_db": "weaviate",
            "total_documents": 15,
            "total_uploaded_files": 15,
            "supported_dbs": ["weaviate", "chroma"],
            "endpoints": [
                "POST /pdf/upload",
                "GET /pdf/documents",
                "GET /pdf/documents/{document_id}",
                "GET /pdf/search",
                "GET /pdf/search/{document_id}",
                "GET /pdf/health",
                "POST /pdf/switch-db"
            ]
        }
        ```

### Response (Fail)
* HTTP 503
    * JSON 타입 반환
        ```json
        {
            "status": "unhealthy",
            "error": "데이터베이스 연결 오류"
        }
        ```

### 사용 예시
```bash
curl -X GET "http://localhost:8000/pdf/health"
```

---

## 🔄 POST /pdf/switch-db

### Description
* 벡터 데이터베이스를 다른 타입으로 전환합니다.
* 지원 데이터베이스: **weaviate** (권장), **chroma**

### Request
* **method**: POST
* **query parameters**
    * db_type: (str, required) 전환할 데이터베이스 타입
        - **"weaviate"**: 고성능, 확장성 우수 (권장)
        - **"chroma"**: 가벼움, 메모리 효율적

### 데이터베이스 비교표
| 데이터베이스 | 성능 | 메모리 사용량 | 확장성 | 안정성 | 권장도 |
|-------------|------|---------------|--------|--------|--------|
| **Weaviate** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🥇 |
| **ChromaDB** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🥈 |

### Response (Success)
* HTTP 200
    * JSON 타입 반환
        ```json
        {
            "message": "데이터베이스가 weaviate으로 변경되었습니다",
            "previous_db": "chroma",
            "current_db": "weaviate",
            "total_documents": 0
        }
        ```

### Response (Fail)
* HTTP 400
    - 지원하지 않는 DB 타입: [db_type]. 지원 타입: ["weaviate", "chroma"]
* HTTP 500
    - DB 변경 오류: [상세 에러 메시지]

### 사용 예시
```bash
# Weaviate로 전환 (권장)
curl -X POST "http://localhost:8000/pdf/switch-db?db_type=weaviate"

# ChromaDB로 전환
curl -X POST "http://localhost:8000/pdf/switch-db?db_type=chroma"
```

---

## 📊 GET /pdf/stats

### Description
* 벡터 데이터베이스의 상세 통계 정보를 조회합니다.

### Request
* **method**: GET
* **parameters**: 없음

### Response (Success)
* HTTP 200
    * JSON 타입 반환
        ```json
        {
            "db_type": "weaviate",
            "total_documents": 15,
            "total_uploaded_files": 15,
            "supported_dbs": ["weaviate", "chroma"],
            "performance_info": {
                "avg_search_time": 0.001,
                "avg_upload_time": 0.5,
                "total_chunks": 342
            }
        }
        ```

### Response (Fail)
* HTTP 500
    - 통계 조회 오류: [상세 에러 메시지]

### 사용 예시
```bash
curl -X GET "http://localhost:8000/pdf/stats"
```

---

## 🛠️ 사용 시나리오

### 시나리오 1: 기본 PDF 업로드 및 검색

```bash
# 1. PDF 업로드
response=$(curl -X POST "http://localhost:8000/pdf/upload" \
     -F "file=@algorithm_study.pdf")
document_id=$(echo $response | jq -r '.document_id')

# 2. 업로드된 문서 목록 확인
curl -X GET "http://localhost:8000/pdf/documents"

# 3. 특정 문서에서 검색
curl -X GET "http://localhost:8000/pdf/search/$document_id?query=동적계획법&top_k=3"
```

### 시나리오 2: RAG 퀴즈 생성용 컨텍스트 추출

```python
import requests
import json

# 1. PDF 업로드
with open('algorithm_study.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/pdf/upload',
        files={'file': f}
    )
    document_id = response.json()['document_id']

# 2. RAG 컨텍스트 추출
context_response = requests.get(
    f'http://localhost:8000/pdf/search/{document_id}',
    params={'query': '동적계획법', 'top_k': 5}
)

rag_context = context_response.json()['rag_context']['combined_text']

# 3. 이제 rag_context를 GPT/Claude에 넣어서 퀴즈 생성!
print(f"RAG 컨텍스트 길이: {len(rag_context)}자")
```

### 시나리오 3: 데이터베이스 성능 테스트

```bash
# 1. 현재 상태 확인
curl -X GET "http://localhost:8000/pdf/health"

# 2. ChromaDB로 전환하여 테스트
curl -X POST "http://localhost:8000/pdf/switch-db?db_type=chroma"

# 3. PDF 업로드 및 검색 성능 측정
time curl -X POST "http://localhost:8000/pdf/upload" -F "file=@test.pdf"
time curl -X GET "http://localhost:8000/pdf/search?query=테스트&top_k=5"

# 4. Weaviate로 다시 전환
curl -X POST "http://localhost:8000/pdf/switch-db?db_type=weaviate"

# 5. 동일한 성능 측정
time curl -X POST "http://localhost:8000/pdf/upload" -F "file=@test.pdf"
time curl -X GET "http://localhost:8000/pdf/search?query=테스트&top_k=5"
```

---

## ⚠️ 에러 코드 및 해결 방법

### 공통 에러
- **HTTP 400**: 요청 파라미터 오류
- **HTTP 404**: 리소스를 찾을 수 없음
- **HTTP 500**: 서버 내부 오류

### 주요 에러 상황 및 해결책

#### 1. PDF 업로드 실패
```json
{"detail": "PDF에서 충분한 텍스트를 추출할 수 없습니다"}
```
**해결책**: 텍스트가 포함된 PDF 파일을 사용하세요. 이미지만 있는 PDF는 OCR 처리가 필요합니다.

#### 2. 문서를 찾을 수 없음
```json
{"detail": "문서를 찾을 수 없습니다: invalid-document-id"}
```
**해결책**: 유효한 `document_id`를 사용하세요. `/pdf/documents` 엔드포인트로 문서 목록을 확인할 수 있습니다.

#### 3. 검색 쿼리 누락
```json
{"detail": "검색 쿼리가 비어있습니다"}
```
**해결책**: `query` 파라미터에 검색할 텍스트를 입력하세요.

#### 4. 지원하지 않는 데이터베이스
```json
{"detail": "지원하지 않는 DB 타입: invalid_db. 지원 타입: ['weaviate', 'chroma']"}
```
**해결책**: `weaviate` 또는 `chroma`만 사용 가능합니다.

---

## 🚀 성능 최적화 팁

### 1. 적절한 top_k 값 설정
- **검색 속도 우선**: `top_k=3~5`
- **정확도 우선**: `top_k=10~20`

### 2. 데이터베이스 선택
- **대용량 문서**: Weaviate 권장
- **메모리 제한 환경**: ChromaDB 권장

### 3. 효율적인 검색 쿼리
- **구체적인 키워드** 사용
- **동의어나 관련 용어** 포함

### 4. 배치 처리
```python
# 여러 PDF 동시 업로드 시
import asyncio
import aiohttp

async def upload_multiple_pdfs(pdf_files):
    async with aiohttp.ClientSession() as session:
        tasks = [
            upload_pdf(session, pdf_file)
            for pdf_file in pdf_files
        ]
        return await asyncio.gather(*tasks)
```

---

## 📱 클라이언트 SDK 예시

### Python 클라이언트
```python
class PDFVectorClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def upload_pdf(self, file_path):
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/pdf/upload",
                files={'file': f}
            )
        return response.json()

    def search_in_document(self, document_id, query, top_k=5):
        response = requests.get(
            f"{self.base_url}/pdf/search/{document_id}",
            params={'query': query, 'top_k': top_k}
        )
        return response.json()

    def get_documents(self):
        response = requests.get(f"{self.base_url}/pdf/documents")
        return response.json()

# 사용 예시
client = PDFVectorClient()
result = client.upload_pdf("study_material.pdf")
document_id = result["document_id"]

search_result = client.search_in_document(
    document_id,
    "동적계획법",
    top_k=3
)
```

### JavaScript/Node.js 클라이언트
```javascript
class PDFVectorClient {
    constructor(baseUrl = "http://localhost:8000") {
        this.baseUrl = baseUrl;
    }

    async uploadPDF(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseUrl}/pdf/upload`, {
            method: 'POST',
            body: formData
        });

        return await response.json();
    }

    async searchInDocument(documentId, query, topK = 5) {
        const response = await fetch(
            `${this.baseUrl}/pdf/search/${documentId}?query=${query}&top_k=${topK}`
        );
        return await response.json();
    }

    async getDocuments() {
        const response = await fetch(`${this.baseUrl}/pdf/documents`);
        return await response.json();
    }
}

// 사용 예시
const client = new PDFVectorClient();
const result = await client.uploadPDF(pdfFile);
const documentId = result.document_id;

const searchResult = await client.searchInDocument(
    documentId,
    "동적계획법",
    3
);
```

---

**📝 마지막 업데이트**: 2024년 1월 15일
**🔗 관련 문서**: [RAG 통합 가이드](./RAG_INTEGRATION_GUIDE.md), [최종 요약](./FINAL_SUMMARY.md)