"""
PDF 벡터 검색 API Swagger 문서 설명
각 API 엔드포인트별 상세 설명 및 파라미터 가이드
"""

desc_upload_pdf = """
PDF 파일을 업로드하여 벡터 데이터베이스에 저장하는 API입니다.

### 기능 설명
- PDF 파일을 업로드하면 자동으로 텍스트를 추출합니다
- 추출된 텍스트를 청크 단위로 분할하여 벡터화합니다
- 벡터 데이터베이스(Weaviate/ChromaDB)에 저장합니다
- 고유한 document_id를 생성하여 나중에 특정 문서 검색이 가능합니다

### 파라미터 설명
**file**: 업로드할 PDF 파일 (.pdf 확장자만 허용)
- 지원 포맷: PDF
- 최대 크기: 10MB 권장
- 텍스트가 포함된 PDF만 처리 가능

### 응답 (HTTP 200)
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
  "upload_timestamp": "2024-01-15T10:30:45.123456"
}
```

### 에러 응답
- **HTTP 400**: PDF 파일만 지원됩니다
- **HTTP 400**: PDF에서 충분한 텍스트를 추출할 수 없습니다
- **HTTP 500**: 벡터 저장 실패 또는 서버 오류

### 사용 예시
```bash
curl -X POST "http://localhost:7000/pdf/upload" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@algorithm_study.pdf"
```
"""

desc_get_documents = """
업로드된 모든 PDF 문서의 목록을 조회하는 API입니다.

### 기능 설명
- 벡터 데이터베이스에 저장된 모든 문서의 메타데이터를 조회합니다
- 각 문서의 청크 수, 업로드 시간 등의 정보를 제공합니다
- RAG 퀴즈 생성을 위한 권장 정보를 포함합니다

### 파라미터 설명
별도의 파라미터가 필요하지 않습니다.

### 응답 (HTTP 200)
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
  ]
}
```

### 에러 응답
- **HTTP 500**: 문서 목록 조회 오류

### 사용 예시
```bash
curl -X GET "http://localhost:7000/pdf/documents"
```
"""

desc_get_document_info = """
특정 문서의 상세 정보를 조회하는 API입니다.

### 기능 설명
- document_id로 특정 문서의 상세 메타데이터를 조회합니다
- RAG 퀴즈 생성을 위한 추가 분석 정보를 제공합니다
- 문서의 품질과 퀴즈 생성 적합성을 평가합니다

### 파라미터 설명
**document_id**: 조회할 문서의 고유 식별자 (URL 경로 파라미터)
- 형식: UUID 문자열
- 예시: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

### 응답 (HTTP 200)
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
  "rag_info": {
    "can_generate_quiz": true,
    "recommended_questions": 9,
    "content_quality": "high"
  }
}
```

### 에러 응답
- **HTTP 404**: 문서를 찾을 수 없습니다
- **HTTP 500**: 문서 정보 조회 오류

### 사용 예시
```bash
curl -X GET "http://localhost:7000/pdf/documents/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```
"""

desc_search_all_documents = """
업로드된 모든 문서에서 텍스트 검색을 수행하는 API입니다.

### 기능 설명
- 벡터 유사도 기반으로 모든 문서에서 관련 내용을 검색합니다
- 코사인 유사도를 계산하여 결과를 랭킹합니다
- 검색 결과는 유사도 순으로 정렬되어 반환됩니다

### 파라미터 설명
**query**: 검색할 텍스트 쿼리 (필수)
- 예시: "동적계획법", "머신러닝 알고리즘"

**top_k**: 반환할 결과 개수 (선택사항)
- 기본값: 5
- 최소값: 1
- 최대값: 20

### 응답 (HTTP 200)
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

### 에러 응답
- **HTTP 400**: 검색 쿼리가 비어있습니다
- **HTTP 500**: 검색 오류

### 사용 예시
```bash
curl -X GET "http://localhost:7000/pdf/search?query=동적계획법&top_k=5"
```
"""

desc_search_in_document = """
특정 문서 내에서만 텍스트 검색을 수행하는 API입니다.

### 기능 설명
- 지정된 document_id의 문서에서만 검색을 수행합니다
- RAG 퀴즈 생성을 위한 컨텍스트 추출에 최적화되어 있습니다
- 검색 결과를 결합하여 완전한 컨텍스트를 제공합니다

### 파라미터 설명
**document_id**: 검색할 문서의 고유 식별자 (URL 경로 파라미터)

**query**: 검색할 텍스트 쿼리 (필수)
- 예시: "알고리즘 복잡도", "데이터 구조"

**top_k**: 반환할 결과 개수 (선택사항)
- 기본값: 5
- 최소값: 1
- 최대값: 10

### 응답 (HTTP 200)
```json
{
  "message": "문서 내 검색 완료",
  "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "document_filename": "algorithm_study.pdf",
  "query": "동적계획법",
  "total_results": 5,
  "results": [
    {
      "doc_id": "a1b2c3d4_chunk_0",
      "text_preview": "동적계획법은 복잡한 문제를...",
      "full_text": "동적계획법은 복잡한 문제를 간단한 하위 문제로 나누어 해결하는...",
      "similarity": 0.8542,
      "chunk_index": 0
    }
  ],
  "rag_context": {
    "combined_text": "동적계획법은 복잡한 문제를 간단한 하위 문제로...",
    "context_length": 4280,
    "ready_for_rag": true
  }
}
```

### 에러 응답
- **HTTP 400**: 검색 쿼리가 비어있습니다
- **HTTP 404**: 문서를 찾을 수 없습니다
- **HTTP 500**: 문서 내 검색 오류

### 사용 예시
```bash
curl -X GET "http://localhost:7000/pdf/search/a1b2c3d4-e5f6-7890-abcd-ef1234567890?query=동적계획법&top_k=5"
```
"""

desc_health_check = """
PDF 벡터 서비스의 상태를 확인하는 API입니다.

### 기능 설명
- 서비스의 전반적인 상태를 점검합니다
- 현재 사용 중인 벡터 데이터베이스 정보를 제공합니다
- 사용 가능한 모든 엔드포인트 목록을 반환합니다

### 파라미터 설명
별도의 파라미터가 필요하지 않습니다.

### 응답 (HTTP 200 - 정상)
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
    "GET /pdf/search"
  ]
}
```

### 응답 (HTTP 503 - 비정상)
```json
{
  "status": "unhealthy",
  "error": "데이터베이스 연결 오류"
}
```

### 에러 응답
- **HTTP 503**: 서비스 비정상 상태 (데이터베이스 연결 오류 등)

### 사용 예시
```bash
curl -X GET "http://localhost:7000/pdf/health"
```
"""

desc_switch_database = """
벡터 데이터베이스를 다른 타입으로 전환하는 API입니다.

### 기능 설명
- 현재 사용 중인 벡터 DB를 다른 타입으로 실시간 전환합니다
- 지원되는 DB: Weaviate (권장), ChromaDB
- 전환 시 기존 데이터는 초기화되므로 주의가 필요합니다

### 파라미터 설명
**db_type**: 전환할 데이터베이스 타입 (필수)
- **"weaviate"**: 고성능, 확장성 우수 (프로덕션 권장)
- **"chroma"**: 가벼움, 메모리 효율적 (개발/테스트 권장)

### 응답 (HTTP 200)
```json
{
  "message": "데이터베이스가 weaviate으로 변경되었습니다",
  "previous_db": "chroma",
  "current_db": "weaviate",
  "total_documents": 0
}
```

### 에러 응답
- **HTTP 400**: 지원하지 않는 DB 타입
- **HTTP 500**: DB 전환 오류

### 주의사항
**경고**: 데이터베이스 전환 시 기존 모든 데이터가 삭제됩니다!
- 전환 후 PDF를 다시 업로드해야 합니다
- 되돌릴 수 없으므로 신중하게 사용하세요

### 사용 예시
```bash
# Weaviate로 전환 (프로덕션 권장)
curl -X POST "http://localhost:7000/pdf/switch-db?db_type=weaviate"

# ChromaDB로 전환 (개발/테스트 권장)
curl -X POST "http://localhost:7000/pdf/switch-db?db_type=chroma"
```
"""

desc_get_stats = """
벡터 데이터베이스의 상세 통계 정보를 조회하는 API입니다.

### 기능 설명
- 현재 벡터 DB의 성능 및 사용량 통계를 제공합니다
- 검색 및 업로드 성능 메트릭을 포함합니다
- 시스템 모니터링 및 최적화에 활용할 수 있습니다

### 파라미터 설명
별도의 파라미터가 필요하지 않습니다.

### 응답 (HTTP 200)
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

### 에러 응답
- **HTTP 500**: 통계 조회 오류

### 사용 예시
```bash
curl -X GET "http://localhost:7000/pdf/stats"
```
"""