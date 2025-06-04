"""
퀴즈 생성 API Swagger 문서 설명
PDF 기반 RAG 퀴즈 생성 시스템의 각 엔드포인트별 상세 설명
"""

desc_generate_quiz = """
🧠 **PDF 문서 기반 퀴즈 자동 생성 - 메인 기능**

**🤖 AI가 PDF를 분석하여 자동으로 최적의 퀴즈를 생성합니다**

### 핵심 특징
- ✨ **토픽 자동 추출**: PDF 내용을 분석하여 핵심 주제들을 자동 추출
- 🎯 **RAG 최적화**: 관련성 높은 컨텍스트만 선별하여 고품질 문제 생성
- 🔄 **지능형 난이도 조절**: 요청한 난이도에 맞는 문제 유형과 복잡도 자동 선택
- 📊 **품질 보장**: AI가 생성한 문제를 자동으로 검증

### 처리 과정
1. 📄 **문서 확인**: 업로드된 PDF 문서 존재 및 상태 확인
2. 🤖 **토픽 자동 추출**: AI가 PDF 내용을 분석하여 핵심 주제 추출
3. 🔍 **RAG 컨텍스트 검색**: 추출된 토픽 기반으로 최적 컨텍스트 검색
4. ⚡ **LLM 퀴즈 생성**: 컨텍스트와 토픽을 바탕으로 문제 생성
5. ✅ **품질 검증**: 생성된 문제의 품질 자동 검증 및 최적화

### 요청 파라미터
**document_id** (필수): 업로드된 PDF 문서 ID
- 형식: UUID 문자열
- 예시: "f7dbd017-426e-4919-8a88-feda68949615"

**num_questions** (선택): 생성할 문제 수
- 기본값: 5
- 범위: 1-20개
- 권장: 5-10개

**difficulty** (선택): 기본 난이도
- 기본값: "medium"
- 옵션: "easy", "medium", "hard"
- 참고: 각 문제별로 자동 조정됨

**question_types** (선택): 문제 유형 목록
- 생략 시: 자동 선택
- 옵션: "multiple_choice", "short_answer", "fill_blank", "true_false"

**language** (선택): 언어 설정
- 기본값: "ko"
- 옵션: "ko", "en"

### 요청 예시 (간단)
```json
{
    "document_id": "f7dbd017-426e-4919-8a88-feda68949615",
    "num_questions": 5,
    "difficulty": "medium"
}
```
→ AI가 자동으로 토픽을 추출하고 적절한 문제 유형을 선택합니다

### 요청 예시 (커스텀)
```json
{
    "document_id": "f7dbd017-426e-4919-8a88-feda68949615",
    "num_questions": 8,
    "difficulty": "hard",
    "question_types": ["multiple_choice", "short_answer"],
    "language": "ko"
}
```

### 응답 (HTTP 200)
```json
{
    "message": "퀴즈 생성 성공",
    "quiz_id": "12345678-abcd-1234-efgh-567890abcdef",
    "document_id": "f7dbd017-426e-4919-8a88-feda68949615",
    "questions": [
        {
            "question": "동적계획법의 핵심 원리는 무엇입니까?",
            "question_type": "multiple_choice",
            "correct_answer": "중복 계산을 피하기 위해 결과를 저장",
            "options": [
                "중복 계산을 피하기 위해 결과를 저장",
                "모든 경우의 수를 확인",
                "재귀함수만 사용",
                "반복문으로만 해결"
            ],
            "explanation": "동적계획법은 중복되는 하위 문제의 계산 결과를 저장하여...",
            "difficulty": "easy",
            "topic": "동적계획법"
        }
    ],
    "total_questions": 5,
    "difficulty": "medium",
    "generation_time": 12.34,
    "generation_info": {
        "llm_model_used": "gpt-4o-mini",
        "extracted_topics": ["동적계획법", "재귀식", "최적해"],
        "contexts_used": 8,
        "avg_context_similarity": 0.82
    },
    "quality_assessment": {
        "overall_quality": "excellent",
        "valid_questions": 5
    }
}
```

### 에러 응답
- **HTTP 400**: 잘못된 요청 (document_id 없음, 문제 수 범위 초과 등)
- **HTTP 400**: 퀴즈 생성 실패 (적절한 컨텍스트를 찾을 수 없음)
- **HTTP 500**: 서버 오류

### 사용 팁
- `difficulty`는 기본값이며, 각 문제별로 자동 조정됩니다
- 특정 주제에 집중하려면 해당 내용이 많은 PDF를 업로드하세요
- 더 나은 품질을 위해 텍스트가 풍부한 PDF를 사용하세요
"""

desc_extract_topics = """
📚 **문서 토픽 자동 추출**

### 기능 설명
- 업로드된 PDF 문서에서 퀴즈 생성에 적합한 핵심 토픽을 자동 추출합니다
- AI가 문서 내용을 분석하여 중요한 주제들을 식별합니다
- 퀴즈 생성 전 문서의 내용을 미리 파악할 때 유용합니다

### 파라미터 설명
**document_id**: 분석할 문서의 고유 식별자 (URL 경로 파라미터)
- 형식: UUID 문자열

**max_topics**: 추출할 최대 토픽 수 (선택사항)
- 기본값: 10
- 범위: 1-20개

### 응답 (HTTP 200)
```json
{
    "message": "토픽 추출 완료",
    "document_id": "f7dbd017-426e-4919-8a88-feda68949615",
    "extracted_topics": [
        "동적계획법",
        "최적화 문제",
        "재귀식",
        "메모이제이션",
        "시간복잡도"
    ],
    "extraction_info": {
        "total_topics_found": 5,
        "document_analysis_time": 3.45,
        "content_quality": "high"
    }
}
```

### 에러 응답
- **HTTP 404**: 문서를 찾을 수 없습니다
- **HTTP 500**: 토픽 추출 오류

### 사용 예시
```bash
curl -X GET "http://localhost:7000/quiz/topics/f7dbd017-426e-4919-8a88-feda68949615?max_topics=5"
```
"""

desc_switch_llm = """
🔄 **LLM 모델 교체**

### 기능 설명
- 퀴즈 생성에 사용할 LLM 모델을 동적으로 교체할 수 있습니다
- 다양한 LLM 제공업체를 지원합니다 (OpenAI, Anthropic, 한국어 로컬 모델 등)
- 모델별 특성에 따라 퀴즈 품질과 스타일이 달라집니다

### 파라미터 설명
**provider**: LLM 제공업체 (필수)
- 옵션: "openai", "anthropic", "korean_local", "huggingface"

**model_name**: 모델 이름 (필수)
- OpenAI 예시: "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"
- Anthropic 예시: "claude-3-sonnet", "claude-3-haiku"

**api_key**: API 키 (선택사항)
- 기본값: 환경변수에서 자동 로드
- 새로운 키를 사용할 때만 제공

### 요청 예시
```json
{
    "provider": "openai",
    "model_name": "gpt-4",
    "api_key": "sk-..."
}
```

### 응답 (HTTP 200)
```json
{
    "message": "LLM 모델 교체 완료",
    "previous_model": {
        "provider": "openai",
        "model_name": "gpt-4o-mini"
    },
    "current_model": {
        "provider": "openai",
        "model_name": "gpt-4"
    },
    "switch_time": "2024-01-15T10:30:45.123456"
}
```

### 에러 응답
- **HTTP 400**: 지원하지 않는 LLM 제공업체
- **HTTP 400**: 잘못된 모델 이름
- **HTTP 401**: API 키 인증 실패
- **HTTP 500**: 모델 교체 오류
"""

desc_get_models = """
📋 **사용 가능한 LLM 모델 목록 조회**

### 기능 설명
- 현재 시스템에서 지원하는 모든 LLM 제공업체와 모델을 조회합니다
- 각 모델의 특성과 권장 사용처를 제공합니다
- 모델 교체 시 참고용으로 활용할 수 있습니다

### 파라미터 설명
별도의 파라미터가 필요하지 않습니다.

### 응답 (HTTP 200)
```json
{
    "message": "사용 가능한 LLM 모델 목록",
    "current_model": {
        "provider": "openai",
        "model_name": "gpt-4o-mini"
    },
    "available_providers": [
        {
            "provider": "openai",
            "models": ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            "status": "available",
            "description": "OpenAI GPT 시리즈"
        },
        {
            "provider": "anthropic",
            "models": ["claude-3-sonnet", "claude-3-haiku"],
            "status": "coming_soon",
            "description": "Anthropic Claude 시리즈"
        }
    ],
    "recommendations": {
        "korean_quiz": "OpenAI gpt-4o-mini (한국어 최적화)",
        "high_quality": "OpenAI gpt-4 (최고 품질)",
        "fast_generation": "OpenAI gpt-3.5-turbo (빠른 생성)"
    }
}
```
"""

desc_health_check = """
🔍 **퀴즈 생성 서비스 상태 확인**

### 기능 설명
- 퀴즈 생성 서비스의 전반적인 상태를 점검합니다
- 현재 사용 중인 LLM 모델과 벡터 DB 정보를 제공합니다
- 사용 가능한 모든 기능과 엔드포인트 목록을 반환합니다

### 파라미터 설명
별도의 파라미터가 필요하지 않습니다.

### 응답 (HTTP 200 - 정상)
```json
{
    "status": "healthy",
    "service": "PDF RAG Quiz Generation Service",
    "llm_model": "gpt-4o-mini",
    "llm_provider": "openai",
    "vector_db": "weaviate",
    "supported_features": [
        "PDF 기반 퀴즈 생성",
        "RAG 컨텍스트 검색",
        "동적 토픽 추출",
        "다양한 문제 유형",
        "난이도별 문제 생성",
        "LLM 모델 교체",
        "문제 품질 검증"
    ],
    "available_difficulties": ["easy", "medium", "hard"],
    "available_question_types": [
        "multiple_choice", "short_answer", "fill_blank", "true_false"
    ],
    "supported_llm_providers": ["openai", "anthropic", "korean_local"],
    "endpoints": [
        "POST /quiz/generate",
        "GET /quiz/topics/{document_id}",
        "POST /quiz/switch-llm",
        "GET /quiz/health"
    ]
}
```

### 응답 (HTTP 503 - 비정상)
```json
{
    "status": "unhealthy",
    "error": "LLM 서비스 연결 실패"
}
```
"""
