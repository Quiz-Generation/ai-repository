# 🚀 Quiz Generation Streaming System

Redis 스트림을 활용하여 문제 생성을 실시간으로 스프링 서버에 전송하는 시스템입니다.

## 📋 시스템 구조

```
스프링 서버 → FastAPI → AI 에이전트 → Redis 스트림 → 스프링 서버
```

## 🔄 워크플로우

1. **스프링 서버**에서 문제 생성 요청
2. **FastAPI**에서 고유 요청 ID 생성 및 백그라운드 작업 시작
3. **AI 에이전트**가 배치별로 문제 생성 (3개씩)
4. **Redis 스트림**으로 각 배치 완료 시 즉시 전송
5. **스프링 서버**에서 스트림을 구독하여 실시간 진행 상황 확인

## 📡 API 엔드포인트

### 1. 문제 생성 요청 (스트리밍)
```http
POST /quiz/generate
```

**응답:**
```json
{
  "success": true,
  "message": "문제 생성이 시작되었습니다. Redis 스트림을 통해 실시간으로 진행 상황을 확인할 수 있습니다.",
  "request_id": "uuid-here",
  "stream_key": "quiz-generation-stream:uuid-here",
  "status": "started"
}
```

### 2. 스트림 구독 (스프링 서버용)
```http
GET /quiz/stream/{request_id}?count=10
```

**응답:**
```json
{
  "success": true,
  "request_id": "uuid-here",
  "messages": [
    {
      "message_id": "stream-id",
      "data": {
        "request_id": "uuid-here",
        "batch_num": 3,
        "total_batches": 4,
        "questions": [...],
        "questions_count": 3,
        "status": "batch_completed",
        "timestamp": "2024-01-01T12:00:00",
        "progress_percent": 75
      }
    }
  ],
  "message_count": 1
}
```

## 🔍 Redis 스트림 메시지 구조

### 배치 완료 메시지
```json
{
  "request_id": "uuid-here",
  "batch_num": 3,
  "total_batches": 4,
  "questions": [
    {
      "id": 1,
      "question": "문제 내용...",
      "choices": ["A", "B", "C", "D"],
      "correct_answer": "A",
      "explanation": "해설..."
    }
  ],
  "questions_count": 3,
  "status": "batch_completed",
  "timestamp": "2024-01-01T12:00:00",
  "progress_percent": 75,
  "batch_quality_score": 0.85
}
```

### 완료 메시지
```json
{
  "request_id": "uuid-here",
  "status": "completed",
  "total_questions": 5,
  "final_questions": [...],
  "timestamp": "2024-01-01T12:00:00",
  "progress_percent": 100,
  "total_time": 15.5,
  "avg_quality_score": 0.875,
  "failed_batches": 0
}
```

### 에러 메시지
```json
{
  "request_id": "uuid-here",
  "status": "error",
  "error_message": "에러 내용...",
  "timestamp": "2024-01-01T12:00:00",
  "batch_num": 2
}
```

## 🚀 스프링 서버 연동 방법

### 1. 문제 생성 요청
```java
// 문제 생성 요청
POST /quiz/generate
{
  "file_id": "file-uuid",
  "num_questions": 5,
  "difficulty": "medium",
  "question_type": "multiple_choice"
}

// 응답에서 request_id 추출
String requestId = response.getRequestId();
```

### 2. 스트림 구독 (폴링 방식)
```java
// 주기적으로 스트림 조회 (예: 1초마다)
while (true) {
    try {
        // 스트림 메시지 조회
        GET /quiz/stream/{requestId}?count=10
        
        // 메시지 처리
        for (Message message : messages) {
            String status = message.getData().getStatus();
            
            if ("batch_completed".equals(status)) {
                // 배치 완료 처리
                List<Question> questions = message.getData().getQuestions();
                processQuestions(questions);
            } else if ("completed".equals(status)) {
                // 전체 완료 처리
                processCompletion(message.getData());
                break;
            } else if ("error".equals(status)) {
                // 에러 처리
                handleError(message.getData());
                break;
            }
        }
        
        Thread.sleep(1000); // 1초 대기
        
    } catch (Exception e) {
        // 에러 처리
        handleError(e);
    }
}
```

## ⚙️ 설정

### Redis 설정
```python
# src/common/conf/settings.py
REDIS_HOST = "localhost"
REDIS_PORT = 6379
```

### 스트림 키 형식
- 문제 생성 스트림: `quiz-generation-stream:{request_id}`
- 스트림 만료 시간: 24시간

## 🧪 테스트

### 테스트 스크립트 실행
```bash
python test_streaming.py
```

### Redis CLI로 스트림 확인
```bash
# Redis 연결
redis-cli

# 스트림 키 확인
KEYS quiz-generation-stream:*

# 특정 스트림 메시지 조회
XRANGE quiz-generation-stream:test-request-123 - +
```

## 📊 모니터링

### 로그 확인
```bash
# FastAPI 로그
tail -f logs/fastapi.log

# Redis 연결 로그
tail -f logs/redis.log
```

### 진행 상황 추적
- `batch_num`: 현재 배치 번호
- `total_batches`: 전체 배치 수
- `progress_percent`: 진행률 (%)
- `status`: 현재 상태
- `timestamp`: 메시지 생성 시간

## 🔧 문제 해결

### 일반적인 문제들

1. **Redis 연결 실패**
   - Redis 서버 상태 확인
   - 호스트/포트 설정 확인

2. **스트림 메시지 누락**
   - 스트림 만료 시간 확인 (24시간)
   - 메시지 ID 순서 확인

3. **배치 생성 실패**
   - OpenAI API 키 확인
   - 토큰 제한 확인
   - 로그에서 에러 메시지 확인

## 📈 성능 최적화

- **배치 크기**: 3개씩 생성 (토큰 제한 고려)
- **병렬 처리**: asyncio.gather로 배치 동시 생성
- **스트림 만료**: 24시간 후 자동 정리
- **에러 처리**: 배치별 개별 에러 처리

## 🎯 다음 단계

1. **WebSocket 지원**: 실시간 양방향 통신
2. **배치 크기 조정**: 동적 배치 크기 설정
3. **재시도 로직**: 실패한 배치 자동 재시도
4. **모니터링 대시보드**: 실시간 진행 상황 시각화
