import asyncio
import json
import uuid
from datetime import datetime
from src.common.conf.settings import settings
from redis.asyncio import Redis
from src.common.utils.logger import set_logger

REDIS_URL = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"
STREAM_KEY = "quiz-stream"
QUIZ_STREAM_KEY = "quiz-stream"  # 문제 생성 전용 스트림

logger = set_logger('redis')

# Redis 연결 URL 로깅
logger.info(f"Redis 연결 URL: {REDIS_URL}")
logger.info(f"Redis 호스트: {settings.REDIS_HOST}")
logger.info(f"Redis 포트: {settings.REDIS_PORT}")

# --- Redis Stream Producer ---
async def push_quiz_to_stream(quiz_data: dict):
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    # XADD: stream에 데이터 추가
    await redis.xadd(STREAM_KEY, quiz_data)
    await redis.close()

# --- 문제 생성 배치별 Redis 스트림 전송 (스프링 서버용) ---
async def push_quiz_batch_to_stream(
    request_id: str,
    batch_num: int,
    questions: list = None,
    total_batches: int = 1,
    status: str = "processing",
    metadata: dict = None
):
    """
    문제 생성 배치를 Redis 스트림으로 전송 (스프링 서버로 실시간 전송)
    """
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    
    try:
        # 스트림 키 생성 (요청별로 구분)
        # stream_key = f"{QUIZ_STREAM_KEY}:{request_id}"
        stream_key = f"{QUIZ_STREAM_KEY}"
        
        # 전송할 데이터 구성 (questions가 있을 때만 포함)
        batch_data = {
            "request_id": request_id,
            "batch_num": batch_num,
            "total_batches": total_batches,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "progress_percent": int((batch_num / total_batches) * 100)
        }
        
        # questions가 있고 실제 데이터가 있을 때만 추가
        if questions and len(questions) > 0:
            batch_data["questions"] = json.dumps(questions, ensure_ascii=False)
            batch_data["questions_count"] = len(questions)
        
        # 메타데이터가 있으면 추가 (Redis 호환성을 위해 리스트를 JSON 문자열로 변환)
        if metadata:
            # 메타데이터의 리스트 타입을 JSON 문자열로 변환
            processed_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, list):
                    processed_metadata[key] = json.dumps(value, ensure_ascii=False)
                else:
                    processed_metadata[key] = value
            batch_data.update(processed_metadata)
        
        # Redis 스트림에 추가
        await redis.xadd(stream_key, batch_data)
        
        # 스트림 만료 시간 설정 (24시간)
        await redis.expire(stream_key, 86400)
        
        if questions and len(questions) > 0:
            logger.info(f"✅ 배치 {batch_num}/{total_batches} Redis 스트림 전송 완료: {len(questions)}개 문제")
        else:
            logger.info(f"✅ 배치 {batch_num}/{total_batches} 상태 알림 Redis 스트림 전송 완료")
        
    except Exception as e:
        logger.error(f"❌ Redis 스트림 전송 실패: {e}")
    finally:
        await redis.close()

# --- 문제 생성 완료 알림 전송 ---
async def push_quiz_completion_to_stream(
    request_id: str,
    total_questions: int,
    final_questions: list,
    metadata: dict = None
):
    """
    문제 생성 완료를 Redis 스트림으로 전송
    """
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    
    try:
        # stream_key = f"{QUIZ_STREAM_KEY}:{request_id}"
        stream_key = f"{QUIZ_STREAM_KEY}"
        
        completion_data = {
            "request_id": request_id,
            "status": "completed",
            "total_questions": total_questions,
            "final_questions": json.dumps(final_questions, ensure_ascii=False),  # 리스트를 JSON 문자열로 변환
            "timestamp": datetime.now().isoformat(),
            "progress_percent": 100
        }
        
        if metadata:
            # 메타데이터의 리스트 타입을 JSON 문자열로 변환
            processed_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, list):
                    processed_metadata[key] = json.dumps(value, ensure_ascii=False)
                else:
                    processed_metadata[key] = value
            completion_data.update(processed_metadata)
        
        await redis.xadd(stream_key, completion_data)
        await redis.expire(stream_key, 86400)
        
        logger.info(f"🎉 문제 생성 완료 Redis 스트림 전송: {total_questions}개 문제")
        
    except Exception as e:
        logger.error(f"❌ 완료 알림 Redis 스트림 전송 실패: {e}")
    finally:
        await redis.close()

# --- 문제 생성 에러 알림 전송 ---
async def push_quiz_error_to_stream(
    request_id: str,
    error_message: str,
    batch_num: int = None,
    metadata: dict = None
):
    """
    문제 생성 에러를 Redis 스트림으로 전송
    """
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    
    try:
        # stream_key = f"{QUIZ_STREAM_KEY}:{request_id}"
        stream_key = f"{QUIZ_STREAM_KEY}"
        
        error_data = {
            "request_id": request_id,
            "status": "error",
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        if batch_num is not None:
            error_data["batch_num"] = batch_num
            
        if metadata:
            # 메타데이터의 리스트 타입을 JSON 문자열로 변환
            processed_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, list):
                    processed_metadata[key] = json.dumps(value, ensure_ascii=False)
                else:
                    processed_metadata[key] = value
            error_data.update(processed_metadata)
        
        await redis.xadd(stream_key, error_data)
        await redis.expire(stream_key, 86400)
        
        logger.error(f"❌ 에러 Redis 스트림 전송: {error_message}")
        
    except Exception as e:
        logger.error(f"❌ 에러 알림 Redis 스트림 전송 실패: {e}")
    finally:
        await redis.close()

# --- Redis Stream Consumer ---
async def consume_quiz_stream():
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    group = "quiz_consumers"
    consumer = "worker1"
    try:
        await redis.xgroup_create(STREAM_KEY, group, id='0', mkstream=True)
    except Exception:
        pass  # 이미 그룹이 있으면 무시
    while True:
        resp = await redis.xreadgroup(group, consumer, {STREAM_KEY: '>'}, count=1, block=5000)
        if resp:
            for stream, messages in resp:
                for msg_id, msg in messages:
                    print(f"[Stream] 받은 메시지: {msg}")
                    await redis.xack(STREAM_KEY, group, msg_id)
        await asyncio.sleep(1)

# --- 스프링 서버용 스트림 구독 함수 ---
async def get_quiz_stream_messages(request_id: str, count: int = 10):
    """
    특정 요청의 문제 생성 스트림 메시지들을 조회 (스프링 서버에서 호출)
    """
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    
    try:
        # stream_key = f"{QUIZ_STREAM_KEY}:{request_id}"
        stream_key = f"{QUIZ_STREAM_KEY}"
        
        # 최근 메시지들 조회
        messages = await redis.xrevrange(stream_key, count=count)
        
        # 메시지 형식 변환 (JSON 문자열을 다시 파싱)
        formatted_messages = []
        for msg_id, msg_data in messages:
            # JSON 문자열로 저장된 리스트 데이터를 파싱
            processed_data = {}
            for key, value in msg_data.items():
                if key in ["questions", "final_questions"] and isinstance(value, str):
                    try:
                        processed_data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        processed_data[key] = value  # 파싱 실패 시 원본 값 유지
                else:
                    processed_data[key] = value
            
            formatted_messages.append({
                "message_id": msg_id,
                "data": processed_data
            })
        
        return formatted_messages
        
    except Exception as e:
        logger.error(f"❌ 스트림 메시지 조회 실패: {e}")
        return []
    finally:
        await redis.close()

# --- 사용 예시 (테스트 시 아래 주석 해제) ---
# asyncio.run(push_quiz_to_stream({"quiz_id": "123", "status": "done", "result": "..."}))
# asyncio.run(consume_quiz_stream())
# asyncio.run(publish_quiz_pubsub("퀴즈 생성 완료!"))
# asyncio.run(subscribe_quiz_pubsub())
# asyncio.run(test_subscribe())
# asyncio.run(test_publish())