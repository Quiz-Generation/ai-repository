import asyncio
from src.common.conf.settings import settings
from redis.asyncio import Redis
from src.common.utils.logger import set_logger

REDIS_URL = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"
STREAM_KEY = "test-stream"

logger = set_logger('redis')

# --- Redis Stream Producer ---
async def push_quiz_to_stream(quiz_data: dict):
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    # XADD: stream에 데이터 추가
    await redis.xadd(STREAM_KEY, quiz_data)
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


# --- 사용 예시 (테스트 시 아래 주석 해제) ---
# asyncio.run(push_quiz_to_stream({"quiz_id": "123", "status": "done", "result": "..."}))
# asyncio.run(consume_quiz_stream())
# asyncio.run(publish_quiz_pubsub("퀴즈 생성 완료!"))
# asyncio.run(subscribe_quiz_pubsub())
# asyncio.run(test_subscribe())
# asyncio.run(test_publish())