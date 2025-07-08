import asyncio
from src.common.conf.settings import settings
from redis.asyncio import Redis
from src.common.utils.logger import set_logger

REDIS_URL = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"
STREAM_KEY = "quiz_generation_stream"
PUBSUB_CHANNEL = "quiz_pubsub_channel"
TEST_PUBSUB_CHANNEL = "test-group"

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

# --- Redis Pub/Sub Publisher ---
async def publish_quiz_pubsub(message: str):
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    await redis.publish(PUBSUB_CHANNEL, message)
    await redis.close()

# --- Redis Pub/Sub Subscriber ---
async def subscribe_quiz_pubsub():
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe(PUBSUB_CHANNEL)
    print(f"[PubSub] {PUBSUB_CHANNEL} 구독 시작")
    async for msg in pubsub.listen():
        if msg["type"] == "message":
            print(f"[PubSub] 받은 메시지: {msg['data']}")

# --- 테스트용 Pub/Sub Publisher ---
async def test_publish():
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    msg = "hello from publisher!"
    await redis.publish(TEST_PUBSUB_CHANNEL, msg)
    logger.info(f"[TestPub] 발행: {msg}")
    await redis.close()

# --- 테스트용 Pub/Sub Subscriber ---
async def test_subscribe():
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe(TEST_PUBSUB_CHANNEL)
    logger.info(f"[TestSub] {TEST_PUBSUB_CHANNEL} 구독 시작")
    async for msg in pubsub.listen():
        if msg["type"] == "message":
            logger.info(f"[TestSub] 받은 메시지: {msg['data']}")
            break  # 한 번만 받고 종료(테스트 목적)
    await pubsub.unsubscribe(TEST_PUBSUB_CHANNEL)
    await redis.close()

# --- 사용 예시 (테스트 시 아래 주석 해제) ---
# asyncio.run(push_quiz_to_stream({"quiz_id": "123", "status": "done", "result": "..."}))
# asyncio.run(consume_quiz_stream())
# asyncio.run(publish_quiz_pubsub("퀴즈 생성 완료!"))
# asyncio.run(subscribe_quiz_pubsub())
asyncio.run(test_subscribe())
# asyncio.run(test_publish())