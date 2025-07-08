from src.common.redis.connect import push_quiz_to_stream



async def test_stream():
    await push_quiz_to_stream({"quiz_id": "123", "status": "done", "result": "..."})
    return {"message": "Hello, World! test_stream"}