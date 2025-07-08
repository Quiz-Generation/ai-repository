from src.common.redis.connect import test_publish, test_subscribe

async def test_pub():
    await test_publish()
    return {"message": "Hello, World! test_pub"}

async def test_sub():
    await test_subscribe()
    return {"message": "Hello, World! test_sub"}