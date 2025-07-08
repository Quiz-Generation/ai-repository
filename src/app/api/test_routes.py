
"""
🎯 Test API Routes
"""
from fastapi import APIRouter
from src.common.utils.logger import set_logger
from src.app.service import test as test_service

logger = set_logger("api.test")

router = APIRouter(tags=["test"])

@router.get("/test-pubsub")
async def test_pubsub():
    return await test_service.test_pub()

@router.get("/test-subscribe")
async def test_sub():
    return await test_service.test_sub()
