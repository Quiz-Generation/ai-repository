
"""
🎯 Test API Routes
"""
from fastapi import APIRouter
from src.common.utils.logger import set_logger
from src.app.service import test as test_service

logger = set_logger("test")

router = APIRouter(tags=["test"])

@router.get("/test-stream")
async def test_stream():
    return await test_service.test_stream()

