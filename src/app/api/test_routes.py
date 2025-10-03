
"""
🎯 Test API Routes
"""
from fastapi import APIRouter

from src.app.service import test as test_service
from src.common.utils.logger import set_logger

logger = set_logger("api.test")

router = APIRouter(tags=["test"])

@router.get("/test-stream")
async def test_stream():
    return await test_service.test_stream()

