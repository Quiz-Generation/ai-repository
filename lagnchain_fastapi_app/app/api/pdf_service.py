"""
PDF 업로드 → 벡터 DB 저장 API 라우터

엔드포인트:
- POST /pdf/upload: PDF 파일 업로드 및 벡터 저장
- GET /pdf/search: 벡터 검색
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import tempfile
import os
import logging

# 기존 서비스들 import
from lagnchain_fastapi_app.app.services.pdf_service import DynamicPDFService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pdf", tags=["PDF Vector"])

# 전역 서비스 인스턴스들 (싱글톤 패턴)
pdf_service = DynamicPDFService()

@router.get("/health")
async def health_check() -> JSONResponse:
    """벡터 DB 서비스 상태 확인"""
    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "PDF Vector Service",
                "vector_db": "chroma",
                "endpoints": [
                    "POST /pdf/upload",
                    "GET /pdf/search",
                    "GET /pdf/health"
                ]
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )