"""
📄 Document API Routes (간단 테스트용)
"""
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from ..schemas.document_schema import DocumentUploadResponse
from ..service.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


async def get_document_service() -> DocumentService:
    """문서 서비스 의존성 주입"""
    return DocumentService()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service)
):
    """📄 PDF 문서 업로드 (동적 로더 선택)"""
    try:
        result = await service.upload_document(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/loaders", response_model=Dict[str, Any])
async def get_pdf_loaders_info(
    service: DocumentService = Depends(get_document_service)
):
    """🔧 PDF 로더 목록 및 선택 규칙 조회"""
    try:
        result = await service.get_loader_selection_info()
        return {
            "message": "PDF 로더 정보 조회 성공",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
