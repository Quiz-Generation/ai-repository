import time
from fastapi import APIRouter, Form, Request, UploadFile, File
from fastapi.responses import JSONResponse

from src.app.docs import document as document_docs
from src.app.service import document as document_service
from src.common.utils.logger import set_logger

logger = set_logger("api.document")

router = APIRouter(tags=["documents"])


# 🚀 1. PDF 업로드 및 벡터 저장 (+ 문서 ID 반환)
@router.post(
        "/upload",
        summary="PDF 업로드 및 벡터 저장",
        description=document_docs.upload_pdf_to_vector_db_description,
    )
async def upload_pdf_to_vector_db(
    request: Request,
    file: UploadFile = File(...),
) -> JSONResponse:
    return await document_service.upload_document(
        logger=logger,
        vector_db=request.app.state.vector_db,
        file=file
    )



# 💥 5. 벡터 DB 모든 데이터 삭제 (위험한 작업)
@router.delete("/clear-all")
async def clear_all_documents(
    request: Request,
    confirm_token: str = Form(..., description="삭제 확인 토큰: CLEAR_ALL_CONFIRM"),
) -> JSONResponse:
    """
    💥 벡터 DB의 모든 데이터 삭제 (위험한 작업)

    ⚠️ 주의: 이 작업은 되돌릴 수 없습니다!
    confirm_token에 "CLEAR_ALL_CONFIRM"을 입력해야 합니다.
    """
    return await document_service.clear_all_documents(
        logger=logger,
        vector_db=request.app.state.vector_db,
        confirm_token=confirm_token
    )
