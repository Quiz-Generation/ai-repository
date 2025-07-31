import time

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from src.app.document import docs as document_docs
from src.app.document.service import DocumentService
from src.common.utils.logger import set_logger

logger = set_logger("api.document")

router = APIRouter(tags=["documents"])

def get_document_service(
    request: Request
) -> DocumentService:
    return DocumentService(
        logger=logger,
        vector_db=request.app.state.vector_db,
    )

@router.post(
    "/upload",
    summary="PDF 업로드 및 벡터 저장",
    description=document_docs.upload_pdf_to_vector_db_description,
)
async def upload_pdf_to_vector_db(
    request: Request,
    document_service: DocumentService = Depends(get_document_service),
    file: UploadFile = File(...),
) -> JSONResponse:
    return await document_service.upload_document(
        file=file
    )



@router.delete(
    "/clear-all",
    summary="벡터 DB의 모든 데이터 삭제 (위험한 작업)",
    description=document_docs.clear_all_documents_description,
)
async def clear_all_documents(
    request: Request,
    document_service: DocumentService = Depends(get_document_service),
    confirm_token: str = Form(..., description="삭제 확인 토큰: CLEAR_ALL_CONFIRM"),
) -> JSONResponse:
    """
    벡터 DB의 모든 데이터 삭제 (위험한 작업)

    주의: 이 작업은 되돌릴 수 없습니다!
    confirm_token에 "CLEAR_ALL_CONFIRM"을 입력해야 합니다.
    """
    return await document_service.clear_all_documents(
        confirm_token=confirm_token
    )