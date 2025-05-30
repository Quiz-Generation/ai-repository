"""
최적화된 PDF 처리 API 라우트
"""
import os
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from app.services.pdf_extractor import PDFExtractorService
from app.services import optimized_pdf_extractor  as optimized_pdf_extractor_service

router = APIRouter(prefix="/pdf", tags=["PDF Processing"])

# 업로드 디렉토리 설정
UPLOAD_DIR = "static/temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/health")
async def health_check():
    """
    PDF 서비스 상태 확인
    """
    try:
        # 기본 추출기로 간단한 테스트
        service = optimized_pdf_extractor_service.OptimizedPDFService()

        return JSONResponse(content={
            "status": "healthy",
            "default_extractor": optimized_pdf_extractor_service.OptimizedPDFExtractorFactory.get_default_extractor(),
            "available_extractors": optimized_pdf_extractor_service.OptimizedPDFExtractorFactory.get_available_extractors()
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@router.post("/upload-and-extract")
async def upload_and_extract_pdf(
    file: UploadFile = File(...),
    extractor_type: str = Query(default="pymupdf", description="추출기 타입 (기본: pymupdf)")
):
    """
    PDF 파일 업로드 및 텍스트 추출

    - **file**: PDF 파일
    - **extractor_type**: 추출기 타입 (pymupdf, pypdf2, langchain_pymupdf)
    """

    # 파일 검증
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    # 파일 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # PDF 추출
        result = PDFExtractorService(extractor_type).extract_text(file_path)

        # 임시 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

        return JSONResponse(content={
            "message": "PDF 추출 완료",
            "filename": file.filename,
            **result
        })

    except Exception as e:
        # 오류 시 임시 파일 정리
        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류 발생: {str(e)}")




