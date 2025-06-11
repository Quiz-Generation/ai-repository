"""
📄 Document API Routes - Simplified
"""
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Query
from fastapi.responses import JSONResponse

from ..service.document_service import DocumentService
from ..service.vector_db_service import VectorDBService
from ..helper.pdf_loader_helper import PDFLoaderHelper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# 서비스 의존성 주입
async def get_document_service() -> DocumentService:
    """문서 서비스 의존성 주입"""
    return DocumentService()

async def get_vector_service() -> VectorDBService:
    """벡터 DB 서비스 의존성 주입"""
    return VectorDBService()


# 🚀 1. PDF 업로드 및 벡터 저장 (+ 문서 ID 반환)
@router.post("/upload")
async def upload_pdf_to_vector_db(
    file: UploadFile = File(...),
    vector_db_type: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    doc_service: DocumentService = Depends(get_document_service),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    📄 PDF 파일 업로드 및 벡터 DB 저장 (문서 ID 반환)
    """
    try:
        logger.info("=" * 50)
        logger.info("STEP1 PDF 업로드 및 벡터 저장 시작")

        # 파일 검증
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다")

        # PDF 특성 분석 및 최적 로더 선택
        logger.info("STEP2 PDF 특성 분석 시작")
        analysis_result = await PDFLoaderHelper.analyze_pdf_characteristics(file)

        # PDF 내용 추출
        logger.info("STEP3 PDF 내용 추출 시작")
        extraction_result = await doc_service.process_pdf_with_dynamic_selection(
            file, analysis_result.recommended_loader
        )

        if not extraction_result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"PDF 처리 실패: {extraction_result.get('error', 'Unknown error')}"
            )

        # 벡터 DB 초기화 (지정된 타입 또는 자동 선택)
        logger.info("STEP4 벡터 DB 초기화 시작")
        if vector_db_type:
            selected_db = await vector_service.initialize_vector_db(vector_db_type)
        else:
            selected_db = await vector_service.initialize_vector_db()

        # 메타데이터 구성
        metadata = {
            "filename": file.filename,
            "file_size": file.size,
            "pdf_loader": extraction_result["loader_used"],
            "language": analysis_result.language,
            "upload_timestamp": extraction_result["processing_time"],
            "source": "document_upload"
        }

        # 벡터 DB에 저장
        logger.info("STEP5 벡터 DB 저장 시작")
        vector_result = await vector_service.store_pdf_content(
            pdf_content=extraction_result["content"],
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 🔥 문서 ID 생성 (첫 번째 저장된 ID 사용)
        document_id = vector_result.get("stored_ids", [None])[0] if vector_result.get("stored_ids") else None

        # 결과 반환
        response_data = {
            "success": vector_result["success"],
            "message": "PDF 업로드 및 벡터 저장 완료",
            "document_id": document_id,  # 🎯 문서 ID 반환
            "filename": file.filename,
            "vector_db_type": selected_db,
            "chunk_count": vector_result.get("chunk_count", 0),
            "stored_document_count": vector_result.get("stored_document_count", 0)
        }

        if not vector_result["success"]:
            response_data["error"] = vector_result.get("error")

        logger.info("SUCCESS PDF 업로드 완료")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"ERROR PDF 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 🔄 2. 벡터 DB 스위칭
@router.post("/vector-switch")
async def switch_vector_db(
    db_type: str = Form(..., description="전환할 벡터 DB 타입 (milvus/faiss)"),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    🔄 벡터 DB 타입 전환
    """
    try:
        logger.info(f"STEP_SWITCH 벡터 DB 전환 시작: {db_type}")

        # 벡터 DB 전환
        success = await vector_service.switch_vector_db(db_type)

        if success:
            # 전환 후 상태 조회
            status = await vector_service.get_vector_db_status()

            response_data = {
                "success": True,
                "message": f"{db_type.upper()} 벡터 DB로 전환 완료",
                "current_db_type": status.get("current_db_type"),
                "document_count": status.get("current_db_health", {}).get("document_count", 0)
            }
        else:
            response_data = {
                "success": False,
                "message": f"{db_type.upper()} 벡터 DB 전환 실패"
            }

        logger.info(f"SUCCESS 벡터 DB 전환 결과: {success}")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"ERROR 벡터 DB 전환 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 📋 3. 현재 벡터 DB의 모든 문서 조회
@router.get("/all-documents")
async def get_all_documents(
    limit: Optional[int] = Query(None, description="조회할 문서 수 제한"),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    📋 현재 벡터 DB에 저장된 모든 문서 조회
    """
    try:
        logger.info("STEP_DOCS 모든 문서 조회 시작")

        # 모든 문서 조회
        result = await vector_service.get_all_documents(limit)

        if result["success"]:
            response_data = {
                "success": True,
                "message": "문서 조회 완료",
                "vector_db_type": result["vector_db_type"],
                "total_documents": result["total_documents"],
                "total_files": result["total_files"],
                "limit_applied": result.get("limit_applied"),
                "files": result["files"]
            }
        else:
            response_data = {
                "success": False,
                "message": "문서 조회 실패",
                "error": result.get("error")
            }

        logger.info(f"SUCCESS 문서 조회 완료: {result.get('total_documents', 0)}개 문서")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"ERROR 문서 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 🔍 4. 현재 벡터 DB 상태 조회 (보너스 - 디버깅용)
@router.get("/vector-status")
async def get_vector_db_status(
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    🔍 현재 벡터 DB 상태 조회
    """
    try:
        status = await vector_service.get_vector_db_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"ERROR 벡터 DB 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
