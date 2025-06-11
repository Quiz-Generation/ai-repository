"""
📄 Document API Routes with Vector DB Integration
"""
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Query
from fastapi.responses import JSONResponse

from ..schemas.document_schema import DocumentUploadResponse
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


# 🔄 기존 API (하위 호환성 유지)
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


# 🚀 새로운 벡터 DB 통합 API
@router.post("/upload-and-store")
async def upload_and_store_to_vector_db(
    file: UploadFile = File(...),
    vector_db_type: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    doc_service: DocumentService = Depends(get_document_service),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    📄🗄️ PDF 파일 업로드 및 벡터 DB 저장
    - 동적 PDF 로더 선택
    - 텍스트 추출 및 청킹
    - 임베딩 생성 및 벡터 DB 저장
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
            "has_tables": analysis_result.has_tables,
            "complexity": analysis_result.complexity,
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

        # 종합 결과 반환
        response_data = {
            "success": True,
            "message": "PDF 업로드 및 벡터 저장 완료",
            "pdf_analysis": {
                "filename": file.filename,
                "language": analysis_result.language,
                "recommended_loader": analysis_result.recommended_loader,
                "complexity": analysis_result.complexity,
                "has_tables": analysis_result.has_tables,
                "file_size": analysis_result.file_size
            },
            "pdf_extraction": {
                "loader_used": extraction_result["loader_used"],
                "content_length": len(extraction_result["content"]),
                "processing_time": extraction_result["processing_time"],
                "fallback_attempts": extraction_result.get("fallback_attempts", 0)
            },
            "vector_storage": {
                "vector_db_type": selected_db,
                "success": vector_result["success"],
                "chunk_count": vector_result.get("chunk_count", 0),
                "stored_document_count": vector_result.get("stored_document_count", 0),
                "embedding_dimension": vector_result.get("embedding_dimension", 0),
                "model_name": vector_result.get("model_name", "unknown")
            }
        }

        if not vector_result["success"]:
            response_data["vector_storage"]["error"] = vector_result.get("error")
            response_data["message"] = "PDF 처리 완료, 벡터 저장 실패"

        logger.info("SUCCESS 전체 프로세스 완료")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"ERROR PDF 업로드 및 벡터 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_similar_content(
    query: str = Form(...),
    top_k: int = Form(5),
    filename_filter: Optional[str] = Form(None),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    🔍 벡터 DB에서 유사한 내용 검색
    """
    try:
        logger.info(f"STEP_SEARCH 벡터 검색 시작: '{query[:50]}...'")

        # 필터 조건 구성
        filters = {}
        if filename_filter:
            filters["filename"] = filename_filter

        # 벡터 검색 수행
        search_results = await vector_service.search_similar_content(
            query=query,
            top_k=top_k,
            filters=filters if filters else None
        )

        # 결과 포맷팅
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "content": result.document.content,
                "score": result.score,
                "distance": result.distance,
                "metadata": result.document.metadata
            })

        response_data = {
            "success": True,
            "query": query,
            "top_k": top_k,
            "filters": filters,
            "result_count": len(formatted_results),
            "results": formatted_results,
            "vector_db_type": vector_service.current_db_type
        }

        logger.info(f"SUCCESS 벡터 검색 완료: {len(formatted_results)}개 결과")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"ERROR 벡터 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector-status")
async def get_vector_db_status(
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    🔧 벡터 DB 상태 정보 조회
    """
    try:
        logger.info("STEP_STATUS 벡터 DB 상태 조회")

        status = await vector_service.get_vector_db_status()

        logger.info("SUCCESS 벡터 DB 상태 조회 완료")
        return JSONResponse(content=status)

    except Exception as e:
        logger.error(f"ERROR 벡터 DB 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vector-initialize")
async def initialize_vector_db(
    db_type: Optional[str] = Form(None, description="초기화할 DB 타입 (선택사항)"),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    🔧 벡터 DB 강제 초기화
    - db_type 미지정 시 우선순위에 따라 자동 선택
    """
    try:
        logger.info(f"STEP_INIT 벡터 DB 초기화 시작: {db_type or '자동 선택'}")

        # 벡터 DB 초기화
        if db_type:
            selected_db = await vector_service.initialize_vector_db(db_type)
        else:
            selected_db = await vector_service.initialize_vector_db()

        # 상태 정보 조회
        status = await vector_service.get_vector_db_status()

        response_data = {
            "success": True,
            "message": f"{selected_db.upper()} 벡터 DB 초기화 완료",
            "selected_db": selected_db,
            "status": status
        }

        logger.info(f"SUCCESS {selected_db.upper()} 벡터 DB 초기화 완료")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"ERROR 벡터 DB 초기화 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "벡터 DB 초기화 실패"
            }
        )


@router.post("/vector-switch")
async def switch_vector_db(
    db_type: str = Form(...),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    🔄 벡터 DB 타입 변경
    """
    try:
        logger.info(f"STEP_SWITCH {db_type.upper()}로 벡터 DB 전환 시도")

        success = await vector_service.switch_vector_db(db_type)

        if success:
            response_data = {
                "success": True,
                "message": f"{db_type.upper()}로 전환 완료",
                "current_db_type": db_type
            }
            logger.info(f"SUCCESS {db_type.upper()}로 전환 완료")
        else:
            response_data = {
                "success": False,
                "message": f"{db_type.upper()}로 전환 실패",
                "current_db_type": vector_service.current_db_type
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"ERROR 벡터 DB 전환 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vector-documents/{filename}")
async def delete_vector_documents_by_filename(
    filename: str,
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    🗑️ 파일명으로 벡터 DB의 문서들 삭제
    """
    try:
        logger.info(f"STEP_DELETE {filename} 관련 문서 삭제 시작")

        result = await vector_service.delete_documents_by_filename(filename)

        if result["success"]:
            logger.info(f"SUCCESS {filename} 관련 {result['deleted_count']}개 문서 삭제 완료")
        else:
            logger.error(f"ERROR {filename} 문서 삭제 실패: {result.get('error')}")

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"ERROR 문서 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_document_system_info(
    doc_service: DocumentService = Depends(get_document_service),
    vector_service: VectorDBService = Depends(get_vector_service)
):
    """
    📊 문서 처리 시스템 전체 정보 조회
    """
    try:
        # PDF 로더 정보
        loader_info = await doc_service.get_loader_selection_info()

        # 벡터 DB 상태
        vector_status = await vector_service.get_vector_db_status()

        # 로더 선택 규칙
        selection_rules = PDFLoaderHelper.get_loader_selection_rules()

        system_info = {
            "system_name": "PDF Processing with Vector DB",
            "version": "2.0.0",
            "pdf_processing": {
                "supported_loaders": loader_info.get("supported_loaders", []),
                "loader_capabilities": loader_info.get("capabilities", {}),
                "selection_rules": selection_rules
            },
            "vector_database": {
                "current_db": vector_status.get("current_db_type"),
                "supported_dbs": vector_status.get("supported_db_types", []),
                "priority_order": vector_status.get("priority_order", {}),
                "embedding_model": vector_status.get("embedding_model"),
                "all_db_health": vector_status.get("all_db_health", {})
            },
            "features": [
                "🔍 동적 PDF 로더 선택 (4개 라이브러리)",
                "🗄️ 벡터 데이터베이스 통합 (3개 DB)",
                "🧠 임베딩 생성 및 유사도 검색",
                "🌐 다국어 지원 (한국어 특화)",
                "📊 복잡도 기반 자동 선택",
                "🔄 폴백 메커니즘"
            ]
        }

        return JSONResponse(content=system_info)

    except Exception as e:
        logger.error(f"ERROR 시스템 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all-documents")
async def get_all_documents(
    limit: Optional[int] = Query(None, description="조회할 문서 수 제한"),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    📋 벡터 DB의 모든 문서 목록 조회
    - 파일별로 그룹핑하여 요약 정보 제공
    - 각 파일의 청크 수, 메타데이터, 미리보기 포함
    """
    try:
        logger.info(f"STEP_LIST 모든 문서 조회 시작 (제한: {limit or '없음'})")

        # 벡터 DB에서 모든 문서 조회
        result = await vector_service.get_all_documents(limit)

        if result["success"]:
            logger.info(f"SUCCESS 모든 문서 조회 완료: {result['total_files']}개 파일, {result['total_documents']}개 청크")
        else:
            logger.error(f"ERROR 모든 문서 조회 실패: {result.get('error')}")

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"ERROR 모든 문서 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
