"""
📄 Document API Routes - Simplified
"""
import logging
import time
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Query
from fastapi.responses import JSONResponse

from ..service.document_service import DocumentService
from ..service.vector_db_service import VectorDBService
from ..helper.pdf_loader_helper import PDFLoaderHelper

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

# 서비스 의존성 주입
async def get_document_service() -> DocumentService:
    """문서 서비스 의존성 주입"""
    return DocumentService()

async def get_vector_service() -> VectorDBService:
    """벡터 DB 서비스 의존성 주입 (전역 서비스 사용)"""
    from ..main import global_vector_service
    if global_vector_service is None:
        raise HTTPException(status_code=500, detail="전역 벡터 DB 서비스가 초기화되지 않았습니다")
    return global_vector_service


# 🚀 1. PDF 업로드 및 벡터 저장 (+ 문서 ID 반환)
@router.post("/upload")
async def upload_pdf_to_vector_db(
    file: UploadFile = File(...),
    doc_service: DocumentService = Depends(get_document_service),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    📄 PDF 파일 업로드 및 벡터 DB 저장 (간단 버전)
    - 파일명만 입력, 나머지는 자동 처리
    - 벡터 DB: Milvus 우선 (전역 설정)
    - 청크 크기: 자동 최적화
    """
    total_start_time = time.time()

    try:
        logger.info("=" * 50)
        logger.info("STEP1 PDF 업로드 및 벡터 저장 시작")

        # 파일 검증
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다")

        # PDF 특성 분석 및 최적 로더 선택
        analysis_start_time = time.time()
        logger.info("STEP2 PDF 특성 분석 시작")
        analysis_result = await PDFLoaderHelper.analyze_pdf_characteristics(file)
        analysis_time = time.time() - analysis_start_time
        logger.info(f"⏱️ PDF 분석 완료: {analysis_time:.2f}초")

        # PDF 내용 추출
        extraction_start_time = time.time()
        logger.info("STEP3 PDF 내용 추출 시작")
        extraction_result = await doc_service.process_pdf_with_dynamic_selection(
            file, analysis_result.recommended_loader
        )
        extraction_time = time.time() - extraction_start_time
        logger.info(f"⏱️ PDF 추출 완료: {extraction_time:.2f}초")

        if not extraction_result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"PDF 처리 실패: {extraction_result.get('error', 'Unknown error')}"
            )

        # 🔥 벡터 DB 강제 Milvus 초기화 (기존 서비스 무시)
        vector_init_start_time = time.time()
        logger.info("STEP4 Milvus 벡터 DB 강제 초기화")
        await vector_service.force_switch_to_milvus()
        vector_init_time = time.time() - vector_init_start_time
        logger.info(f"⏱️ 벡터 DB 초기화: {vector_init_time:.2f}초")

        # 🎯 자동 청크 설정 (한국어 최적화)
        auto_chunk_size = 800  # 한국어에 최적화된 크기
        auto_chunk_overlap = 100  # 적당한 오버랩

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
        vector_store_start_time = time.time()
        logger.info("STEP5 Milvus 벡터 DB 저장 시작")
        vector_result = await vector_service.store_pdf_content(
            pdf_content=extraction_result["content"],
            metadata=metadata,
            chunk_size=auto_chunk_size,
            chunk_overlap=auto_chunk_overlap
        )
        vector_store_time = time.time() - vector_store_start_time
        logger.info(f"⏱️ 벡터 DB 저장 완료: {vector_store_time:.2f}초")

        # 🔥 파일 ID 가져오기 (파일별 단일 ID)
        file_id = vector_result.get("file_id")

        # 전체 처리 시간 계산
        total_time = time.time() - total_start_time

        # 간단한 응답 반환
        response_data = {
            "success": vector_result["success"],
            "message": "PDF 업로드 완료",
            "file_id": file_id,
            "filename": file.filename,
            "vector_db_type": vector_service.current_db_type,  # 🎯 실제 사용된 DB
            "chunk_count": vector_result.get("chunk_count", 0),
            "auto_settings": {
                "chunk_size": auto_chunk_size,
                "chunk_overlap": auto_chunk_overlap,
                "pdf_loader": extraction_result["loader_used"],
                "language": analysis_result.language
            },
            "question_analysis": {
                "recommended_questions": await doc_service.calculate_optimal_question_count(
                    content=extraction_result["content"],
                    metadata=metadata
                ),
                "content_analysis": {
                    "total_sentences": extraction_result.get("total_sentences", 0),
                    "total_paragraphs": extraction_result.get("total_paragraphs", 0),
                    "key_concepts": extraction_result.get("key_concepts", []),
                    "complexity_score": extraction_result.get("complexity_score", 0)
                }
            },
            "performance_metrics": {
                "total_time": total_time,
                "analysis_time": analysis_time,
                "extraction_time": extraction_time,
                "vector_init_time": vector_init_time,
                "vector_store_time": vector_store_time,
                "vector_performance": vector_result.get("performance_metrics", {})
            }
        }

        if not vector_result["success"]:
            response_data["error"] = vector_result.get("error")

        logger.info(f"🎉 SUCCESS PDF 업로드 완료: {file.filename} -> {vector_service.current_db_type}")
        logger.info(f"⏱️ 전체 처리 시간: {total_time:.2f}초")
        logger.info(f"📊 성능 요약: 분석({analysis_time:.2f}s) + 추출({extraction_time:.2f}s) + 벡터화({vector_store_time:.2f}s)")

        return JSONResponse(content=response_data)

    except Exception as e:
        total_time = time.time() - total_start_time
        logger.error(f"ERROR PDF 업로드 실패: {e} (총 소요시간: {total_time:.2f}초)")
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
    limit: int = Query(100, description="조회할 파일 수 제한 (기본: 100개 파일)"),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    📋 현재 벡터 DB에 저장된 파일 조회 (최신순)
    - limit: 조회할 파일 개수 제한 (기본: 100개 파일)
    - 파일별로 그룹화하여 표시 (최신 업로드순)
    """
    try:
        logger.info(f"STEP_DOCS 파일 조회 시작 (limit: {limit}개 파일)")

        # limit 범위 제한 (1~1000 파일)
        actual_limit = max(1, min(limit, 1000))

        # 파일 조회
        result = await vector_service.get_all_documents(actual_limit)

        if result["success"]:
            response_data = {
                "success": True,
                "message": "파일 조회 완료",
                "vector_db_type": result["vector_db_type"],
                "total_documents": result["total_documents"],  # 전체 청크 수
                "total_files": result["total_files"],  # 실제 반환된 파일 수
                "all_files_count": result.get("all_files_count", 0),  # 전체 파일 수
                "limit_applied": result.get("limit_applied"),
                "files": result["files"]
            }
        else:
            response_data = {
                "success": False,
                "message": "파일 조회 실패",
                "error": result.get("error")
            }

        logger.info(f"SUCCESS 파일 조회 완료: {result.get('total_files', 0)}개 파일 반환 (전체 {result.get('all_files_count', 0)}개 중)")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"ERROR 파일 조회 실패: {e}")
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


# 💥 5. 벡터 DB 모든 데이터 삭제 (위험한 작업)
@router.delete("/clear-all")
async def clear_all_documents(
    confirm_token: str = Form(..., description="삭제 확인 토큰: CLEAR_ALL_CONFIRM"),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> JSONResponse:
    """
    💥 벡터 DB의 모든 데이터 삭제 (위험한 작업)

    ⚠️ 주의: 이 작업은 되돌릴 수 없습니다!
    confirm_token에 "CLEAR_ALL_CONFIRM"을 입력해야 합니다.
    """
    try:
        logger.info("🚨 DANGER 벡터 DB 전체 삭제 요청")

        # 전체 삭제 실행
        result = await vector_service.clear_all_documents(confirm_token)

        if result["success"]:
            response_data = {
                "success": True,
                "message": result["message"],
                "vector_db_type": result["vector_db_type"],
                "deleted_count": result.get("deleted_count", 0),
                "remaining_count": result.get("remaining_count", 0)
            }
            logger.info(f"SUCCESS 벡터 DB 전체 삭제 완료: {result.get('deleted_count', 0)}개 삭제")
        else:
            response_data = {
                "success": False,
                "message": "전체 삭제 실패",
                "error": result.get("error"),
                "vector_db_type": result.get("vector_db_type")
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"ERROR 벡터 DB 전체 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
