"""
PDF 업로드 → 벡터 DB 저장 API 라우터

핵심 기능:
- POST /pdf/upload: PDF 파일 업로드 및 벡터 저장 → document_id 반환
- GET /pdf/documents: 업로드된 문서 목록 조회
- GET /pdf/documents/{document_id}: 특정 문서 정보 조회
- GET /pdf/search: 전체 문서에서 검색
- GET /pdf/search/{document_id}: 특정 문서에서 검색
- GET /pdf/health: 서비스 상태 확인
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Path
from fastapi.responses import JSONResponse
import logging
import tempfile
import os
import time
from datetime import datetime

# PDF 추출용
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# 벡터 서비스 import (상대 경로로 변경)
from ..services.vector_service import get_global_vector_service

# 🔥 동적 PDF 추출 시스템 import 추가
from ..services.dynamic_pdf import DynamicPDFService
from ..schemas.dynamic_pdf import Priority

# Swagger 문서 설명 import
from ..docs.pdf_service import (
    desc_upload_pdf,
    desc_get_documents,
    desc_get_document_info,
    desc_search_all_documents,
    desc_search_in_document,
    desc_health_check,
    desc_switch_database,
    desc_get_stats
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pdf", tags=["PDF Vector"])

# 전역 벡터 서비스 인스턴스 (WEAVIATE 기본 사용) - 싱글톤 사용
vector_service = get_global_vector_service()

# 🔥 동적 PDF 추출 서비스 인스턴스 생성
dynamic_pdf_service = DynamicPDFService()


@router.get("/health", description=desc_health_check)
async def health_check() -> JSONResponse:
    """벡터 DB 서비스 상태 확인"""
    try:
        stats = vector_service.get_stats()
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "PDF Vector Service (동적 추출기 지원)",
                "vector_db": stats["db_type"],
                "total_documents": stats["total_documents"],
                "total_uploaded_files": stats["total_uploaded_files"],
                "supported_dbs": stats["supported_dbs"],
                # 🔥 동적 추출기 정보 추가
                "extraction_system": {
                    "type": "smart_auto_optimization",
                    "available_extractors": ["pdfminer", "pdfplumber", "pymupdf"],
                    "default_mode": "auto",
                    "auto_optimization": True,
                    "manual_priorities": ["speed", "quality", "balanced"],
                    "smart_features": [
                        "파일 크기 기반 자동 우선순위 결정",
                        "내용 유형 기반 추출기 선택",
                        "파일명 기반 fallback 분석",
                        "다중 페이지 텍스트 추출"
                    ]
                },
                "endpoints": [
                    "POST /pdf/upload (스마트 자동 최적화)",
                    "POST /pdf/analyze (추출기 추천 분석)",
                    "GET /pdf/documents",
                    "GET /pdf/documents/{document_id}",
                    "GET /pdf/search",
                    "GET /pdf/search/{document_id}",
                    "GET /pdf/health",
                    "POST /pdf/switch-db"
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


@router.post("/upload", description=desc_upload_pdf)
async def upload_pdf(
    file: UploadFile = File(...),
    priority: str = Query("auto", description="추출 우선순위: auto(자동최적화), speed, quality, balanced")
) -> JSONResponse:
    """📤 PDF 파일 업로드 및 벡터 저장 → document_id 반환 (스마트 자동 최적화)"""

    upload_start_time = time.time()

    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원됩니다")

    filename = file.filename
    logger.info(f"PDF 업로드 시작: {filename} (우선순위: {priority})")

    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"파일 크기: {file_size_mb:.1f}MB")

        # 🧠 스마트 우선순위 결정
        if priority == "auto":
            logger.info("자동 최적화 모드: 파일 분석 중...")
            recommendations = dynamic_pdf_service.get_extractor_recommendations(temp_path)

            # 파일 크기와 내용 기반으로 최적 우선순위 자동 결정
            content_type = recommendations["file_info"]["content_type"]
            size_mb = recommendations["file_info"]["size_mb"]

            if size_mb > 10:
                optimal_priority = "speed"
                reason = f"대용량 파일({size_mb:.1f}MB) → 속도 우선"
            elif content_type in ["korean", "mixed"]:
                optimal_priority = "quality"
                reason = f"한글 문서 → 품질 우선"
            elif size_mb > 5:
                optimal_priority = "speed"
                reason = f"중대용량 파일({size_mb:.1f}MB) → 속도 우선"
            else:
                optimal_priority = "balanced"
                reason = f"소용량 파일({size_mb:.1f}MB) → 균형 모드"

            logger.info(f"자동 결정: {optimal_priority} ({reason})")
            extraction_priority = Priority(optimal_priority)
            auto_selected = True
        else:
            # 사용자 지정 우선순위 검증
            try:
                extraction_priority = Priority(priority.lower())
                auto_selected = False
                reason = f"사용자 지정: {priority}"
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"잘못된 우선순위: {priority}. 사용 가능: auto, speed, quality, balanced"
                )

        try:
            # 동적 PDF 텍스트 추출 (최적화된 우선순위 사용)
            extraction_result = dynamic_pdf_service.extract_text(temp_path, extraction_priority)

            if not extraction_result.success:
                logger.error(f"PDF 추출 실패: {extraction_result.error}")
                raise HTTPException(status_code=400, detail=f"PDF 추출 실패: {extraction_result.error}")

            pdf_text = extraction_result.text

            if len(pdf_text.strip()) < 100:
                logger.warning(f"추출된 텍스트가 부족함: {len(pdf_text)}자")
                raise HTTPException(status_code=400, detail="PDF에서 충분한 텍스트를 추출할 수 없습니다")

            # 벡터 저장
            logger.info(f"벡터 저장 시작...")
            vector_start = time.time()
            result = vector_service.process_pdf_text(pdf_text, filename)
            vector_time = time.time() - vector_start

            # 벡터 저장 결과 검증 (새로운 형식)
            if not result.get("document_id"):
                logger.error(f"벡터 저장 실패: document_id가 없음")
                raise HTTPException(status_code=500, detail="벡터 저장 실패: 문서 ID 생성 오류")

            total_time = time.time() - upload_start_time
            logger.info(f"업로드 완료: 총 {total_time:.2f}초 (추출: {extraction_result.extraction_time:.2f}초, 벡터화: {vector_time:.2f}초)")

            return JSONResponse(
                status_code=200,
                content={
                    "message": "PDF 업로드 및 벡터 저장 성공 (스마트 자동 최적화)",
                    "document_id": result["document_id"],  # 🔑 RAG용 문서 ID
                    "filename": filename,
                    "file_size": len(content),
                    "text_length": len(pdf_text),
                    "total_chunks": result["total_chunks"],
                    "stored_chunks": result["stored_chunks"],
                    "db_type": vector_service.db_type,
                    "upload_timestamp": datetime.now().isoformat(),
                    # 🧠 스마트 최적화 정보
                    "optimization_info": {
                        "priority_mode": "auto" if auto_selected else "manual",
                        "selected_priority": extraction_priority.value,
                        "selection_reason": reason,
                        "extractor_used": extraction_result.extractor_used,
                        "content_type": extraction_result.content_type,
                        "auto_optimized": auto_selected
                    },
                    # 동적 추출 정보
                    "extraction_info": {
                        "extractor_used": extraction_result.extractor_used,
                        "content_type": extraction_result.content_type,
                        "priority": extraction_result.priority,
                        "extraction_time": extraction_result.extraction_time,
                        "speed_mbps": extraction_result.speed_mbps,
                        "auto_selected": extraction_result.metadata.get("auto_selected", True),
                        "selection_reason": extraction_result.metadata.get("selection_reason", "")
                    },
                    # 성능 메트릭
                    "performance": {
                        "total_time": round(total_time, 3),
                        "extraction_time": extraction_result.extraction_time,
                        "vectorization_time": round(vector_time, 3),
                        "extraction_speed_mbps": extraction_result.speed_mbps
                    },
                    "note": "🧠 스마트 자동 최적화로 처리되었습니다. document_id를 저장하여 RAG 퀴즈 생성에 사용하세요."
                }
            )

        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        error_time = time.time() - upload_start_time
        logger.error(f"PDF 업로드 처리 중 오류: {str(e)} ({error_time:.2f}초)")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


@router.get("/documents", description=desc_get_documents)
async def get_document_list() -> JSONResponse:
    """📋 업로드된 문서 목록 조회 (RAG용)"""
    try:
        documents = vector_service.get_document_list()

        # RAG용 정보 추가
        for doc in documents:
            doc["available_for_rag"] = True
            doc["recommended_for_quiz"] = doc["chunk_count"] >= 5  # 5개 이상 청크면 퀴즈 생성 권장

        return JSONResponse(
            status_code=200,
            content={
                "message": "문서 목록 조회 성공",
                "total_documents": len(documents),
                "db_type": vector_service.db_type,
                "documents": documents,
                "note": "document_id를 사용하여 특정 문서로 RAG 퀴즈를 생성할 수 있습니다"
            }
        )
    except Exception as e:
        logger.error(f"문서 목록 조회 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"문서 목록 조회 오류: {str(e)}")


@router.get("/documents/{document_id}", description=desc_get_document_info)
async def get_document_info(
    document_id: str = Path(..., description="문서 ID")
) -> JSONResponse:
    """📄 특정 문서 정보 조회 (RAG용 상세 정보)"""
    try:
        document_info = vector_service.get_document_info(document_id)

        if not document_info:
            raise HTTPException(status_code=404, detail=f"문서를 찾을 수 없습니다: {document_id}")

        # RAG용 추가 정보
        document_info["rag_ready"] = True
        document_info["chunk_size_avg"] = document_info["total_chars"] // document_info["chunk_count"]
        document_info["quiz_generation_score"] = min(10, document_info["chunk_count"] * 2)  # 점수 계산

        return JSONResponse(
            status_code=200,
            content={
                "message": "문서 정보 조회 성공",
                "document": document_info,
                "db_type": vector_service.db_type,
                "rag_info": {
                    "can_generate_quiz": document_info["chunk_count"] >= 3,
                    "recommended_questions": min(10, document_info["chunk_count"] // 2),
                    "content_quality": "high" if document_info["chunk_count"] >= 10 else "medium"
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 정보 조회 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"문서 정보 조회 오류: {str(e)}")


@router.get("/search", description=desc_search_all_documents)
async def search_all_documents(
    query: str = Query(..., description="검색 쿼리"),
    top_k: int = Query(5, ge=1, le=20, description="결과 개수")
) -> JSONResponse:
    """🔍 전체 문서에서 검색"""
    if not query.strip():
        raise HTTPException(status_code=400, detail="검색 쿼리가 비어있습니다")

    try:
        results = vector_service.search_documents(query, top_k)

        # 결과 정리
        formatted_results = []
        for result in results:
            formatted_results.append({
                "doc_id": result["doc_id"],
                "document_id": result["metadata"].get("document_id", ""),
                "source_filename": result["metadata"].get("source", ""),
                "text_preview": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "similarity": round(result["similarity"], 4),
                "chunk_index": result["metadata"].get("chunk_index", 0)
            })

        return JSONResponse(
            status_code=200,
            content={
                "message": "전체 검색 완료",
                "query": query,
                "total_results": len(formatted_results),
                "db_type": vector_service.db_type,
                "results": formatted_results
            }
        )

    except Exception as e:
        logger.error(f"검색 처리 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"검색 오류: {str(e)}")


@router.get("/search/{document_id}", description=desc_search_in_document)
async def search_in_document(
    document_id: str = Path(..., description="문서 ID"),
    query: str = Query(..., description="검색 쿼리"),
    top_k: int = Query(5, ge=1, le=10, description="결과 개수")
) -> JSONResponse:
    """🎯 특정 문서 내에서만 검색 (RAG용 컨텍스트 추출)"""
    if not query.strip():
        raise HTTPException(status_code=400, detail="검색 쿼리가 비어있습니다")

    try:
        # 문서 존재 확인
        document_info = vector_service.get_document_info(document_id)
        if not document_info:
            raise HTTPException(status_code=404, detail=f"문서를 찾을 수 없습니다: {document_id}")

        # 특정 문서에서 검색
        results = vector_service.search_in_document(query, document_id, top_k)

        # RAG용 결과 정리
        formatted_results = []
        full_context = ""

        for result in results:
            formatted_result = {
                "doc_id": result["doc_id"],
                "text_preview": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "full_text": result["text"],  # RAG 컨텍스트용
                "similarity": round(result["similarity"], 4),
                "chunk_index": result["metadata"].get("chunk_index", 0)
            }
            formatted_results.append(formatted_result)
            full_context += result["text"] + "\n\n"

        return JSONResponse(
            status_code=200,
            content={
                "message": f"문서 내 검색 완료",
                "document_id": document_id,
                "document_filename": document_info["source_filename"],
                "query": query,
                "total_results": len(formatted_results),
                "db_type": vector_service.db_type,
                "results": formatted_results,
                "rag_context": {
                    "combined_text": full_context.strip(),
                    "context_length": len(full_context),
                    "ready_for_rag": len(full_context) > 100
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 내 검색 처리 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"문서 내 검색 오류: {str(e)}")


@router.post("/switch-db", description=desc_switch_database)
async def switch_database(db_type: str) -> JSONResponse:
    """벡터 데이터베이스 변경"""
    try:
        success = vector_service.switch_database(db_type)

        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 DB 타입: {db_type}. 지원 타입: {vector_service.get_stats()['supported_dbs']}"
            )

        return JSONResponse(
            status_code=200,
            content={
                "message": f"데이터베이스가 {db_type}으로 변경되었습니다",
                "previous_db": vector_service.db_type,
                "current_db": db_type,
                "total_documents": 0  # 새 DB이므로 0
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DB 변경 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DB 변경 오류: {str(e)}")


@router.get("/stats", description=desc_get_stats)
async def get_stats() -> JSONResponse:
    """벡터 DB 통계"""
    try:
        stats = vector_service.get_stats()
        return JSONResponse(
            status_code=200,
            content=stats
        )
    except Exception as e:
        logger.error(f"통계 조회 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"통계 조회 오류: {str(e)}")


@router.post("/analyze", description="PDF 파일 분석 및 추출기 추천")
async def analyze_pdf(file: UploadFile = File(...)) -> JSONResponse:
    """🔍 PDF 파일 분석 및 최적 추출기 추천 (업로드 전 미리보기)"""

    analyze_start = time.time()

    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원됩니다")

    filename = file.filename
    logger.info(f"PDF 분석 요청: {filename}")

    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"파일 크기: {file_size_mb:.1f}MB")

        try:
            # 파일 분석 및 추천
            recommendations = dynamic_pdf_service.get_extractor_recommendations(temp_path)

            analyze_time = time.time() - analyze_start
            logger.info(f"분석 완료: {analyze_time:.2f}초")

            return JSONResponse(
                status_code=200,
                content={
                    "message": "PDF 파일 분석 완료",
                    "filename": filename,
                    "file_size": len(content),
                    "analysis_time": round(analyze_time, 3),
                    "analysis": recommendations,
                    "usage_tip": "이 정보를 참고하여 /pdf/upload API에서 priority 파라미터를 설정하세요"
                }
            )

        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        error_time = time.time() - analyze_start
        logger.error(f"PDF 분석 중 오류: {str(e)} ({error_time:.2f}초)")
        raise HTTPException(status_code=500, detail=f"PDF 분석 오류: {str(e)}")