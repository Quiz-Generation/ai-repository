"""
PDF 업로드 → 벡터 DB 저장 API 라우터

엔드포인트:
- POST /pdf/upload: PDF 파일 업로드 및 벡터 저장
- GET /pdf/search: 벡터 검색
- GET /pdf/health: 서비스 상태 확인
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import tempfile
import os

# PDF 추출용
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# 벡터 서비스 import
from lagnchain_fastapi_app.app.services.vector_service import PDFVectorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pdf", tags=["PDF Vector"])

# 전역 벡터 서비스 인스턴스 (WEAVIATE 기본 사용)
vector_service = PDFVectorService(db_type="weaviate")


class SimplePDFReader:
    """간단한 PDF 텍스트 추출기"""

    def extract_text(self, pdf_path: str) -> str:
        if not HAS_PYMUPDF:
            raise Exception("PyMuPDF not available")

        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"PDF 추출 실패: {str(e)}")


@router.get("/health")
async def health_check() -> JSONResponse:
    """벡터 DB 서비스 상태 확인"""
    try:
        stats = vector_service.get_stats()
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "PDF Vector Service",
                "vector_db": stats["db_type"],
                "total_documents": stats["total_documents"],
                "supported_dbs": stats["supported_dbs"],
                "endpoints": [
                    "POST /pdf/upload",
                    "GET /pdf/search",
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


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> JSONResponse:
    """PDF 파일 업로드 및 벡터 저장"""
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원됩니다")

    filename = file.filename  # None이 아님을 보장

    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # PDF 텍스트 추출
            pdf_reader = SimplePDFReader()
            pdf_text = pdf_reader.extract_text(temp_path)

            if len(pdf_text.strip()) < 100:
                raise HTTPException(status_code=400, detail="PDF에서 충분한 텍스트를 추출할 수 없습니다")

            # 벡터 저장
            result = vector_service.process_pdf_text(pdf_text, filename)

            if not result["success"]:
                raise HTTPException(status_code=500, detail=result.get("error", "벡터 저장 실패"))

            return JSONResponse(
                status_code=200,
                content={
                    "message": "PDF 업로드 및 벡터 저장 성공",
                    "filename": filename,
                    "file_size": len(content),
                    "text_length": len(pdf_text),
                    "total_chunks": result["total_chunks"],
                    "stored_chunks": result["stored_chunks"],
                    "db_type": result["db_type"]
                }
            )

        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF 업로드 처리 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


@router.get("/search")
async def search_documents(
    query: str = Query(..., description="검색 쿼리"),
    top_k: int = Query(5, ge=1, le=20, description="결과 개수")
) -> JSONResponse:
    """벡터 검색"""
    if not query.strip():
        raise HTTPException(status_code=400, detail="검색 쿼리가 비어있습니다")

    try:
        results = vector_service.search_documents(query, top_k)

        # 결과 정리
        formatted_results = []
        for result in results:
            formatted_results.append({
                "doc_id": result["doc_id"],
                "text_preview": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "similarity": round(result["similarity"], 4),
                "metadata": {
                    "source": result["metadata"].get("source", ""),
                    "chunk_index": result["metadata"].get("chunk_index", 0)
                }
            })

        return JSONResponse(
            status_code=200,
            content={
                "query": query,
                "total_results": len(formatted_results),
                "db_type": vector_service.db_type,
                "results": formatted_results
            }
        )

    except Exception as e:
        logger.error(f"검색 처리 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"검색 오류: {str(e)}")


@router.post("/switch-db")
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


@router.get("/stats")
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