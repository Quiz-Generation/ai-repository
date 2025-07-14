import time
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from src.app.service.vector_db_service import VectorDBService

from ..docs import document_docs
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
    file: UploadFile = File(...),
) -> JSONResponse:
    total_start_time = time.time()

    return await document_service.upload_document(
        logger=logger,
        vector_db=VectorDBService(),
        file=file
    )


    #     # 🔥 파일 ID 가져오기 (파일별 단일 ID)
    #     file_id = vector_result.get("file_id")

    #     # 전체 처리 시간 계산
    #     total_time = time.time() - total_start_time

    #     # 간단한 응답 반환
    #     response_data = {
    #         "success": vector_result["success"],
    #         "message": "PDF 업로드 완료",
    #         "file_id": file_id,
    #         "filename": file.filename,
    #         "vector_db_type": vector_service.current_db_type,  # 🎯 실제 사용된 DB
    #         "chunk_count": vector_result.get("chunk_count", 0),
    #         "auto_settings": {
    #             "chunk_size": auto_chunk_size,
    #             "chunk_overlap": auto_chunk_overlap,
    #             "pdf_loader": extraction_result["loader_used"],
    #             "language": analysis_result.language
    #         },
    #         "question_analysis": {
    #             "recommended_questions": await doc_service.calculate_optimal_question_count(
    #                 content=extraction_result["content"],
    #                 metadata=metadata
    #             ),
    #             "content_analysis": {
    #                 "total_sentences": extraction_result.get("total_sentences", 0),
    #                 "total_paragraphs": extraction_result.get("total_paragraphs", 0),
    #                 "key_concepts": extraction_result.get("key_concepts", []),
    #                 "complexity_score": extraction_result.get("complexity_score", 0)
    #             }
    #         },
    #         "performance_metrics": {
    #             "total_time": total_time,
    #             "analysis_time": analysis_time,
    #             "extraction_time": extraction_time,
    #             "vector_init_time": vector_init_time,
    #             "vector_store_time": vector_store_time,
    #             "vector_performance": vector_result.get("performance_metrics", {})
    #         }
    #     }

    #     if not vector_result["success"]:
    #         response_data["error"] = vector_result.get("error")

    #     logger.info(f"🎉 SUCCESS PDF 업로드 완료: {file.filename} -> {vector_service.current_db_type}")
    #     logger.info(f"⏱️ 전체 처리 시간: {total_time:.2f}초")
    #     logger.info(f"📊 성능 요약: 분석({analysis_time:.2f}s) + 추출({extraction_time:.2f}s) + 벡터화({vector_store_time:.2f}s)")

    #     return JSONResponse(content=response_data)

    # except Exception as e:
    #     total_time = time.time() - total_start_time
    #     logger.error(f"ERROR PDF 업로드 실패: {e} (총 소요시간: {total_time:.2f}초)")
    #     raise HTTPException(status_code=500, detail=str(e))



# # 💥 5. 벡터 DB 모든 데이터 삭제 (위험한 작업)
# @router.delete("/clear-all")
# async def clear_all_documents(
#     confirm_token: str = Form(..., description="삭제 확인 토큰: CLEAR_ALL_CONFIRM"),
#     vector_service: VectorDBService = Depends(get_vector_service)
# ) -> JSONResponse:
#     """
#     💥 벡터 DB의 모든 데이터 삭제 (위험한 작업)

#     ⚠️ 주의: 이 작업은 되돌릴 수 없습니다!
#     confirm_token에 "CLEAR_ALL_CONFIRM"을 입력해야 합니다.
#     """
#     try:
#         logger.info("🚨 DANGER 벡터 DB 전체 삭제 요청")

#         # 전체 삭제 실행
#         result = await vector_service.clear_all_documents(confirm_token)

#         if result["success"]:
#             response_data = {
#                 "success": True,
#                 "message": result["message"],
#                 "vector_db_type": result["vector_db_type"],
#                 "deleted_count": result.get("deleted_count", 0),
#                 "remaining_count": result.get("remaining_count", 0)
#             }
#             logger.info(f"SUCCESS 벡터 DB 전체 삭제 완료: {result.get('deleted_count', 0)}개 삭제")
#         else:
#             response_data = {
#                 "success": False,
#                 "message": "전체 삭제 실패",
#                 "error": result.get("error"),
#                 "vector_db_type": result.get("vector_db_type")
#             }

#         return JSONResponse(content=response_data)

#     except Exception as e:
#         logger.error(f"ERROR 벡터 DB 전체 삭제 실패: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
