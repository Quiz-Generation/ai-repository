from fastapi.responses import JSONResponse
from src.common.error import ErrorCode, JSendError
from src.common.vector.connect import VectorDBService



async def get_available_files(
        logger,
        vector_db: VectorDBService,
    ) -> JSONResponse:
        """문제 생성 가능한 파일 목록 조회"""
        try:
            logger.info("STEP_FILES 사용 가능한 파일 목록 조회")

            # 벡터 DB에서 파일 목록 조회
            files_result = await vector_db.get_all_documents(1000)

            if not files_result["success"]:
                raise JSendError(
                    code=ErrorCode.Quiz.FILE_NOT_FOUND[0],
                    message=ErrorCode.Quiz.FILE_NOT_FOUND[1]
                )

            # 문제 생성에 적합한 파일들만 필터링
            suitable_files = []
            for file_info in files_result["files"]:
                # 최소 청크 수 확인 (너무 작은 파일 제외)
                if file_info.get("total_chunks", 0) >= 10:
                    suitable_file = {
                        "file_id": file_info["file_id"],
                        "filename": file_info["filename"],
                        "language": file_info.get("language", "unknown"),
                        "total_chunks": file_info.get("total_chunks", 0),
                        "file_size": file_info.get("file_size", 0),
                        "upload_timestamp": file_info.get("upload_timestamp"),
                    }
                    suitable_files.append(suitable_file)

            return JSONResponse(content={
                "success": True,
                "message": f"문제 생성 가능한 파일 {len(suitable_files)}개 조회 완료",
                "total_files": len(suitable_files),
                "files": suitable_files,
            })

        except Exception as e:
            logger.error(f"ERROR 파일 목록 조회 실패: {e}")
            raise JSendError(
                code=ErrorCode.Common.DEFAULT_ERROR[0],
                message=ErrorCode.Common.DEFAULT_ERROR[1]
            )