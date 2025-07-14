from fastapi.responses import JSONResponse
from src.common.error import ErrorCode, JSendError
from src.common.vector.connect import VectorDBService
from src.app.func import quiz as quiz_func


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


async def generate_quiz_from_file(
    logger,
    vector_db: VectorDBService,
    file_id: str,
    num_questions: int,
    difficulty: str,
    question_type: str,
    category: str,
    sub_category: str
) -> JSONResponse:
    try :
        logger.info("🚀 AI 문제 생성 API 시작")
        logger.info(f"STEP_REQUEST 문제 생성 요청: {file_id}, {num_questions}개 문제, {difficulty} 난이도")

        # 기본 검증
        if not file_id:
            raise JSendError(
                code=ErrorCode.Quiz.FILE_NOT_FOUND[0],
                message=ErrorCode.Quiz.FILE_NOT_FOUND[1]
            )

        if not (1 <= num_questions <= 50):
            raise JSendError(
                code=ErrorCode.Quiz.INVALID_QUESTION_COUNT[0],
                message=ErrorCode.Quiz.INVALID_QUESTION_COUNT[1]
            )

        if difficulty not in ["easy", "medium", "hard"]:
            raise JSendError(
                code=ErrorCode.Quiz.INVALID_DIFFICULTY[0],
                message=ErrorCode.Quiz.INVALID_DIFFICULTY[1]
            )

        valid_types = ["multiple_choice", "true_false", "short_answer", "essay", "fill_blank"]
        if question_type not in valid_types:
            raise JSendError(
                code=ErrorCode.Quiz.INVALID_QUESTION_TYPE[0],
                message=ErrorCode.Quiz.INVALID_QUESTION_TYPE[1]
            )

        # 문제 생성 실행
        result = await quiz_func.generate_quiz_from_file(
            logger=logger,
            vector_db=vector_db,
            file_id=file_id,
            num_questions=num_questions,
            difficulty=difficulty,
            question_type=question_type,
            category=category,
            sub_category=sub_category
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"ERROR 문제 생성 실패: {e}")
        raise JSendError(
            code=ErrorCode.Common.DEFAULT_ERROR[0],
            message=ErrorCode.Common.DEFAULT_ERROR[1]
        )
