import time
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from src.common.error import ErrorCode, JSendError
from src.app.func import document_loader as document_loader_func
from src.app.func import document as document_func
from src.app.func import vector as vector_func

async def upload_document(
        logger,
        vector_db,
        file: UploadFile
    ) -> JSONResponse:
    """
    PDF 파일 업로드 및 벡터 저장

    PDF 파일 업로드 시 파일 특성 분석 및 최적 로더 선택

    1. 파일 검증
    2. 파일 특성 분석 및 최적 로더 선택
    3. PDF 내용 추출
    4. 벡터 데이터 베이스 저장
    5. 파일 벡터 삭제


    """
    logger.info(
        f"""
            STEP1 PDF 업로드 시작
            "파일명": {file.filename}
        """
    )

    #1. 파일 검증
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        logger.error(
            f"""
                [PDF 업로드 실패]
                PDF 파일만 업로드 가능합니다: {file.filename}
            """
        )
        raise JSendError(
            code=ErrorCode.Document.PDF_UPLOAD_ERROR[0],
            message=ErrorCode.Document.PDF_UPLOAD_ERROR[1]
        )

    #2. 해당 파일 특성 분석 및 최적 로더 선택
    analysis_start_time = time.time()
    logger.info(
        f"""
            STEP2 PDF 특성 분석 시작
        """
    )
    analysis_result = await document_loader_func.analyze_pdf_characteristics(
        logger=logger,
        file=file
    )
    analysis_time = time.time() - analysis_start_time
    logger.info(f"⏱️ PDF 분석 완료: {analysis_time:.2f}초")

    if not analysis_result.recommended_loader:
        logger.error(
            f"""
                STEP2 PDF 특성 분석 실패
                "파일명": {file.filename}
                "최적 로더": {analysis_result.recommended_loader}
            """
        )
        raise JSendError(
            code=ErrorCode.Document.PDF_ANALYSIS_ERROR[0],
            message=ErrorCode.Document.PDF_ANALYSIS_ERROR[1] + f" {analysis_result.recommended_loader}"
        )

    #3. PDF 내용 추출
    extraction_start_time = time.time()
    logger.info(
        f"""
            STEP3 PDF 내용 추출 시작
            "파일명": {file.filename}
            "최적 로더": {analysis_result.recommended_loader}
        """
    )
    extraction_result = await document_func.process_pdf(
        logger=logger,
        file=file,
        loader=analysis_result.recommended_loader
    )
    extraction_time = time.time() - extraction_start_time
    logger.info(f"⏱️ PDF 추출 완료: {extraction_time:.2f}초")

    if not extraction_result["success"]:
        raise JSendError(
            code=ErrorCode.Document.PDF_EXTRACT_ERROR[0],
            message=ErrorCode.Document.PDF_EXTRACT_ERROR[1] + f" {extraction_result.get('error', 'Unknown error')}"
        )

    #4. 벡터 데이터 베이스 저장

    # 🔥 벡터 DB 강제 Milvus 초기화 (기존 서비스 무시)
    vector_init_start_time = time.time()
    vector_init_time = time.time() - vector_init_start_time
    logger.info(f"⏱️ 벡터 DB 초기화: {vector_init_time:.2f}초")

    # 🎯 자동 청크 설정 (한국어 최적화)
    auto_chunk_size = 2000  # 한국어에 최적화된 크기
    auto_chunk_overlap = 200  # 적당한 오버랩

    # 메타데이터 구성
    metadata = {
        "filename": file.filename,
        "file_size": file.size,
        "pdf_loader": analysis_result.recommended_loader,
        "language": analysis_result.language,
        "upload_timestamp": extraction_result["processing_time"],
        "source": "document_upload"
    }

    # 벡터 DB에 저장
    vector_store_start_time = time.time()
    logger.info("STEP5 Milvus 벡터 DB 저장 시작")
    vector_result = await vector_func.store_pdf_content(
        logger=logger,
        vector_db=vector_db,
        pdf_content=extraction_result["content"],
        metadata=metadata,
        chunk_size=auto_chunk_size,
        chunk_overlap=auto_chunk_overlap
    )
    vector_store_time = time.time() - vector_store_start_time
    logger.info(f"⏱️ 벡터 DB 저장 완료: {vector_store_time:.2f}초")



