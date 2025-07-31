import time

from fastapi import UploadFile
from fastapi.responses import JSONResponse

from src.app.core.pdf_loader.factory import PDFLoaderFactory
from src.app.document.func.document import Document
from src.app.document.func.document_loader import DocumentLoader
from src.app.document.func.vector import Vector
from src.common.error import ErrorCode, JSendError
from src.common.vector.connect import VectorDBService


class DocumentService:
    def __init__(
        self,
        logger,
        vector_db: VectorDBService,
    ):
        self.logger = logger
        self.vector_db = vector_db
        self.document_loader = DocumentLoader(logger=logger)
        self.document = Document(logger=logger, pdf_loader_factory=PDFLoaderFactory())
        self.vector = Vector(logger=logger, vector_db=vector_db)

    async def upload_document(
        self,
        file: UploadFile,
    ) -> JSONResponse:
        """
        PDF 파일 업로드 시 파일 특성 분석 및 최적 로더 선택

        1. 파일 검증
        2. 파일 특성 분석 및 최적 로더 선택
        3. PDF 내용 추출
        4. 벡터 데이터 베이스 저장
        """
        try:
            total_start_time = time.time()

            self.logger.info(f"PDF 업로드 시작: {file.filename}")

            #1. 파일 검증
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                self.logger.error(
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

            analysis_result = await self.document_loader.analyze_pdf_characteristics(
                file=file
            )
            analysis_time = time.time() - analysis_start_time
            self.logger.info(f"PDF 분석 완료: {analysis_time:.2f}초")

            if not analysis_result.recommended_loader:
                self.logger.error(
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

            extraction_result = await self.document.process_pdf(
                file=file,
                loader=analysis_result.recommended_loader
            )
            extraction_time = time.time() - extraction_start_time
            self.logger.info(f"PDF 추출 완료: {extraction_time:.2f}초")

            if not extraction_result["success"]:
                raise JSendError(
                    code=ErrorCode.Document.PDF_EXTRACT_ERROR[0],
                    message=ErrorCode.Document.PDF_EXTRACT_ERROR[1] + f" {extraction_result.get('error', 'Unknown error')}"
                )

            #4. 벡터 데이터 베이스 저장

            # 🔥 벡터 DB 강제 Milvus 초기화 (기존 서비스 무시)
            vector_init_start_time = time.time()
            vector_init_time = time.time() - vector_init_start_time
            self.logger.info(f"벡터 DB 초기화: {vector_init_time:.2f}초")

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

            vector_result = await self.vector.store_pdf_content(
                pdf_content=extraction_result["content"],
                metadata=metadata,
                chunk_size=auto_chunk_size,
                chunk_overlap=auto_chunk_overlap
            )
            vector_store_time = time.time() - vector_store_start_time
            self.logger.info(f"벡터 DB 저장 완료: {vector_store_time:.2f}초")

            #5. 문제 수 계산

            question_count_result = await self.document.calculate_optimal_question_count(
                content=extraction_result["content"],
                metadata=metadata
            )
            self.logger.info(f"문제 수 계산 완료 총 문제 수: {question_count_result.get('recommended_questions', 0)}")

            # 🔥 파일 ID 가져오기 (파일별 단일 ID)
            file_id = vector_result.get("file_id")

            # 전체 처리 시간 계산
            total_time = time.time() - total_start_time

            response_data = {
                "success": vector_result["success"],
                "message": "PDF 업로드 완료",
                "file_id": file_id,
                "filename": file.filename,
                "chunk_count": vector_result.get("chunk_count", 0),
                "recommended_questions": question_count_result.get("recommended_questions", 0),
                "total_time": total_time,
                "analysis_time": analysis_time,
                "extraction_time": extraction_time,
                "vector_init_time": vector_init_time,
                "vector_store_time": vector_store_time,
                "vector_performance": vector_result.get("performance_metrics", {})
            }

            if not vector_result["success"]:
                response_data["error"] = vector_result.get("error")

            self.logger.info(f"PDF 업로드 완료: {file.filename} 총 문제 수: {question_count_result.get('recommended_questions', 0)}")
            self.logger.info(f"전체 처리 시간: {total_time:.2f}초")
            self.logger.info(f"성능 요약: 분석({analysis_time:.2f}s) + 추출({extraction_time:.2f}s) + 벡터화({vector_store_time:.2f}s)")

            return JSONResponse(content=response_data)
        except Exception as e:
            self.logger.error(f"PDF 업로드 실패: {e}")
            raise JSendError(
                code=ErrorCode.Document.PDF_UPLOAD_ERROR[0],
                message=ErrorCode.Document.PDF_UPLOAD_ERROR[1]
            )

    async def clear_all_documents(
        self,
        confirm_token: str
    ) -> JSONResponse:
        """
        벡터 DB 전체 삭제
        """
        try:
            self.logger.info("벡터 DB 전체 삭제 요청")

            # 전체 삭제 실행
            result = await self.vector_db.clear_all_documents(
                confirm_token=confirm_token
            )

            if result["success"]:
                response_data = {
                    "success": True,
                    "message": result["message"],
                    "vector_db_type": result["vector_db_type"],
                    "deleted_count": result.get("deleted_count", 0),
                    "remaining_count": result.get("remaining_count", 0)
                }
                self.logger.info(f"벡터 DB 전체 삭제 완료: {result.get('deleted_count', 0)}개 삭제")
            else:
                response_data = {
                    "success": False,
                    "message": "전체 삭제 실패",
                    "error": result.get("error"),
                    "vector_db_type": result.get("vector_db_type")
                }

            return JSONResponse(content=response_data)

        except Exception as e:
            self.logger.error(f"ERROR 벡터 DB 전체 삭제 실패: {e}")
            raise JSendError(
                code=ErrorCode.Document.CLEAR_ALL_DOCUMENTS_ERROR[0],
                message=ErrorCode.Document.CLEAR_ALL_DOCUMENTS_ERROR[1]
            )