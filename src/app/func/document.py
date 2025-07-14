"""
📄 PDF Helper
"""
from datetime import datetime
import os
from typing import Dict, Any
from fastapi import UploadFile

from src.app.core.pdf_loader.factory import PDFLoaderFactory


# class PDFHelper:
#     """PDF 처리 유틸리티"""

#     def __init__(self):
#         pass

#     async def extract_text_from_file(self, file: UploadFile) -> str:
#         """업로드된 PDF 파일에서 텍스트 추출"""
#         # TODO: 실제 PDF 텍스트 추출 구현
#         # import PyPDF2 또는 pdfplumber 등 사용
#         return "추출된 텍스트 내용"

#     async def extract_text_from_path(self, file_path: str) -> str:
#         """파일 경로에서 PDF 텍스트 추출"""
#         # TODO: 실제 PDF 텍스트 추출 구현
#         return "추출된 텍스트 내용"

#     def validate_pdf_file(self, file: UploadFile) -> bool:
#         """PDF 파일 유효성 검증"""
#         if not file.filename:
#             return False

#         # 파일 확장자 검사
#         if not file.filename.lower().endswith('.pdf'):
#             return False

#         # 파일 크기 검사 (예: 10MB 제한)
#         # TODO: 실제 파일 크기 검사 구현

#         return True

#     def get_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
#         """PDF 메타데이터 추출"""
#         # TODO: 실제 PDF 메타데이터 추출 구현
#         return {
#             "title": "",
#             "author": "",
#             "subject": "",
#             "creator": "",
#             "producer": "",
#             "creation_date": None,
#             "modification_date": None,
#             "pages": 0
#         }
async def _validate_pdf_file(
        logger,
        file: UploadFile
    ) -> bool:
        """PDF 파일 유효성 검증"""
        logger.info(
            f"""
                [PDF 파일 유효성 검증 시작]
            """
        )
        if not file.filename:
            return False
        if not file.filename.lower().endswith('.pdf'):
            logger.error(
                f"""
                    [PDF 파일 유효성 검증 실패]
                    "파일명": {file.filename}
                """
            )
            return False
        return True


async def _extract_pdf_with_selected_loader(
        logger,
        file: UploadFile,
        loader_type: str
    ):
        """선택된 로더로 PDF 텍스트 추출"""
        try:
            # 🔥 파일 포인터를 처음으로 리셋 (안전장치)
            await file.seek(0)

            # 팩토리에서 로더 생성
            pdf_loader = PDFLoaderFactory.create(loader_type)

            # 파일 유효성 검증
            if not pdf_loader.validate_file(file):
                raise ValueError(f"파일 유효성 검사 실패: {file.filename}")

            # 텍스트 추출
            pdf_content = await pdf_loader.extract_text_from_file(file)

            logger.info(f"STEP4 {loader_type} 로더로 텍스트 추출 완료")
            return pdf_content

        except Exception as e:
            logger.error(f"ERROR PDF 추출 실패 ({loader_type}): {e}")

            # 실패 시 fallback 로더 시도
            if loader_type != "pymupdf":
                logger.info("FALLBACK PyMuPDF 로더로 재시도")
                # 🔥 폴백 시도 전에도 파일 포인터 리셋
                await file.seek(0)
                fallback_loader = PDFLoaderFactory.create("pymupdf")
                return await fallback_loader.extract_text_from_file(file)
            else:
                raise


async def process_pdf(
        logger,
        file: UploadFile,
        loader: str
    ) -> Dict[str, Any]:
        """
        🚀 벡터 DB 통합용 PDF 처리 메서드
        - 동적 로더 선택 및 텍스트 추출
        - 폴백 메커니즘 포함
        """
        loader_used = loader
        fallback_attempts = 0

        try:
            logger.info(
                f"""
                    STEP_PDF PDF 처리 시작:
                    "파일명": {file.filename}
                    "로더": {loader_used}
                """
            )

            # 1. 파일 검증
            if not await _validate_pdf_file(
                logger=logger,
                file=file
            ):
                logger.error(
                    f"""
                        [PDF 파일 유효성 검증 실패]
                        "파일명": {file.filename}
                    """
                )
                return {}

            # 🔥 파일 포인터를 처음으로 리셋 (중요!)
            await file.seek(0)

            # 2. 선택된 로더로 PDF 처리 시도
            try:
                pdf_content = await _extract_pdf_with_selected_loader(
                    logger=logger,
                    file=file,
                    loader_type=loader_used
                )

                if not pdf_content or not hasattr(pdf_content, 'text') or not pdf_content.text.strip():
                    raise ValueError("추출된 텍스트가 비어있습니다")

                logger.info(f"SUCCESS {loader_used} 로더로 PDF 처리 완료")

                return {
                    "success": True,
                    "content": pdf_content.text,
                    "loader_used": loader_used,
                    "processing_time": datetime.now().isoformat(),
                    "fallback_attempts": fallback_attempts,
                    "content_length": len(pdf_content.text),
                    "metadata": getattr(pdf_content, 'metadata', {})
                }

            except Exception as e:
                logger.warning(f"WARNING {loader_used} 로더 실패: {e}")

                # 3. 폴백 메커니즘 - 우선순위 순서로 시도
                fallback_loaders = ["pymupdf", "pdfplumber", "pypdf", "pdfminer"]

                for fallback_loader in fallback_loaders:
                    if fallback_loader == loader_used:
                        continue

                    try:
                        fallback_attempts += 1
                        logger.info(f"FALLBACK {fallback_loader} 로더로 재시도 ({fallback_attempts})")

                        # 🔥 폴백 시도 전에도 파일 포인터 리셋
                        await file.seek(0)

                        pdf_content = await _extract_pdf_with_selected_loader(
                            logger=logger,
                            file=file,
                            loader_type=fallback_loader
                        )

                        if pdf_content and hasattr(pdf_content, 'text') and pdf_content.text.strip():
                            logger.info(f"SUCCESS {fallback_loader} 폴백 로더로 PDF 처리 완료")

                            return {
                                "success": True,
                                "content": pdf_content.text,
                                "loader_used": fallback_loader,
                                "processing_time": datetime.now().isoformat(),
                                "fallback_attempts": fallback_attempts,
                                "content_length": len(pdf_content.text),
                                "metadata": getattr(pdf_content, 'metadata', {}),
                                "fallback_reason": f"원본 로더({loader_used}) 실패: {str(e)}"
                            }

                    except Exception as fallback_error:
                        logger.warning(f"WARNING {fallback_loader} 폴백 로더도 실패: {fallback_error}")
                        continue

                # 모든 로더 실패
                return {
                    "success": False,
                    "error": f"모든 PDF 로더 실패. 마지막 오류: {str(e)}",
                    "loader_used": loader_used,
                    "fallback_attempts": fallback_attempts
                }

        except Exception as e:
            logger.error(f"ERROR PDF 처리 중 예외 발생: {e}")
            return {
                "success": False,
                "error": f"PDF 처리 중 예외: {str(e)}",
                "loader_used": loader_used,
                "fallback_attempts": fallback_attempts
            }

