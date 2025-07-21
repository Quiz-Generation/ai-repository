import logging
import traceback
from http import HTTPStatus
from typing import Any, Dict

from fastapi import HTTPException


class ErrorCode:
    class Common:
        SUCCESS = ("000000", "success", "특정 처리에 대한 성공")
        DEFAULT_ERROR = (
            "C001",
            "일시적인 오류입니다.\n잠시 후 다시 시도해주세요.\n같은 현상이 반복되면 관리자에게 문의해주세요",
            "특정 처리에 대한 실패 (기본 에러)",
        )
        TIMEOUT_ERROR = (
            "C002",
            "타임아웃 발생",
            "요청에 대한 타임아웃 발생한 경우",
        )
        NO_REQUIRED_INPUT = (
            "C003",
            "필수값을 입력해주세요.",
            "필수값 입력 안한 경우",
        )
        BAD_REQUEST = (
            "C004",
            "잘못된 요청입니다.",
            "잘못된 요청을 보낸 경우",
        )
        DB_ERROR = (
            "C005",
            "DB처리에 실패했습니다."
            "DB 관려된 에러가 발생한 경우",
        )
        UNPROCESSABLE_ENTITY_ERROR = (
            "C006",
            "잘못된 데이터 형식입니다.",
            "잘못된 입력 데이터,필수 필드 누락,호환되지 않는 데이터 유형인 경우",
        )
        NO_REQUIRED_PARAMETER = (
            "C007",
            "필수 파라미터가 누락되었습니다.",
            "필수 파라미터가 누락되었을 때",
        )
        NOT_ALLOWED_DOMAIN = (
            "C008",
            "허용되지 않은 도메인입니다.",
            "허용되지 않은 도메인입니다.",
        )
    class Document:
        PDF_UPLOAD_ERROR = (
            "D001",
            "PDF 파일만 업로드 가능합니다.",
            "PDF 파일만 업로드 가능합니다.",
        )
        PDF_ANALYSIS_ERROR = (
            "D002",
            "PDF 분석에 실패했습니다.",
            "PDF 분석에 실패했습니다.",
        )
        PDF_EXTRACT_ERROR = (
            "D003",
            "PDF 추출에 실패했습니다.",
            "PDF 추출에 실패했습니다.",
        )



class JSendError(Exception):
    status: str = "error"
    code: str
    message: str
    data: Dict[str, Any] | None | str = {}

    def __init__(
        self,
        code: str,
        message: str,
        data: Dict[str, Any] | None | str = {},
    ) -> None:
        self.code: str = code
        self.message: str = message
        self.data: Dict[str, Any] | None | str = data

    def __call__(self, **kwargs: Any) -> Any:
        return JSendError(
            code=self.code,
            message=self.message,
            data=kwargs,
        )

class ValidateError(Exception):
    pass

class PaymentRequestError(Exception):
    pass



def handle_http_server_error(e: Exception, logger: logging.Logger):
    logger.error(
        f"""
            error: {str(e)}
            traceback: {traceback.format_exc()}
        """
    )
    raise HTTPException(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        detail = f"Internal server error"
    )