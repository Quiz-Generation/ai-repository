from typing import Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel
from pydantic.generics import GenericModel

T = TypeVar('T')


class JSendResponse(GenericModel, Generic[T]):
    status: str = "success"
    data: Optional[T] = None


class JSendErrorResponse(BaseModel):
    status: str = "error"
    code: str
    message: str


# NOTE: 외부용, 내부 통신시 JSendResponse(data=CommonOutput) 으로 반환
class CommonOutput(BaseModel):
    message: str | None = None