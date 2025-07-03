from pydantic import BaseModel


# NOTE: 외부용, 내부 통신시 JSendResponse(data=CommonOutput) 으로 반환
class CommonOutput(BaseModel):
    message: str | None = None