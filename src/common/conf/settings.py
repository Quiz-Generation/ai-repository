import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    STAGE: str = os.getenv('STAGE', "")

    ERR_LOG_PATH: str = os.getenv('ERR_LOG_PATH', "")
    TMP_FILE_PATH: str = os.getenv('TMP_FILE_PATH', "")
    DEFAULT_LOGGING_PATH: str = os.getenv('DEFAULT_LOGGING_PATH', "")
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', "")

    #레디스
    REDIS_HOST: str = os.getenv('REDIS_HOST', "")
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', 0))

    # PDF 로더 설정
    PDF_FORCE_FAST_LOADER: bool = os.getenv('PDF_FORCE_FAST_LOADER', 'true').lower() == 'true'
    PDF_FAST_LOADER_TYPE: str = os.getenv('PDF_FAST_LOADER_TYPE', 'pymupdf')  # pymupdf, pypdf


    class Config:
        config_path = Path(__file__)
        env_file = f"{config_path.parent.parent.parent.parent}/.env"


settings = Settings()
