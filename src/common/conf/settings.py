import os
from pathlib import Path

from pydantic_settings import BaseSettings

def safe_int(value, default=0):
    try:
        # tcp://host:port 형태인 경우
        if isinstance(value, str) and value.startswith('tcp://'):
            return int(value.split(':')[-1])
        return int(value)
    except (ValueError, TypeError, AttributeError):
        return default

class Settings(BaseSettings):
    STAGE: str = os.getenv('STAGE', "")

    ERR_LOG_PATH: str = os.getenv('ERR_LOG_PATH', "")
    TMP_FILE_PATH: str = os.getenv('TMP_FILE_PATH', "")
    DEFAULT_LOGGING_PATH: str = os.getenv('DEFAULT_LOGGING_PATH', "")
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', "")

    #레디스
    REDIS_HOST: str = os.getenv('REDIS_HOST', "")
    REDIS_PORT: int = safe_int(os.getenv('REDIS_PORT', 6379))


    class Config:
        config_path = Path(__file__)
        env_file = f"{config_path.parent.parent.parent.parent}/.env"


settings = Settings()
