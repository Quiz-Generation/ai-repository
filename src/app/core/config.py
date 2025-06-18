"""
⚙️ Application Configuration
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """애플리케이션 설정"""

    # API 설정
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PDF Vector DB API"

    # OpenAI 설정
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    # 파일 업로드 설정
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf"]
    UPLOAD_DIR: str = "data/uploads"
    TEMP_DIR: str = "data/temp"

    # 벡터 DB 설정
    VECTOR_DB_TYPE: str = "milvus"  # milvus, weaviate, faiss
    VECTOR_DB_PATH: str = "data/vector_db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # 처리된 문서 저장 경로
    PROCESSED_DOCS_DIR: str = "document/processed"
    FAILED_DOCS_DIR: str = "document/failed"

    model_config = {
        "env_file": ".env",
        "extra": "allow"  # 추가 필드 허용
    }


# 전역 설정 인스턴스
settings = Settings()