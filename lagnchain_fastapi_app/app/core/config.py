#!/usr/bin/env python3
"""
애플리케이션 설정 관리
"""
import os
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # 기본 설정
    APP_NAME: str = "AI Quiz Generator"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, description="디버그 모드")

    # 서버 설정
    HOST: str = Field(default="127.0.0.1", description="서버 호스트")
    PORT: int = Field(default=7000, description="서버 포트")

    # OpenAI 설정
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API 키")
    OPENAI_MODEL: str = Field(default="gpt-3.5-turbo", description="사용할 OpenAI 모델")

    # 업로드 설정
    UPLOAD_DIR: str = Field(default="./uploads", description="파일 업로드 디렉토리")
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024, description="최대 파일 크기 (10MB)")
    ALLOWED_EXTENSIONS: list = Field(default=[".pdf"], description="허용된 파일 확장자")

    # 벡터 DB 설정 (ChromaDB만 사용)
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chromadb")  # chromadb, fallback
    VECTOR_DATA_DIR: str = Field(default="./vector_data", description="벡터 데이터 저장 디렉토리")

    # 퀴즈 생성 설정
    DEFAULT_QUIZ_COUNT: int = Field(default=10, description="기본 퀴즈 문제 수")
    MAX_QUIZ_COUNT: int = Field(default=50, description="최대 퀴즈 문제 수")

    # 로깅 설정
    LOG_LEVEL: str = Field(default="INFO", description="로그 레벨")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="로그 포맷"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_database_config(self) -> Dict[str, Any]:
        """벡터 데이터베이스 설정 반환"""
        return {
            "type": self.VECTOR_DB_TYPE.lower(),
            "data_dir": self.VECTOR_DATA_DIR
        }

    def get_openai_config(self) -> Dict[str, Any]:
        """OpenAI 설정 반환"""
        return {
            "api_key": self.OPENAI_API_KEY,
            "model": self.OPENAI_MODEL
        }

    def get_upload_config(self) -> Dict[str, Any]:
        """파일 업로드 설정 반환"""
        return {
            "upload_dir": self.UPLOAD_DIR,
            "max_file_size": self.MAX_FILE_SIZE,
            "allowed_extensions": self.ALLOWED_EXTENSIONS
        }


# 전역 설정 인스턴스
def get_settings() -> Settings:
    """설정 인스턴스 반환"""
    return Settings()


# 설정 캐싱
_cached_settings: Optional[Settings] = None

def get_cached_settings() -> Settings:
    """캐시된 설정 반환 (성능 최적화)"""
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = get_settings()
    return _cached_settings


def load_development_config():
    """개발 환경 설정"""
    pass


def load_production_config():
    """프로덕션 환경 설정"""
    pass


def load_testing_config():
    """테스트 환경 설정"""
    pass


if __name__ == "__main__":
    # 설정 테스트
    print("=== 설정 정보 ===")
    print(f"앱 이름: {get_settings().APP_NAME}")
    print(f"디버그 모드: {get_settings().DEBUG}")
    print(f"OpenAI API 키: {'설정됨' if get_settings().OPENAI_API_KEY else '미설정'}")
    print(f"데이터베이스: {get_settings().DATABASE_URL}")
    print(f"벡터 DB: {get_settings().VECTOR_DB_TYPE}")
    print(f"업로드 경로: {get_settings().UPLOAD_DIR}")