"""
애플리케이션 설정 관리
모든 환경 변수와 설정을 중앙에서 관리
"""
import os
from typing import Optional
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


class Settings:
    """애플리케이션 설정 클래스"""

    # 기본 애플리케이션 설정
    APP_NAME: str = "LangChain FastAPI Quiz Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # API 설정
    API_PREFIX: str = "/api/v1"

    # OpenAI 설정
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

    # Anthropic 설정 (미래용)
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")


    # 벡터 데이터베이스 설정
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    CHROMADB_PATH: str = os.getenv("CHROMADB_PATH", "./chroma_db")
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chromadb")  # weaviate, chromadb

    # 파일 업로드 설정
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB
    ALLOWED_EXTENSIONS: list = ["pdf", "txt", "docx"]

    # 퀴즈 생성 설정
    MAX_QUESTIONS_PER_QUIZ: int = int(os.getenv("MAX_QUESTIONS_PER_QUIZ", "20"))
    MIN_QUESTIONS_PER_QUIZ: int = int(os.getenv("MIN_QUESTIONS_PER_QUIZ", "1"))
    DEFAULT_DIFFICULTY: str = os.getenv("DEFAULT_DIFFICULTY", "medium")

    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")

    # CORS 설정
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")

    def __init__(self):
        """설정 초기화 및 검증"""
        self.validate_settings()

    def validate_settings(self):
        """중요한 설정값들 검증"""
        if not self.OPENAI_API_KEY:
            print("⚠️  경고: OPENAI_API_KEY가 설정되지 않았습니다. 퀴즈 생성 기능이 제한될 수 있습니다.")

        # 업로드 디렉토리 생성
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

    @property
    def database_config(self) -> dict:
        """데이터베이스 설정 반환 (현재 미사용)"""
        # DATABASE_URL 설정이 제거되어 주석 처리
        # return {
        #     "url": self.DATABASE_URL,
        #     "echo": self.DEBUG
        # }
        return {}

    @property
    def vector_db_config(self) -> dict:
        """벡터 데이터베이스 설정 반환"""
        if self.VECTOR_DB_TYPE.lower() == "weaviate":
            return {
                "type": "weaviate",
                "url": self.WEAVIATE_URL
            }
        else:
            return {
                "type": "chromadb",
                "path": self.CHROMADB_PATH
            }

    @property
    def llm_config(self) -> dict:
        """LLM 설정 반환"""
        return {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "model": self.OPENAI_MODEL,
                "temperature": self.OPENAI_TEMPERATURE,
                "max_tokens": self.OPENAI_MAX_TOKENS
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY
            }
        }


# 전역 설정 인스턴스
settings = Settings()


def get_settings() -> Settings:
    """설정 인스턴스 반환"""
    return settings


# 환경별 설정 로드 함수들
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
    print(f"앱 이름: {settings.APP_NAME}")
    print(f"디버그 모드: {settings.DEBUG}")
    print(f"OpenAI API 키: {'설정됨' if settings.OPENAI_API_KEY else '미설정'}")
    print(f"데이터베이스: {settings.DATABASE_URL}")
    print(f"벡터 DB: {settings.VECTOR_DB_TYPE}")
    print(f"업로드 경로: {settings.UPLOAD_DIR}")