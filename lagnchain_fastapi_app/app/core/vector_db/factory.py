"""
🏭 Vector Database Factory
"""
from typing import Type
from .base import VectorDatabase
from .milvus_db import MilvusDB
from .weaviate_db import WeaviateDB
from .faiss_db import FaissDB


class VectorDBFactory:
    """벡터 데이터베이스 팩토리"""

    _databases = {
        "milvus": MilvusDB,     # 1순위
        "weaviate": WeaviateDB, # 2순위
        "faiss": FaissDB,       # 3순위
    }

    @classmethod
    def create(cls, db_type: str, db_path: str) -> VectorDatabase:
        """벡터 DB 인스턴스 생성"""
        if db_type not in cls._databases:
            raise ValueError(f"지원하지 않는 벡터 DB 타입: {db_type}")

        db_class = cls._databases[db_type]
        return db_class(db_path)

    @classmethod
    def get_supported_types(cls) -> list[str]:
        """지원하는 DB 타입 목록 (우선순위 순)"""
        return ["milvus", "weaviate", "faiss"]

    @classmethod
    def get_priority_order(cls) -> dict[str, int]:
        """벡터 DB 우선순위 반환"""
        return {
            "milvus": 1,
            "weaviate": 2,
            "faiss": 3
        }

    @classmethod
    def get_recommended_db(cls) -> str:
        """권장 벡터 DB 반환"""
        return "milvus"