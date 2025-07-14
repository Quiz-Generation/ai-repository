"""
🏭 Vector Database Factory
"""
from typing import Dict, Type, List
from .base import VectorDatabase
from .milvus_db import MilvusDB


class VectorDBFactory:
    """벡터 데이터베이스 팩토리 클래스"""

    _registry: Dict[str, Type[VectorDatabase]] = {
        "milvus": MilvusDB,    # 1순위 - 고성능 분산 벡터 DB
    }

    _priority_order = {
        "milvus": 1,  # 최우선 (Docker 컨테이너)
    }

    @classmethod
    def create(cls, db_type: str, db_path: str) -> VectorDatabase:
        """벡터 DB 인스턴스 생성"""
        if db_type not in cls._registry:
            raise ValueError(f"지원하지 않는 벡터 DB 타입: {db_type}")

        db_class = cls._registry[db_type]
        return db_class(db_path)

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """지원되는 벡터 DB 타입 목록"""
        return list(cls._registry.keys())

    @classmethod
    def get_priority_order(cls) -> Dict[str, int]:
        """우선순위 정보"""
        return cls._priority_order.copy()

    @classmethod
    def get_recommended_db(cls) -> str:
        """권장 벡터 DB 반환"""
        return "milvus"