"""
🚀 Milvus Vector Database Implementation (1순위)
"""
import logging
import os
import uuid
from typing import List, Dict, Any, Optional
from .base import VectorDatabase, VectorDocument, SearchResult

logger = logging.getLogger(__name__)


class MilvusDB(VectorDatabase):
    """Milvus 벡터 데이터베이스 구현체 (1순위 - 고성능 분산)"""

    def __init__(
            self, db_path: str):
        super().__init__(db_path)
        self.client = None
        self.collection_name = "pdf_documents"
        # self.dimension = 384  # sentence-transformers all-MiniLM-L6-v2 기본 차원
        self.dimension = 3072  # openai-embed-large

        # 환경변수에서 호스트와 포트 읽기
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")

        # 🔥 강제로 서버 모드 사용 (Docker 컨테이너와 연결)
        self.use_lite = False  # 무조건 서버 모드

        logger.info(f"INIT Milvus 설정 - dimension: {self.dimension}, Host: {self.host}, Port: {self.port}, Lite: {self.use_lite}")

    async def initialize(
            self,
            model_name: Optional[str] = None
        ) -> None:
        """Milvus 클라이언트 초기화 및 컬렉션 생성"""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            if model_name == "text-embedding-3-large":
                self.dimension = 3072
            elif model_name == "all-MiniLM-L6-v2":
                self.dimension = 384

            logger.info("STEP_VECTOR Milvus 초기화 시작")

            # Milvus Lite 모드 시도
            if self.use_lite:
                try:
                    logger.info("STEP_VECTOR Milvus Lite 모드 시도")

                    # Lite 모드 연결 (로컬 파일 기반)
                    lite_db_path = f"{self.db_path}/milvus_lite.db"
                    connections.connect(
                        alias="default",
                        uri=lite_db_path,  # Lite 모드는 파일 경로 사용
                        # host와 port는 사용하지 않음
                    )

                    logger.info(f"SUCCESS Milvus Lite 연결 완료: {lite_db_path}")

                except Exception as lite_error:
                    logger.warning(f"WARNING Milvus Lite 모드 실패: {lite_error}")
                    self.use_lite = False

            # 서버 모드 폴백
            if not self.use_lite:
                logger.info("STEP_VECTOR Milvus 서버 모드 시도")
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port
                )
                logger.info(f"SUCCESS Milvus 서버 연결 완료: {self.host}:{self.port}")

            # 컬렉션 스키마 정의
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]

            schema = CollectionSchema(fields, description="PDF 문서 벡터 저장소")

            # 컬렉션 생성 또는 로드
            if utility.has_collection(self.collection_name):
                logger.info(f"STEP_VECTOR 기존 Milvus 컬렉션 로드: {self.collection_name}")
                self.client = Collection(self.collection_name)
            else:
                logger.info(f"STEP_VECTOR 새 Milvus 컬렉션 생성: {self.collection_name}")
                self.client = Collection(self.collection_name, schema)

                # 인덱스 생성 (HNSW 알고리즘 사용)
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64}
                }
                self.client.create_index("embedding", index_params)

            # 컬렉션 로드 (검색 가능하도록)
            self.client.load()

            mode = "Lite" if self.use_lite else "Server"
            logger.info(f"SUCCESS Milvus ({mode}) 초기화 완료")

        except ImportError:
            logger.error("ERROR pymilvus 라이브러리가 설치되지 않았습니다. 'pip install pymilvus[model]' 실행하세요.")
            raise ImportError("pymilvus 라이브러리가 필요합니다")
        except Exception as e:
            logger.error(f"ERROR Milvus 초기화 실패: {e}")
            if not self.use_lite:
                logger.warning("WARNING Milvus 서버가 실행 중인지 확인하세요")
                logger.info("TIP: Milvus Lite 모드로 전환하려면 use_lite=True 설정")
            raise

    async def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """문서들을 Milvus에 추가"""
        try:
            if not self.client:
                await self.initialize()

            # 데이터 준비
            ids = [doc.id for doc in documents]
            contents = [doc.content for doc in documents]
            embeddings = [doc.embedding for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # Milvus에 삽입
            entities = [ids, contents, embeddings, metadatas]
            insert_result = self.client.insert(entities)

            # 인덱스 플러시 (즉시 검색 가능하도록)
            self.client.flush()

            logger.info(f"SUCCESS Milvus에 {len(documents)}개 문서 추가 완료")
            return ids

        except Exception as e:
            logger.error(f"ERROR Milvus 문서 추가 실패: {e}")
            raise

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Milvus에서 유사도 검색"""
        try:
            if not self.client:
                await self.initialize()

            # 검색 파라미터 설정
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}  # HNSW 검색 파라미터
            }

            # 필터 조건 생성
            expr = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f'metadata["{key}"] == "{value}"')
                    else:
                        filter_conditions.append(f'metadata["{key}"] == {value}')
                expr = " and ".join(filter_conditions)

            # 벡터 검색 수행
            search_results = self.client.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["id", "content", "metadata"]
            )

            results = []
            for hits in search_results:
                for hit in hits:
                    doc = VectorDocument(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        embedding=[],  # 검색 시에는 임베딩 반환하지 않음
                        metadata=hit.entity.get("metadata", {})
                    )

                    results.append(SearchResult(
                        document=doc,
                        score=hit.score,
                        distance=hit.distance
                    ))

            logger.info(f"SUCCESS Milvus 검색 완료: {len(results)}개 결과")
            return results

        except Exception as e:
            logger.error(f"ERROR Milvus 검색 실패: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """문서 삭제"""
        try:
            if not self.client:
                await self.initialize()

            # ID로 문서 삭제
            expr = f'id == "{document_id}"'
            self.client.delete(expr)

            logger.info(f"SUCCESS 문서 {document_id} 삭제 완료")
            return True

        except Exception as e:
            logger.error(f"ERROR 문서 삭제 실패: {e}")
            return False

    async def get_document_count(self) -> int:
        """총 문서 수 조회"""
        try:
            if not self.client:
                await self.initialize()

            from pymilvus import utility
            stats = utility.get_query_segment_info(self.collection_name)
            total_count = sum(stat.num_rows for stat in stats)

            return total_count

        except Exception as e:
            logger.error(f"ERROR 문서 수 조회 실패: {e}")
            return 0

    async def get_all_documents(self, limit: Optional[int] = None) -> List[VectorDocument]:
        """모든 문서 조회"""
        try:
            if not self.client:
                raise Exception("벡터 DB가 초기화되지 않았습니다.")

            limit_count = limit if limit else 1000  # 기본 제한
            logger.info(f"STEP_QUERY Milvus 전체 문서 조회 시작 (제한: {limit_count})")

            # 🔥 timestamp 기반 쿼리 (실제로 작동하는 방법)
            expr = 'metadata["upload_timestamp"] != ""'

            query_results = self.client.query(
                expr=expr,
                output_fields=["id", "content", "metadata"],
                limit=limit_count
            )

            documents = self._parse_query_results(query_results)
            logger.info(f"SUCCESS Milvus에서 {len(documents)}개 문서 조회 완료")
            return documents

        except Exception as e:
            logger.error(f"ERROR Milvus 문서 조회 실패: {e}")
            return []

    def _parse_query_results(self, query_results) -> List[VectorDocument]:
        """쿼리 결과를 VectorDocument 리스트로 변환"""
        documents = []
        for result in query_results:
            doc = VectorDocument(
                id=result.get("id", ""),
                content=result.get("content", ""),
                embedding=[],  # 임베딩은 조회하지 않음 (성능상 이유)
                metadata=result.get("metadata", {})
            )
            documents.append(doc)
        return documents

    async def health_check(self) -> Dict[str, Any]:
        """Milvus 헬스체크"""
        try:
            from pymilvus import connections, utility

            # 기존 연결 정리
            if connections.has_connection("default"):
                connections.disconnect("default")

            # 실제 연결 시도
            if self.use_lite:
                lite_db_path = f"{self.db_path}/milvus_lite.db"
                connections.connect(
                    alias="default",
                    uri=lite_db_path
                )
                connection_info = f"Lite mode: {lite_db_path}"
                mode = "Lite"
            else:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port
                )
                connection_info = f"Server: {self.host}:{self.port}"
                mode = "Server"

            # 연결 성공 시 추가 확인
            collection_exists = utility.has_collection(self.collection_name)

            if collection_exists:
                # 기존 컬렉션 통계 조회
                from pymilvus import Collection
                collection = Collection(self.collection_name)
                collection.load()
                document_count = collection.num_entities
            else:
                document_count = 0

            logger.info(f"SUCCESS Milvus 헬스체크 성공 - {mode} 모드")

            return {
                "status": "healthy",
                "db_type": "milvus",
                "mode": mode,
                "priority": 1,
                "document_count": document_count,
                "collection_name": self.collection_name,
                "collection_exists": collection_exists,
                "dimension": self.dimension,
                "library_available": True,
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "storage": "file-based" if self.use_lite else "distributed",
                "connection": connection_info
            }

        except ImportError:
            return {
                "status": "unhealthy",
                "db_type": "milvus",
                "error": "pymilvus 라이브러리가 설치되지 않음",
                "library_available": False
            }
        except Exception as e:
            logger.error(f"ERROR Milvus 헬스체크 실패: {e}")
            return {
                "status": "unhealthy",
                "db_type": "milvus",
                "error": f"연결 실패: {str(e)}",
                "library_available": True,
                "attempted_connection": f"{self.host}:{self.port}" if not self.use_lite else "Lite mode",
                "use_lite": self.use_lite
            }

    async def clear_all(self) -> bool:
        """Milvus 컬렉션의 모든 데이터 삭제"""
        try:
            if not self.client:
                await self.initialize()

            from pymilvus import utility

            logger.info("🚨 DANGER Milvus 컬렉션 전체 삭제 시작")

            # 컬렉션이 존재하는지 확인
            if utility.has_collection(self.collection_name):
                # 컬렉션 삭제 (데이터와 함께)
                utility.drop_collection(self.collection_name)
                logger.info(f"SUCCESS Milvus 컬렉션 '{self.collection_name}' 삭제 완료")

                # 컬렉션 재생성 (initialize 메서드 재호출)
                await self.initialize()
                logger.info(f"SUCCESS Milvus 컬렉션 '{self.collection_name}' 재생성 완료")

                return True
            else:
                logger.info(f"INFO Milvus 컬렉션 '{self.collection_name}'이 존재하지 않음")
                return True

        except Exception as e:
            logger.error(f"ERROR Milvus 전체 삭제 실패: {e}")
            return False