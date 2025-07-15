import os
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.common.conf.settings import settings
from src.common.utils.logger import set_logger

logger = set_logger('milvus')
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

MILVUS_COLLECTION = 'pdf_documents'
MILVUS_URI ="http://localhost:19530"

class MilvusDB:
    def __init__(self):
        logger.info(
            f"""
                [MilvusDB 초기화]
                "MILVUS_URI": {MILVUS_URI}
                "MILVUS_COLLECTION": {MILVUS_COLLECTION}
                "EMBEDDING_MODEL": {getattr(embeddings, 'model', 'unknown')}
            """
        )
        self.store = Milvus(
            embedding_function=embeddings,
            collection_name=MILVUS_COLLECTION,
            connection_args={"uri": MILVUS_URI},
            auto_id=True,
            drop_old=False,
        )

    async def add_documents_async(
            self,
            docs,
            metadata,
            chunk_size=2000,
            chunk_overlap=200
            ):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        all_docs = []
        for doc_text in docs:  # 예: docs = [“전체 텍스트1”, “전체 텍스트2”]
            chunks = splitter.split_text(doc_text)
            for i, chunk in enumerate(chunks):
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={**metadata, "chunk_index": i}
                ))

        await self.store.aadd_documents(all_docs)

    async def search_async(self, query, k=3):
        return await self.store.asimilarity_search(query=query, k=k)

    async def delete_async(self, expr):
        await self.store.adelete(expr=expr)

# 전역 인스턴스 선언
milvus_db = MilvusDB()

