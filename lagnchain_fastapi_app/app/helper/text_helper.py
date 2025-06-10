"""
📝 Text Helper
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..models.document_model import DocumentChunk
from ..core.config import settings


class TextHelper:
    """텍스트 처리 유틸리티"""

    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def split_text_into_chunks(
        self,
        text: str,
        document_id: str = "",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[DocumentChunk]:
        """텍스트를 청크로 분할"""
        # 매개변수 기본값 설정
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        chunks = []

        # 간단한 청킹 로직 (실제로는 더 정교한 분할 필요)
        text_length = len(text)
        start_index = 0
        chunk_index = 0

        while start_index < text_length:
            end_index = min(start_index + chunk_size, text_length)

            # 오버랩을 위해 문장 경계에서 자르기 시도
            if end_index < text_length:
                # 마지막 마침표나 줄바꿈 찾기
                last_sentence_end = text.rfind('.', start_index, end_index)
                last_newline = text.rfind('\n', start_index, end_index)

                boundary = max(last_sentence_end, last_newline)
                if boundary > start_index:
                    end_index = boundary + 1

            chunk_content = text[start_index:end_index].strip()

            if chunk_content:
                chunk = DocumentChunk(
                    id="",  # __post_init__에서 자동 생성
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    start_index=start_index,
                    end_index=end_index,
                    metadata={
                        "length": len(chunk_content),
                        "has_overlap": chunk_index > 0
                    },
                    created_at=datetime.now()
                )
                chunks.append(chunk)
                chunk_index += 1

            # 다음 청크 시작점 계산 (오버랩 고려)
            start_index = max(0, end_index - chunk_overlap)

            # 무한 루프 방지
            if start_index >= end_index:
                break

        return chunks

    def split_text_simple(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """텍스트를 단순 문자열 청크로 분할 (DocumentChunk 없이)"""
        chunks = []
        text_length = len(text)
        start_index = 0

        while start_index < text_length:
            end_index = min(start_index + chunk_size, text_length)

            # 오버랩을 위해 문장 경계에서 자르기 시도
            if end_index < text_length:
                last_sentence_end = text.rfind('.', start_index, end_index)
                last_newline = text.rfind('\n', start_index, end_index)
                boundary = max(last_sentence_end, last_newline)
                if boundary > start_index:
                    end_index = boundary + 1

            chunk_content = text[start_index:end_index].strip()
            if chunk_content:
                chunks.append(chunk_content)

            # 다음 청크 시작점 계산
            start_index = max(0, end_index - chunk_overlap)
            if start_index >= end_index:
                break

        return chunks

    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 불필요한 공백 제거
        text = ' '.join(text.split())

        # 특수 문자 정리 (필요시)
        # text = re.sub(r'[^\w\s가-힣]', ' ', text)

        return text.strip()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트 임베딩 생성"""
        # TODO: 실제 임베딩 생성 구현
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(settings.EMBEDDING_MODEL)
        # return model.encode(texts).tolist()

        # 임시로 더미 임베딩 반환
        return [[0.1] * 384 for _ in texts]