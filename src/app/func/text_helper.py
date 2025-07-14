"""
📝 Text Helper
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..models.document_model import DocumentChunk
from ..core.config import settings

logger = logging.getLogger(__name__)


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
        try:
            logger.info(f"STEP5a 텍스트 청킹 시작: 텍스트 길이={len(text)}, 청크크기={chunk_size}, 오버랩={chunk_overlap}")

            if not text or not text.strip():
                logger.warning("WARNING 빈 텍스트입니다.")
                return []

            chunks = []
            text_length = len(text)
            start_index = 0
            loop_count = 0  # 무한루프 방지용 카운터

            while start_index < text_length and loop_count < 1000:  # 최대 1000번 반복 제한
                loop_count += 1
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
                    logger.debug(f"STEP5a 청크 {len(chunks)}: 시작={start_index}, 끝={end_index}, 길이={len(chunk_content)}")

                # 다음 청크 시작점 계산
                next_start = max(start_index + 1, end_index - chunk_overlap)  # 최소 1글자씩 진행

                if next_start >= text_length:  # 텍스트 끝에 도달
                    break

                if next_start <= start_index:  # 진행되지 않는 경우
                    logger.warning(f"WARNING 청킹에서 진행되지 않음: start_index={start_index}, next_start={next_start}")
                    break

                start_index = next_start

            if loop_count >= 1000:
                logger.warning(f"WARNING 청킹이 1000개 제한에 도달했습니다. 현재 {len(chunks)}개 청크 생성됨")

            logger.info(f"STEP5b 텍스트 청킹 완료: {len(chunks)}개 청크 생성")
            return chunks

        except Exception as e:
            logger.error(f"ERROR 텍스트 청킹 실패: {e}")
            return []

    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 불필요한 공백 제거
        text = ' '.join(text.split())

        # 특수 문자 정리 (필요시)
        # text = re.sub(r'[^\w\s가-힣]', ' ', text)

        return text.strip()

    @staticmethod
    def create_text_chunks(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        📝 정적 메서드: 텍스트를 청크로 분할 (벡터 DB용)
        - sentence-transformers와 벡터 DB에서 사용
        """
        try:
            logger.info(f"STEP_CHUNK 텍스트 청킹 시작: 길이={len(text)}, 청크크기={chunk_size}, 오버랩={chunk_overlap}")

            if not text or not text.strip():
                logger.warning("WARNING 빈 텍스트입니다.")
                return []

            chunks = []
            text_length = len(text)
            start_index = 0
            loop_count = 0  # 무한루프 방지용 카운터

            while start_index < text_length and loop_count < 1000:  # 최대 1000번 반복 제한
                loop_count += 1
                end_index = min(start_index + chunk_size, text_length)

                # 문장 경계에서 자르기 시도 (더 나은 청킹을 위해)
                if end_index < text_length:
                    # 마침표, 줄바꿈, 공백 순서로 경계 찾기
                    boundaries = [
                        text.rfind('.', start_index, end_index),
                        text.rfind('\n', start_index, end_index),
                        text.rfind(' ', start_index, end_index)
                    ]

                    best_boundary = max([b for b in boundaries if b > start_index], default=-1)
                    if best_boundary > start_index:
                        end_index = best_boundary + 1

                chunk_content = text[start_index:end_index].strip()
                if chunk_content:
                    chunks.append(chunk_content)
                    logger.debug(f"STEP_CHUNK 청크 {len(chunks)}: 시작={start_index}, 끝={end_index}, 길이={len(chunk_content)}")

                # 다음 청크 시작점 계산 (오버랩 고려)
                next_start = max(start_index + 1, end_index - chunk_overlap)

                if next_start >= text_length or next_start <= start_index:
                    break

                start_index = next_start

            if loop_count >= 1000:
                logger.warning(f"WARNING 청킹이 1000개 제한에 도달했습니다. 현재 {len(chunks)}개 청크 생성됨")

            logger.info(f"SUCCESS 텍스트 청킹 완료: {len(chunks)}개 청크 생성")
            return chunks

        except Exception as e:
            logger.error(f"ERROR 텍스트 청킹 실패: {e}")
            return [text]  # 실패 시 원본 텍스트 그대로 반환

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트 임베딩 생성"""
        # TODO: 실제 임베딩 생성 구현
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(settings.EMBEDDING_MODEL)
        # return model.encode(texts).tolist()

        # 임시로 더미 임베딩 반환
        return [[0.1] * 384 for _ in texts]