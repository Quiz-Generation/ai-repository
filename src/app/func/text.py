from typing import List

async def create_text_chunks(
        logger,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        📝 텍스트를 청크로 분할 (벡터 DB용)
        - 벡터 DB에서 사용
        """
        try:
            logger.info(
                f"""
                    STEP_CHUNK 텍스트 청킹 시작: 길이={len(text)}, 청크크기={chunk_size}, 오버랩={chunk_overlap}
                """
            )

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