"""
📄 PDF Helper
"""
from datetime import datetime
from typing import Any, Dict
from fastapi import UploadFile

from src.app.core.pdf_loader.factory import PDFLoaderFactory


async def _validate_pdf_file(
        logger,
        file: UploadFile
    ) -> bool:
        """PDF 파일 유효성 검증"""
        logger.info(
            f"""
                [PDF 파일 유효성 검증 시작]
            """
        )
        if not file.filename:
            return False
        if not file.filename.lower().endswith('.pdf'):
            logger.error(
                f"""
                    [PDF 파일 유효성 검증 실패]
                    "파일명": {file.filename}
                """
            )
            return False
        return True


async def _extract_pdf_with_selected_loader(
        logger,
        file: UploadFile,
        loader_type: str
    ):
        """선택된 로더로 PDF 텍스트 추출"""
        try:
            # 🔥 파일 포인터를 처음으로 리셋 (안전장치)
            await file.seek(0)

            # 팩토리에서 로더 생성
            pdf_loader = PDFLoaderFactory.create(loader_type)

            # 파일 유효성 검증
            if not pdf_loader.validate_file(file):
                raise ValueError(f"파일 유효성 검사 실패: {file.filename}")

            # 텍스트 추출
            pdf_content = await pdf_loader.extract_text_from_file(file)

            logger.info(f"STEP4 {loader_type} 로더로 텍스트 추출 완료")
            return pdf_content

        except Exception as e:
            logger.error(f"ERROR PDF 추출 실패 ({loader_type}): {e}")

            # 실패 시 fallback 로더 시도
            if loader_type != "pymupdf":
                logger.info("FALLBACK PyMuPDF 로더로 재시도")
                # 🔥 폴백 시도 전에도 파일 포인터 리셋
                await file.seek(0)
                fallback_loader = PDFLoaderFactory.create("pymupdf")
                return await fallback_loader.extract_text_from_file(file)
            else:
                raise


async def process_pdf(
        logger,
        file: UploadFile,
        loader: str
    ) -> Dict[str, Any]:
        """
        🚀 벡터 DB 통합용 PDF 처리 메서드
        - 동적 로더 선택 및 텍스트 추출
        - 폴백 메커니즘 포함
        """
        loader_used = loader
        fallback_attempts = 0

        try:
            logger.info(
                f"""
                    STEP_PDF PDF 처리 시작:
                    "파일명": {file.filename}
                    "로더": {loader_used}
                """
            )

            # 1. 파일 검증
            if not await _validate_pdf_file(
                logger=logger,
                file=file
            ):
                logger.error(
                    f"""
                        [PDF 파일 유효성 검증 실패]
                        "파일명": {file.filename}
                    """
                )
                return {}

            # 🔥 파일 포인터를 처음으로 리셋 (중요!)
            await file.seek(0)

            # 2. 선택된 로더로 PDF 처리 시도
            try:
                pdf_content = await _extract_pdf_with_selected_loader(
                    logger=logger,
                    file=file,
                    loader_type=loader_used
                )

                if not pdf_content or not hasattr(pdf_content, 'text') or not pdf_content.text.strip():
                    raise ValueError("추출된 텍스트가 비어있습니다")

                logger.info(f"SUCCESS {loader_used} 로더로 PDF 처리 완료")

                return {
                    "success": True,
                    "content": pdf_content.text,
                    "loader_used": loader_used,
                    "processing_time": datetime.now().isoformat(),
                    "fallback_attempts": fallback_attempts,
                    "content_length": len(pdf_content.text),
                    "metadata": getattr(pdf_content, 'metadata', {})
                }

            except Exception as e:
                logger.warning(f"WARNING {loader_used} 로더 실패: {e}")

                # 3. 폴백 메커니즘 - 우선순위 순서로 시도
                fallback_loaders = ["pymupdf", "pdfplumber", "pypdf", "pdfminer"]

                for fallback_loader in fallback_loaders:
                    if fallback_loader == loader_used:
                        continue

                    try:
                        fallback_attempts += 1
                        logger.info(f"FALLBACK {fallback_loader} 로더로 재시도 ({fallback_attempts})")

                        # 🔥 폴백 시도 전에도 파일 포인터 리셋
                        await file.seek(0)

                        pdf_content = await _extract_pdf_with_selected_loader(
                            logger=logger,
                            file=file,
                            loader_type=fallback_loader
                        )

                        if pdf_content and hasattr(pdf_content, 'text') and pdf_content.text.strip():
                            logger.info(f"SUCCESS {fallback_loader} 폴백 로더로 PDF 처리 완료")

                            return {
                                "success": True,
                                "content": pdf_content.text,
                                "loader_used": fallback_loader,
                                "processing_time": datetime.now().isoformat(),
                                "fallback_attempts": fallback_attempts,
                                "content_length": len(pdf_content.text),
                                "metadata": getattr(pdf_content, 'metadata', {}),
                                "fallback_reason": f"원본 로더({loader_used}) 실패: {str(e)}"
                            }

                    except Exception as fallback_error:
                        logger.warning(f"WARNING {fallback_loader} 폴백 로더도 실패: {fallback_error}")
                        continue

                # 모든 로더 실패
                return {
                    "success": False,
                    "error": f"모든 PDF 로더 실패. 마지막 오류: {str(e)}",
                    "loader_used": loader_used,
                    "fallback_attempts": fallback_attempts
                }

        except Exception as e:
            logger.error(f"ERROR PDF 처리 중 예외 발생: {e}")
            return {
                "success": False,
                "error": f"PDF 처리 중 예외: {str(e)}",
                "loader_used": loader_used,
                "fallback_attempts": fallback_attempts
            }



async def calculate_optimal_question_count(
        logger,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        PDF 내용을 분석하여 최적의 문제 수를 계산합니다.

        Args:
            content: PDF에서 추출한 텍스트 내용
            metadata: PDF 메타데이터

        Returns:
            Dict[str, Any]: 문제 수 분석 결과
        """
        try:
            # 1. 기본 텍스트 분석
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

            # 2. 문장 복잡도 분석
            complexity_scores = []
            for sentence in sentences:
                # 문장 길이 기반 복잡도
                length_score = min(len(sentence) / 100, 1.0)

                # 전문 용어 기반 복잡도
                technical_terms = len([w for w in sentence.split() if len(w) > 8])
                term_score = min(technical_terms / 5, 1.0)

                # 수식이나 코드 포함 여부
                has_math = any(c in sentence for c in ['=', '+', '-', '*', '/', '(', ')', '[', ']'])
                has_code = any(c in sentence for c in ['{', '}', ';', ':', '->', '=>'])
                special_score = 0.5 if (has_math or has_code) else 0.0

                # 최종 복잡도 점수
                complexity_scores.append((length_score + term_score + special_score) / 3)

            avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0

            # 3. 키워드/개념 추출
            words = content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 4:  # 4글자 이상 단어만 고려
                    word_freq[word] = word_freq.get(word, 0) + 1

            key_concepts = [w for w, f in word_freq.items() if f > 2][:15]  # 상위 15개 키워드

            # 4. 최적 문제 수 계산
            base_questions = len(sentences) // 4  # 4문장당 1문제 (더 집중된 문제 생성)
            complexity_factor = 1 + (avg_complexity * 0.5)  # 복잡도에 따른 가중치 (최대 1.5배)
            concept_factor = min(len(key_concepts) / 5, 1.2)  # 키워드 수에 따른 가중치 (최대 1.2배)

            recommended_questions = int(base_questions * complexity_factor * concept_factor)

            # 5. 5의 배수로 조정
            recommended_questions = round(recommended_questions / 5) * 5

            # 6. 문제 수 제한 (너무 많지 않도록)
            recommended_questions = min(max(recommended_questions, 5), 50)

            return {
                "recommended_questions": recommended_questions,
                "calculation_factors": {
                    "base_questions": base_questions,
                    "complexity_factor": complexity_factor,
                    "concept_factor": concept_factor
                },
                "content_metrics": {
                    "total_sentences": len(sentences),
                    "total_paragraphs": len(paragraphs),
                    "key_concepts": key_concepts,
                    "complexity_score": avg_complexity
                }
            }

        except Exception as e:
            logger.error(f"문제 수 계산 중 오류 발생: {e}")
            return {
                "count": 10,  # 기본값
            }

