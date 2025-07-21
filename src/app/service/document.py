"""
🧠 Document Service
"""
import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import UploadFile

from src.app.helper.pdf_loader_helper import PDFLoaderHelper, PDFAnalysisResult
from src.app.helper.text_helper import TextHelper
from src.app.core.pdf_loader.factory import PDFLoaderFactory

logger = logging.getLogger(__name__)

# 🎯 하드코딩된 설정값들 (config 의존성 제거)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
UPLOAD_DIR = "data/uploads"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class DocumentService:
    """문서 처리 메인 서비스"""

    def __init__(self):
        self.text_helper = TextHelper()

    async def process_pdf_with_dynamic_selection(
        self,
        file: UploadFile,
        recommended_loader: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🚀 벡터 DB 통합용 PDF 처리 메서드
        - 동적 로더 선택 및 텍스트 추출
        - 폴백 메커니즘 포함
        """
        loader_used = recommended_loader or "pymupdf"
        fallback_attempts = 0

        try:
            logger.info(f"STEP_PDF PDF 처리 시작: {file.filename} (로더: {loader_used})")

            # 1. 파일 검증
            if not self._validate_file(file):
                return {
                    "success": False,
                    "error": "파일 검증 실패",
                    "loader_used": loader_used,
                    "fallback_attempts": fallback_attempts
                }

            # 🔥 파일 포인터를 처음으로 리셋 (중요!)
            await file.seek(0)

            # 2. 선택된 로더로 PDF 처리 시도
            try:
                pdf_content = await self._extract_pdf_with_selected_loader(file, loader_used)

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

                        pdf_content = await self._extract_pdf_with_selected_loader(file, fallback_loader)

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

    async def upload_document(self, file: UploadFile) -> Dict[str, Any]:
        """문서 업로드 및 처리 (동적 PDF 로더 사용)"""
        # 초기 변수 설정
        optimal_loader_type = "pymupdf"  # 기본값
        pdf_content = None

        try:
            logger.info(f"STEP1 문서 업로드 시작: {file.filename}")

            # 1. 파일 검증
            if not self._validate_file(file):
                return {
                    "id": "",
                    "filename": file.filename or "unknown.pdf",
                    "file_size": file.size or 0,
                    "status": "failed",
                    "message": "파일 검증 실패",
                    "chunks_created": 0,
                    "created_at": datetime.now(),
                    "metadata": {
                        "loader_used": optimal_loader_type,
                        "analysis_result": {}
                    }
                }

            # 2. 동적 PDF 로더 선택
            optimal_loader_type = await self._select_optimal_pdf_loader(file)
            logger.info(f"STEP2 선택된 PDF 로더: {optimal_loader_type}")

            # 3. 선택된 로더로 PDF 처리
            pdf_content = await self._extract_pdf_with_selected_loader(file, optimal_loader_type)

            # 4. 파일 저장
            saved_path = await self._save_uploaded_file(file)

            # 5. 텍스트 청킹
            chunks = await self._create_text_chunks(pdf_content.text)
            logger.info(f"STEP5 청킹 완료: {len(chunks)}개 청크 생성됨")

            return {
                "id": f"doc_{int(time.time())}",
                "filename": file.filename or "unknown.pdf",
                "file_size": file.size or 0,
                "status": "completed",
                "message": f"SUCCESS {optimal_loader_type} 로더로 성공적으로 처리됨",
                "chunks_created": len(chunks),
                "created_at": datetime.now(),
                "metadata": {
                    "loader_used": optimal_loader_type,
                    "analysis_result": pdf_content.metadata if pdf_content else {}
                }
            }

        except Exception as e:
            logger.error(f"ERROR 문서 업로드 실패: {e}")
            return {
                "id": "",
                "filename": file.filename or "unknown.pdf",
                "file_size": file.size or 0,
                "status": "failed",
                "message": f"처리 실패: {str(e)}",
                "chunks_created": 0,
                "created_at": datetime.now(),
                "metadata": {
                    "loader_used": optimal_loader_type,
                    "analysis_result": pdf_content.metadata if pdf_content else {}
                }
            }

    async def _select_optimal_pdf_loader(self, file: UploadFile) -> str:
        """동적으로 최적의 PDF 로더 선택 (핵심 비즈니스 로직)"""
        try:
            logger.info("STEP3 PDF 파일 특성 분석 중...")

            # Helper에서 세부 분석 로직 호출
            analysis_result = await PDFLoaderHelper.analyze_pdf_characteristics(file)

            logger.info(f"""
            STEP3 PDF 분석 결과:
            - 언어: {analysis_result.language}
            - 테이블 존재: {analysis_result.has_tables}
            - 이미지 존재: {analysis_result.has_images}
            - 복잡도: {analysis_result.complexity}
            - 파일 크기: {analysis_result.file_size:,} bytes
            - 예상 페이지: {analysis_result.estimated_pages}
            - 텍스트 밀도: {analysis_result.text_density}
            - 폰트 복잡도: {analysis_result.font_complexity}
            - 추천 로더: {analysis_result.recommended_loader}
            """)

            return analysis_result.recommended_loader

        except Exception as e:
            logger.error(f"ERROR PDF 로더 선택 실패: {e}")
            logger.info("FALLBACK 기본 로더(PyMuPDF) 사용")
            return "pymupdf"

    async def _extract_pdf_with_selected_loader(self, file: UploadFile, loader_type: str):
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

    async def _create_text_chunks(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        # TextHelper의 인스턴스 메서드 사용 (기존 로직 유지)
        chunks = self.text_helper.split_text_simple(
            text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        logger.info(f"STEP5 텍스트 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks

    def _validate_file(self, file: UploadFile) -> bool:
        """파일 유효성 검증"""
        if not file.filename:
            return False

        if not file.filename.lower().endswith('.pdf'):
            return False

        if file.size and file.size > MAX_FILE_SIZE:
            return False

        return True

    async def _save_uploaded_file(self, file: UploadFile) -> str:
        """업로드된 파일 저장"""
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        save_path = os.path.join(UPLOAD_DIR, filename)

        # 디렉토리 생성
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        logger.info(f"STEP4a 파일 저장: {save_path}")
        return save_path

    async def get_loader_selection_info(self) -> Dict[str, Any]:
        """PDF 로더 선택 규칙 정보 반환"""
        return {
            "supported_loaders": PDFLoaderFactory.get_supported_loaders(),
            "priority_order": PDFLoaderFactory.get_priority_order(),
            "selection_rules": PDFLoaderHelper.get_loader_selection_rules(),
            "capabilities": {
                "pymupdf": "고성능, 빠른 처리, 기본 추천",
                "pdfplumber": "테이블 특화, 레이아웃 보존",
                "pypdf": "경량, 메모리 효율적",
                "pdfminer": "정확도 높음, 복잡한 PDF"
            }
        }

    async def search_documents(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """문서 검색"""
        start_time = time.time()

        # TODO: 실제 벡터 검색 구현
        results = []

        search_time = time.time() - start_time

        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "search_time": search_time
        }

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """문서 목록 조회"""
        # TODO: 실제 구현
        return []

    async def get_document_detail(self, document_id: str) -> Optional[Dict[str, Any]]:
        """문서 상세 정보 조회"""
        # TODO: 실제 구현
        return None

    async def delete_document(self, document_id: str) -> bool:
        """문서 삭제"""
        # TODO: 실제 구현
        return True

    async def calculate_optimal_question_count(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
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
                "count": recommended_questions,
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
                "error": str(e)
            }