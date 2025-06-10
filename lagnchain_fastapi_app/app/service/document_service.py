"""
🧠 Document Service
"""
import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import UploadFile

from ..schemas.document_schema import (
    DocumentUploadResponse,
    DocumentSearchResponse,
    DocumentSearchResult,
    DocumentListResponse,
    DocumentDetailResponse
)
from ..models.document_model import Document, DocumentChunk
from ..repository.document_repository import DocumentRepository
from ..repository.vector_repository import VectorRepository
from ..helper.pdf_helper import PDFHelper
from ..helper.text_helper import TextHelper
from ..helper.pdf_loader_helper import PDFLoaderHelper, PDFAnalysisResult
from ..core.pdf_loader.factory import PDFLoaderFactory
from ..core.config import settings

logger = logging.getLogger(__name__)


class DocumentService:
    """문서 처리 메인 서비스"""

    def __init__(self):
        self.document_repo = DocumentRepository()
        self.vector_repo = VectorRepository()
        self.pdf_helper = PDFHelper()
        self.text_helper = TextHelper()

    async def upload_document(self, file: UploadFile) -> DocumentUploadResponse:
        """문서 업로드 및 처리 (동적 PDF 로더 사용)"""
        try:
            logger.info(f"📄 문서 업로드 시작: {file.filename}")

            # 1. 파일 검증
            if not self._validate_file(file):
                return DocumentUploadResponse(
                    id="",
                    filename=file.filename or "unknown.pdf",
                    file_size=file.size or 0,
                    status="failed",
                    message="파일 검증 실패",
                    chunks_created=0,
                    created_at=datetime.now(),
                    metadata={
                    "loader_used": "pymupdf",
                    "analysis_result": {}
                    }
                )

            # 2. 동적 PDF 로더 선택
            optimal_loader_type = await self._select_optimal_pdf_loader(file)
            logger.info(f"🎯 선택된 PDF 로더: {optimal_loader_type}")

            # 3. 선택된 로더로 PDF 처리
            pdf_content = await self._extract_pdf_with_selected_loader(file, optimal_loader_type)

            # 4. 파일 저장
            saved_path = await self._save_uploaded_file(file)

            # 5. 텍스트 청킹
            chunks = await self._create_text_chunks(pdf_content.text)

            # 6. 벡터화 및 저장 (TODO: 실제 구현)
            # vector_ids = await self._vectorize_and_store(chunks)

            return DocumentUploadResponse(
                id=f"doc_{int(time.time())}",
                filename=file.filename or "unknown.pdf",
                file_size=file.size or 0,
                status="completed",
                message=f"✅ {optimal_loader_type} 로더로 성공적으로 처리됨",
                chunks_created=len(chunks),
                created_at=datetime.now(),
                metadata={
                    "loader_used": optimal_loader_type,
                    "analysis_result": pdf_content.metadata
                }
            )

        except Exception as e:
            logger.error(f"❌ 문서 업로드 실패: {e}")
            return DocumentUploadResponse(
                id="",
                filename=file.filename or "unknown.pdf",
                file_size=file.size or 0,
                status="failed",
                message=f"처리 실패: {str(e)}",
                chunks_created=0,
                created_at=datetime.now(),
                metadata={
                    "loader_used": optimal_loader_type,
                    "analysis_result": pdf_content.metadata
                }
            )

    async def _select_optimal_pdf_loader(self, file: UploadFile) -> str:
        """동적으로 최적의 PDF 로더 선택 (핵심 비즈니스 로직)"""
        try:
            logger.info("🔍 PDF 파일 특성 분석 중...")

            # Helper에서 세부 분석 로직 호출
            analysis_result = await PDFLoaderHelper.analyze_pdf_characteristics(file)

            logger.info(f"""
            📊 PDF 분석 결과:
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
            logger.error(f"❌ PDF 로더 선택 실패: {e}")
            logger.info("🔄 기본 로더(PyMuPDF) 사용")
            return "pymupdf"

    async def _extract_pdf_with_selected_loader(self, file: UploadFile, loader_type: str):
        """선택된 로더로 PDF 텍스트 추출"""
        try:
            # 팩토리에서 로더 생성
            pdf_loader = PDFLoaderFactory.create(loader_type)

            # 파일 유효성 검증
            if not pdf_loader.validate_file(file):
                raise ValueError(f"파일 유효성 검사 실패: {file.filename}")

            # 텍스트 추출
            pdf_content = await pdf_loader.extract_text_from_file(file)

            logger.info(f"✅ {loader_type} 로더로 텍스트 추출 완료")
            return pdf_content

        except Exception as e:
            logger.error(f"❌ PDF 추출 실패 ({loader_type}): {e}")

            # 실패 시 fallback 로더 시도
            if loader_type != "pymupdf":
                logger.info("🔄 PyMuPDF 로더로 재시도")
                fallback_loader = PDFLoaderFactory.create("pymupdf")
                return await fallback_loader.extract_text_from_file(file)
            else:
                raise

    async def _create_text_chunks(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        # TextHelper의 단순 문자열 분할 사용
        chunks = self.text_helper.split_text_simple(
            text,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        logger.info(f"📝 텍스트 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks

    def _validate_file(self, file: UploadFile) -> bool:
        """파일 유효성 검증"""
        if not file.filename:
            return False

        if not file.filename.lower().endswith('.pdf'):
            return False

        if file.size and file.size > settings.MAX_FILE_SIZE:
            return False

        return True

    async def _save_uploaded_file(self, file: UploadFile) -> str:
        """업로드된 파일 저장"""
        # TODO: 실제 파일 저장 로직
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        save_path = os.path.join(settings.UPLOAD_DIR, filename)

        # 디렉토리 생성
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        logger.info(f"💾 파일 저장: {save_path}")
        return save_path

    async def get_loader_selection_info(self) -> Dict[str, Any]:
        """PDF 로더 선택 규칙 정보 반환"""
        return {
            "supported_loaders": PDFLoaderFactory.get_supported_loaders(),
            "priority_order": PDFLoaderFactory.get_priority_order(),
            "selection_rules": PDFLoaderHelper.get_loader_selection_rules(),
            "all_loaders_info": PDFLoaderFactory.get_all_loaders_info()
        }

    async def search_documents(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> DocumentSearchResponse:
        """문서 검색"""
        start_time = time.time()

        # TODO: 실제 벡터 검색 구현
        results = []

        search_time = time.time() - start_time

        return DocumentSearchResponse(
            query=query,
            results=results,
            total_found=len(results),
            search_time=search_time
        )

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 10
    ) -> List[DocumentListResponse]:
        """문서 목록 조회"""
        # TODO: 실제 구현
        return []

    async def get_document_detail(self, document_id: str) -> Optional[DocumentDetailResponse]:
        """문서 상세 정보 조회"""
        # TODO: 실제 구현
        return None

    async def delete_document(self, document_id: str) -> bool:
        """문서 삭제"""
        # TODO: 실제 구현
        return True