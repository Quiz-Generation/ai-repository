"""
📋 Document API Schemas
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """문서 업로드 응답"""
    id: str
    filename: str
    file_size: int
    status: str
    message: str
    chunks_created: int
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터")


class DocumentSearchRequest(BaseModel):
    """문서 검색 요청"""
    query: str = Field(..., description="검색 쿼리")
    top_k: int = Field(10, description="반환할 결과 수", ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(None, description="검색 필터")


class DocumentSearchResult(BaseModel):
    """문서 검색 결과"""
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class DocumentSearchResponse(BaseModel):
    """문서 검색 응답"""
    query: str
    results: List[DocumentSearchResult]
    total_found: int
    search_time: float


class DocumentListResponse(BaseModel):
    """문서 목록 응답"""
    id: str
    filename: str
    file_size: int
    status: str
    chunks_count: int
    created_at: datetime
    updated_at: datetime


class DocumentDetailResponse(BaseModel):
    """문서 상세 응답"""
    id: str
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    status: str
    content_preview: str
    chunks_count: int
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]