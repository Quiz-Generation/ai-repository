"""
📄 Document Data Models
"""
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
import uuid


@dataclass
class Document:
    """문서 모델"""
    id: str
    filename: str
    original_filename: str
    file_path: str
    content: str
    file_size: int
    mime_type: str
    status: str  # 'processing', 'completed', 'failed'
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class DocumentChunk:
    """문서 청크 모델"""
    id: str
    document_id: str
    chunk_index: int
    content: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any]
    created_at: datetime

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.document_id}_chunk_{self.chunk_index}"

