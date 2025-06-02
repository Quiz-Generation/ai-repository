from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ContentType(Enum):
    """문서 내용 유형"""
    KOREAN = "korean"           # 한글 문서
    ENGLISH = "english"         # 영문 문서
    MIXED = "mixed"            # 한영 혼합
    TECHNICAL = "technical"     # 기술 문서
    UNKNOWN = "unknown"        # 알 수 없음


class Priority(Enum):
    """우선순위 유형"""
    SPEED = "speed"           # 속도 우선
    QUALITY = "quality"       # 품질 우선
    BALANCED = "balanced"     # 균형 잡힌




@dataclass
class ExtractionResult:
    """PDF 추출 결과"""
    success: bool
    text: str = ""
    extractor_used: str = ""
    file_size_mb: float = 0.0
    content_type: str = ""
    priority: str = ""
    selection_time: float = 0.0
    extraction_time: float = 0.0
    total_time: float = 0.0
    text_length: int = 0
    speed_mbps: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
