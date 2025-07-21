from pydantic import BaseModel

class PDFAnalysisResult(BaseModel):
    """PDF 분석 결과"""
    language: str  # 'korean', 'english', 'mixed', 'unknown'
    has_tables: bool
    has_images: bool
    complexity: str  # 'simple', 'medium', 'complex'
    file_size: int
    estimated_pages: int
    text_density: str  # 'low', 'medium', 'high'
    font_complexity: str  # 'simple', 'complex'
    recommended_loader: str
