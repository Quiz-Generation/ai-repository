"""
🔍 PDF Loader Selection Helper
"""
import re
from fastapi import UploadFile

from src.app.document.model import PDFAnalysisResult


class DocumentLoader:
    def __init__(self, logger):
        self.logger = logger

    async def detect_language_from_filename(
        self,
        filename: str
    ) -> str:
        """파일명에서 언어 감지"""
        self.logger.info(
            f"""
                [PDF 특성 분석 시작]
                PDF 파일명: {filename}
            """
        )
        if not filename:
            return "unknown"

        # 1. 파일명에 한글 문자가 있는지 직접 체크
        korean_chars = len(re.findall(r'[가-힣]', filename))
        self.logger.info(
            f"""
                STEP3-1b 파일명에서 한글 문자 {korean_chars}개 발견
            """
        )

        if korean_chars > 0:
            self.logger.info(
                f"""
                    STEP3-1b 한글 문자 발견으로 korean 반환
                """
            )
            return "korean"

        # 2. 한글 관련 키워드 체크
        filename_lower = filename.lower()
        korean_keywords = ['한글', '한국', 'korean', 'kr', '보고서', '문서', '계약서', '제안서', '강의', '자료']
        english_keywords = ['english', 'en', 'report', 'document', 'contract', 'proposal', 'lecture', 'material']

        korean_score = sum(1 for keyword in korean_keywords if keyword in filename_lower)
        english_score = sum(1 for keyword in english_keywords if keyword in filename_lower)

        self.logger.info(
            f"""
                STEP3-1b 키워드 점수: korean={korean_score}, english={english_score}
            """
        )

        if korean_score > 0:
            self.logger.info(
                f"""
                    STEP3-1b 한글 키워드 발견으로 korean 반환
                """
            )
            return "korean"
        elif english_score > 0:
            self.logger.info(
                f"""
                    STEP3-1b 영어 키워드 발견으로 english 반환
                """
            )
            return "english"
        else:
            self.logger.info(
                f"""
                    STEP3-1b 키워드 없음으로 unknown 반환
                """
            )
            return "unknown"


    async def detect_language_from_content(
        self,
        file: UploadFile
    ) -> str:
        """PDF 텍스트 내용에서 언어 감지"""
        self.logger.info(
            f"""
                STEP3-1c 텍스트 내용에서 언어 감지 시작
            """
        )
        try:
            # 파일 내용 읽기
            file_content = await file.read()

            # 파일 포인터 원위치
            try:
                await file.seek(0)
            except:
                pass

            if not file_content:
                self.logger.warning(
                    f"""
                        STEP3-1c 파일 내용이 비어있습니다
                    """
                )
                return "unknown"

            # PyMuPDF로 빠른 텍스트 추출
            try:
                import fitz
                doc = fitz.open(stream=file_content, filetype="pdf")

                if len(doc) == 0:
                    self.logger.warning(
                        f"""
                            STEP3-1c PDF 페이지가 없습니다
                        """
                    )
                    return "unknown"

                # 첫 페이지 텍스트 추출
                page = doc.load_page(0)
                sample_text = page.get_text()[:1000]  # 1000자만
                doc.close()

                if not sample_text.strip():
                    self.logger.warning(
                        f"""
                            STEP3-1c 추출된 텍스트가 비어있습니다
                        """
                    )
                    return "unknown"

                # 한글/영어 문자 카운트
                korean_chars = len(re.findall(r'[가-힣]', sample_text))
                english_chars = len(re.findall(r'[a-zA-Z]', sample_text))

                self.logger.info(f"STEP3-1d 텍스트 분석: 한글={korean_chars}자, 영어={english_chars}자")

                if korean_chars > 20:
                    return "korean"
                if english_chars > 50 and korean_chars < 10:
                    return "english"
                if korean_chars > 10 and english_chars > 10:
                    return "mixed"
                return "unknown"

            except Exception as e:
                self.logger.warning(
                    f"""
                        STEP3-1c 텍스트 추출 실패: {e}
                    """
                )
                return "unknown"

        except Exception as e:
            self.logger.error(f"ERROR 언어 감지 실패: {e}")
            return "unknown"


    async def combine_language_results(
        self,
        filename_lang: str,
        text_lang: str
    ) -> str:
        """파일명과 텍스트 분석 결과 종합"""
        self.logger.info(f"STEP3-1c 언어 결합: filename={filename_lang}, text={text_lang}")

        # 파일명에서 korean이 감지되면 우선시 (한글 파일명은 확실함)
        if filename_lang == "korean":
            return "korean"
        # 텍스트 분석 결과를 우선시
        elif text_lang in ["korean", "english", "mixed"]:
            return text_lang
        elif filename_lang in ["english", "mixed"]:
            return filename_lang
        else:
            # 둘 다 unknown이면 기본값으로 english 설정
            return "english"


    async def estimate_complexity_from_size(
        self,
        file_size: int
    ) -> str:
        """파일 크기로 복잡도 추정"""
        self.logger.info(
            f"""
                STEP3-1e 파일 크기로 복잡도 추정 시작: {file_size}
            """
        )
        if file_size < 1024 * 1024:  # 1MB 미만
            return "simple"
        elif file_size < 50 * 1024 * 1024:  # 50MB 미만
            return "medium"
        else:
            return "complex"


    async def estimate_tables_from_filename(
        self,
        filename: str
    ) -> bool:
        """파일명에서 테이블 존재 추정"""
        self.logger.info(
            f"""
                STEP3-1f 파일명에서 테이블 존재 추정 시작: {filename}
            """
        )
        table_keywords = ['table', '표', 'chart', '차트', 'data', '데이터', 'excel', 'sheet']
        filename_lower = filename.lower()
        return any(keyword in filename_lower for keyword in table_keywords)


    async def estimate_images_from_size(
        self,
        file_size: int
    ) -> bool:
        """파일 크기로 이미지 존재 추정"""
        self.logger.info(
            f"""
                STEP3-1g 파일 크기로 이미지 존재 추정 시작: {file_size}
            """
        )
        # 5MB 이상이면 이미지가 있을 가능성 높음
        return file_size > 5 * 1024 * 1024


    async def estimate_text_density(
        self,
        file_size: int,
        pages: int
    ) -> str:
        """텍스트 밀도 추정"""
        self.logger.info(
            f"""
                STEP3-1h 텍스트 밀도 추정 시작: {file_size}, {pages}
            """
        )
        if pages == 0:
            return "medium"

        size_per_page = file_size / pages

        if size_per_page < 50 * 1024:  # 50KB per page
            return "low"
        elif size_per_page < 200 * 1024:  # 200KB per page
            return "medium"
        else:
            return "high"



    async def estimate_font_complexity(
        self,
        language: str,
        complexity: str
    ) -> str:
        """폰트 복잡도 추정"""
        self.logger.info(
            f"""
                STEP3-1i 폰트 복잡도 추정 시작: {language}, {complexity}
            """
        )
        if language == "korean" or language == "mixed":
            return "complex"
        elif complexity == "complex":
            return "complex"
        else:
            return "simple"

    async def recommend_loader(
        self,
        analysis: PDFAnalysisResult
    ) -> str:
        """분석 결과를 바탕으로 최적 로더 추천"""
        self.logger.info(
            f"""
                STEP3-1j 분석 결과를 바탕으로 최적 로더 추천 시작: {analysis}
            """
        )
        # 1. 한글 문서의 경우 PDFMiner 우선 (정확도)
        if analysis.language == "korean":
            if analysis.has_tables:
                return "pdfplumber"  # 한글 + 테이블
            else:
                return "pdfminer"    # 한글 텍스트

        # 2. 테이블이 많은 경우 PDFPlumber
        if analysis.has_tables:
            return "pdfplumber"

        # 3. 복잡한 레이아웃인 경우 PDFMiner
        if analysis.complexity == "complex" or analysis.font_complexity == "complex":
            return "pdfminer"

        # 4. 큰 파일인 경우 PyMuPDF (성능)
        if analysis.file_size > 20 * 1024 * 1024:  # 20MB 이상
            return "pymupdf"

        # 5. 작은 파일인 경우 PyPDF (가벼움)
        if analysis.file_size < 1024 * 1024:  # 1MB 이하
            return "pypdf"

        # 6. 기본값: PyMuPDF (최고 성능)
        return "pymupdf"

    async def analyze_pdf_characteristics(
        self,
        file: UploadFile
    ) -> PDFAnalysisResult:
        """PDF 파일 특성 분석"""
        try:
            file_size = file.size or 0
            estimated_pages = max(1, file_size // (50 * 1024))  # 대략적인 페이지 수 추정

            # 파일명 기반 1차 언어 추정
            filename = file.filename or ""
            filename_language = await self.detect_language_from_filename(
                filename=filename
            )

            # 실제 텍스트 기반 언어 감지
            text_language = await self.detect_language_from_content(
                file=file
            )

            # 파일명과 텍스트 분석 결과 종합
            language = await self.combine_language_results(
                filename_lang=filename_language,
                text_lang=text_language
            )
            self.logger.info(
                f"""
                    STEP3-1 언어 감지 완료: 파일명={filename_language}, 텍스트={text_language}, 최종={language}
                """
            )

            # 파일 크기 기반 복잡도 추정
            complexity = await self.estimate_complexity_from_size(
                file_size=file_size
            )

            # 테이블/이미지 존재 추정 (파일명/크기 기반)
            has_tables = await self.estimate_tables_from_filename(
                filename=filename
            )
            has_images = await self.estimate_images_from_size(
                file_size=file_size
            )

            # 텍스트 밀도 추정
            text_density = await self.estimate_text_density(
                file_size=file_size,
                pages=estimated_pages
            )

            # 폰트 복잡도 추정
            font_complexity = await self.estimate_font_complexity(
                language=language,
                complexity=complexity
            )

            analysis_result = PDFAnalysisResult(
                language=language,
                has_tables=has_tables,
                has_images=has_images,
                complexity=complexity,
                file_size=file_size,
                estimated_pages=estimated_pages,
                text_density=text_density,
                font_complexity=font_complexity,
                recommended_loader=""  # 나중에 설정
            )

            # 최적 로더 추천
            recommended_loader = await self.recommend_loader(
                analysis=analysis_result
            )
            analysis_result.recommended_loader = recommended_loader

            self.logger.info(
                f"""
                    STEP3-2 PDF 분석 완료: {filename} -> {recommended_loader}
                """
            )
            return analysis_result

        except Exception as e:
            self.logger.error(f"ERROR PDF 분석 실패: {e}")
            # 기본값 반환
            return PDFAnalysisResult(
                language="unknown",
                has_tables=False,
                has_images=False,
                complexity="simple",
                file_size=file_size,
                estimated_pages=1,
                text_density="medium",
                font_complexity="simple",
                recommended_loader="pymupdf"
            )