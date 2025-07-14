upload_pdf_to_vector_db_description = """
* ***Description***:\n
    *  PDF 파일을 업로드하고 벡터 DB에 저장합니다.
* ***Request***:\n
    * file: (UploadFile, required) 업로드할 PDF 파일

* ***Response (Success)***:\n
    * HTTP 200
        * Json 타입 반환
        - {
            "success": true,
            "message": "PDF 업로드 완료",
            "file_id": "file_20250714_165131_4be995ae_b462a1",
            "filename": "4주차 강의자료.pdf",
            "chunk_count": 202,
            "recommended_questions": 5,
            "total_time": 4.437061071395874,
            "analysis_time": 0.19776296615600586,
            "extraction_time": 0.1394939422607422,
            "vector_init_time": 0,
            "vector_store_time": 4.09014892578125,
            "vector_performance": {}
        }



* ***Response (Fail)***:\n
    * HTTP 500
        - C001: 내부 서버 에러시
        - C002: 타임아웃 발생
        - C003: 필수값 누락
        - C004: 잘못된 요청
    * HTTP 400
        - D001: PDF 파일만 업로드 가능합니다.
        - D002: PDF 분석에 실패했습니다.
        - D003: PDF 추출에 실패했습니다.
"""
