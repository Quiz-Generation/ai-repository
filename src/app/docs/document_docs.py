upload_pdf_to_vector_db_description = """
* ***Description***:\n
    *  PDF 파일을 업로드하고 벡터 DB에 저장합니다.
* ***Request***:\n
    * file: (UploadFile, required) 업로드할 PDF 파일

* ***Response (Success)***:\n
    * HTTP 200
        * Json 타입 반환
            - {’message’: ’PDF 파일 업로드 및 벡터 저장 성공’, ’data’: ’벡터 저장 결과’}

* ***Response (Fail)***:\n
    * HTTP 500
        - 000001: 내부 서버 에러시
    * HTTP 400
        - 000003: 필수값 누락(000003)
"""
