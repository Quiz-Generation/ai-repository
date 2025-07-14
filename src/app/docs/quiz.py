generate_quiz_description = """
* ***Description***:\n
    * 벡터 DB에 저장된 PDF 파일을 기반으로 퀴즈를 생성합니다.
* ***Request***:\n
    * file_id: (str, required) 퀴즈를 생성할 PDF 파일의 ID
    * num_questions: (int, optional) 생성할 문제 개수 (기본값: 10)
    * difficulty: (str, optional) 문제 난이도 (easy, medium, hard)
    * question_type: (str, optional) 문제 유형 (multiple_choice, true_false, short_answer, essay, fill_blank)
    * custom_topic: (str, optional) 퀴즈 주제
    * category: (str, optional) 대분류(예: 컴퓨터 공학)
    * sub_category: (str, optional) 소분류(예: 데이터베이스)

* ***Response (Success)***:\n
    * HTTP 200
        * Json 타입 반환
            - {’message’: ’퀴즈 생성 성공’, ’data’: ’퀴즈 결과’}

* ***Response (Fail)***:\n
    * HTTP 500
        - 000001: 내부 서버 에러시
    * HTTP 400
        - 000003: 필수값 누락(000003)
"""

available_files_description = """
* ***Description***:\n
    * 벡터 DB에 저장된 PDF 파일을 기반으로 퀴즈를 생성합니다.
* ***Request***:\n
    * file_id: (str, required) 퀴즈를 생성할 PDF 파일의 ID
    * num_questions: (int, optional) 생성할 문제 개수 (기본값: 10)
    * difficulty: (str, optional) 문제 난이도 (easy, medium, hard)
    * question_type: (str, optional) 문제 유형 (multiple_choice, true_false, short_answer, essay, fill_blank)
"""