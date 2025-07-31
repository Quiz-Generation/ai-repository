"""
🎯 Quiz Generation Service
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.common.utils.response import JSendResponse

from src.app.agent.quiz_generator import (
    QuizGeneratorAgent,
    QuizRequest,
    DifficultyLevel,
    QuestionType
)
from src.common.vector.connect import VectorDBService
from src.common.error import ErrorCode, JSendError

logger = logging.getLogger(__name__)

class QuizService:
    def __init__(
        self,
        logger,
        vector_db: VectorDBService,
    ):
        self.logger = logger
        self.vector_db = vector_db
        self.quiz_generator_agent = QuizGeneratorAgent
        self.difficulty_level = DifficultyLevel
        self.question_type = QuestionType
            

    async def generate_quiz_from_file(
        self,
        file_id: str,
        num_questions: int = 5,
        difficulty: str = "medium",
        question_type: str = "multiple_choice",
        custom_topic: Optional[str] = None,
        category: Optional[str] = None,
        sub_category: Optional[str] = None
    ) -> JSendResponse:
        """
        단일 파일 ID를 기반으로 문제 생성
        """
        # 1. 파일 ID로 문서 조회
        document_data = await self.get_document_by_file_id(file_id)
        if not document_data:
            self.logger.error(
                f"""
                    [ERROR] 파일 ID '{file_id}'에 해당하는 문서를 찾을 수 없음
                """
            )
            raise JSendError(
                code=ErrorCode.Quiz.GET_DOCUMENT_BY_FILE_ID_ERROR[0],
                message=ErrorCode.Quiz.GET_DOCUMENT_BY_FILE_ID_ERROR[1]
            )

        # 2. 요청 객체 생성
        try:
            difficulty_enum = self.difficulty_level(difficulty.lower())
            question_type_enum = self.question_type(question_type.lower())
        except ValueError as e:
            self.logger.error(
                f"""
                    [ERROR] 잘못된 파라미터: {str(e)}
                """
            )
            raise JSendError(
                code=ErrorCode.Quiz.GENERATE_QUIZ_ERROR[0],
                message=ErrorCode.Quiz.GENERATE_QUIZ_ERROR[1]
            )

                # 3. AI 에이전트 초기화 (캐싱 적용)
        import os
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise JSendError(
                code=ErrorCode.Common.DEFAULT_ERROR[0],
                message="OpenAI API 키가 설정되지 않았습니다. OPENAI_API_KEY 환경변수를 설정해주세요."
            )

        # 🔥 최적화: 인스턴스 캐시에서 에이전트 재사용
        if not hasattr(self, '_cached_agent'):
            self._cached_agent = QuizGeneratorAgent(openai_api_key)

        quiz_agent = self._cached_agent

        if not quiz_agent:
            self.logger.error(
                f"""
                    [ERROR] 문제 생성 에이전트 초기화 실패
                """
            )
            raise JSendError(
                code=ErrorCode.Quiz.GENERATE_QUIZ_ERROR[0],
                message=ErrorCode.Quiz.GENERATE_QUIZ_ERROR[1]
            )

        # 4. 문제 생성 요청 객체 생성
        quiz_request = QuizRequest(
            file_ids=[file_id],
            num_questions=num_questions,
            difficulty=difficulty_enum,
            question_type=question_type_enum,
            custom_topic=custom_topic,
            category=category,
            sub_category=sub_category,
            additional_instructions=[
                f"전체 {num_questions}개 문제를 생성하되, 난이도는 '{difficulty}'로 통일하세요.",
                "각 문제는 구체적인 예시나 실제 응용 사례를 포함해야 합니다.",
                "문제는 서로 중복되지 않아야 하며, 각각 독립적인 개념을 다뤄야 합니다.",
                "선택지의 경우, 명확한 정답과 그럴듯한 오답을 포함해야 합니다.",
                "문제의 난이도는 일관성을 유지해야 합니다.",
                "문제는 실제 학습 목표와 연관되어야 합니다."
            ]
        )

        # 5. AI 에이전트로 문제 생성
        logger.info(f"문제 생성 시작: {num_questions}개")
        result = await quiz_agent.generate_quiz(quiz_request, [document_data])

        if not result["success"]:
            self.logger.error(
                f"""
                    [ERROR] 문제 생성 실패: {result.get('error')}
                """
            )
            raise JSendError(
                code=ErrorCode.Quiz.GENERATE_QUIZ_ERROR[0],
                message=ErrorCode.Quiz.GENERATE_QUIZ_ERROR[1]
            )

        # 6. 결과 처리 및 품질 검사
        questions = result.get("questions", [])

        # 품질 검사 및 후처리
        processed_questions = []
        for q in questions:
            q["difficulty"] = difficulty  # 명시적으로 태깅
            if q.get("choices") and q.get("answer"):
                q = self.shuffle_choices_and_map_answer(q)
            processed_questions.append(q)

        # 문제 개수 맞추기
        processed_questions = processed_questions[:num_questions]

        # id를 1부터 다시 부여
        for idx, q in enumerate(processed_questions, 1):
            q["id"] = idx

        # 메타데이터 추가
        meta = {
            "generation_timestamp": datetime.now().isoformat(),
            "service_version": "2.0.0",
            "source_file": document_data.get("filename"),
            "file_id": file_id,
            "overall_difficulty": difficulty_enum.value,
            "generated_count": len(processed_questions),
            "quality_metrics": {
                "difficulty_consistency": await self.calculate_difficulty_consistency(processed_questions),
                "question_uniqueness": await self.calculate_question_uniqueness(processed_questions),
                "example_coverage": await self.calculate_example_coverage(processed_questions)
            }
        }

        logger.info(f"문제 생성 완료: {len(processed_questions)}개")

        return JSendResponse(
            status="success",
            data={
                "questions": processed_questions,
                "meta": meta,
                "overall_difficulty": difficulty_enum.value
            }
        )

   


    async def get_available_files(
        self,
    ) -> JSendResponse:
        """문제 생성 가능한 파일 목록 조회"""
        # 벡터 DB에서 파일 목록 조회
        files_result = await self.vector_db.get_all_documents(1000)

        if not files_result["success"]:
            self.logger.error(
                f"""
                    [ERROR] 파일 목록 조회 실패: {files_result}
                """
            )
            raise JSendError(
                code=ErrorCode.Quiz.GET_AVAILABLE_FILES_ERROR[0],
                message=ErrorCode.Quiz.GET_AVAILABLE_FILES_ERROR[1]
            )
            
        # 문제 생성에 적합한 파일들만 필터링
        suitable_files = []
        for file_info in files_result["files"]:
            # 최소 청크 수 확인 (너무 작은 파일 제외)
            if file_info.get("total_chunks", 0) >= 10:
                suitable_file = {
                    "file_id": file_info["file_id"],
                    "filename": file_info["filename"],
                    "language": file_info.get("language", "unknown"),
                    "total_chunks": file_info.get("total_chunks", 0),
                    "file_size": file_info.get("file_size", 0),
                    "upload_timestamp": file_info.get("upload_timestamp"),
                    "domain": self.identify_domain(file_info["filename"])
                }
                suitable_files.append(suitable_file)

        return JSendResponse(
            status="success",
            data={
                "message": f"문제 생성 가능한 파일 {len(suitable_files)}개 조회 완료",
                "total_files": len(suitable_files),
                "files": suitable_files,
                "supported_difficulties": [d.value for d in self.difficulty_level],
                "supported_question_types": [q.value for q in self.question_type]
            }
        )

    async def get_document_by_file_id(
        self,
        file_id: str
    ) -> Optional[Dict[str, Any]]:
        """단일 파일 ID로 문서 내용 조회 (최적화된 버전)"""
        try:


            # 🔥 최적화: file_id로 직접 필터링하여 조회
            if hasattr(self.vector_db, 'vector_db') and self.vector_db.vector_db:
                # 벡터 DB에서 해당 file_id를 가진 문서들만 조회
                all_documents = await self.vector_db.vector_db.get_all_documents(10000)

                # file_id 기준으로 필터링
                target_chunks = []
                target_file_info = None

                for doc in all_documents:
                    if doc.metadata.get("file_id") == file_id:
                        chunk_data = {
                            "id": doc.id,
                            "content": doc.content,
                            "metadata": doc.metadata
                        }
                        target_chunks.append(chunk_data)

                        # 파일 정보 추출 (첫 번째 청크에서)
                        if not target_file_info:
                            target_file_info = {
                                "file_id": file_id,
                                "filename": doc.metadata.get("filename", "Unknown"),
                                "language": doc.metadata.get("language", "unknown"),
                                "file_size": doc.metadata.get("file_size", 0),
                                "pdf_loader": doc.metadata.get("pdf_loader", "unknown"),
                                "upload_timestamp": doc.metadata.get("upload_timestamp"),
                                "total_chunks": len([d for d in all_documents if d.metadata.get("file_id") == file_id])
                            }

                if not target_chunks:
                    logger.warning(f"WARNING 지정된 파일 ID를 찾을 수 없음: {file_id}")
                    return None

                # 청크들을 하나의 문서로 합치기 (정렬 후)
                target_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
                combined_content = ""
                for chunk in target_chunks:
                    combined_content += chunk.get("content", "") + "\n\n"

                # 문서 정보 구성
                document = {
                    "file_id": file_id,
                    "filename": target_file_info["filename"],
                    "content": combined_content.strip(),
                    "language": target_file_info["language"],
                    "file_size": target_file_info["file_size"],
                    "total_chunks": target_file_info["total_chunks"],
                    "pdf_loader": target_file_info["pdf_loader"],
                    "upload_timestamp": target_file_info["upload_timestamp"],
                    "domain": self.identify_domain(target_file_info["filename"])
                }


                return document
            else:
                # 기존 방식으로 fallback
                all_docs_result = await self.vector_db.get_all_documents(10000)
                if not all_docs_result["success"]:
                    logger.error("ERROR 전체 문서 조회 실패")
                    return None

                target_file = None
                for file_info in all_docs_result["files"]:
                    if file_info["file_id"] == file_id:
                        target_file = file_info
                        break

                if not target_file:
                    logger.warning(f"WARNING 지정된 파일 ID를 찾을 수 없음: {file_id}")
                    return None

                file_chunks = await self.get_file_chunks(self.vector_db, file_id)
                combined_content = ""
                for chunk in file_chunks:
                    combined_content += chunk.get("content", "") + "\n\n"

                document = {
                    "file_id": file_id,
                    "filename": target_file["filename"],
                    "content": combined_content.strip(),
                    "language": target_file.get("language", "unknown"),
                    "file_size": target_file.get("file_size", 0),
                    "total_chunks": target_file.get("total_chunks", 0),
                    "pdf_loader": target_file.get("pdf_loader", "unknown"),
                    "upload_timestamp": target_file.get("upload_timestamp"),
                    "domain": self.identify_domain(target_file["filename"])
                }


                return document

        except Exception as e:
            logger.error(f"ERROR 문서 조회 실패: {e}")
            return None


    async def get_file_chunks(
        self,
        vector_db: VectorDBService,
        file_id: str
    ) -> List[Dict[str, Any]]:
        """특정 파일의 모든 청크 조회"""
        try:
            # 벡터 DB에서 해당 file_id를 가진 모든 문서 조회
            all_documents = await vector_db.vector_db.get_all_documents(10000)

            # file_id 기준으로 필터링
            file_chunks = []
            for doc in all_documents:
                if doc.metadata.get("file_id") == file_id:
                    chunk_data = {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                    file_chunks.append(chunk_data)

            # chunk_index 순서로 정렬 (가능한 경우)
            file_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))


            return file_chunks

        except Exception as e:
            logger.error(f"ERROR 파일 청크 조회 실패: {e}")
            return []


    async def calculate_difficulty_consistency(
        self,
        questions: List[Dict[str, Any]]
    ) -> float:
        """문제 난이도 일관성 계산"""
        if not questions:
            return 0.0

        # 난이도 분포 계산
        difficulty_counts = {}
        for q in questions:
            diff = q.get("difficulty", "medium")
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        # 가장 많은 난이도의 비율 계산
        max_count = max(difficulty_counts.values())
        return max_count / len(questions)


    async def calculate_question_uniqueness(
        self,
        questions: List[Dict[str, Any]]
    ) -> float:
        """문제 중복성 계산"""
        if not questions:
            return 0.0

        # 문제 내용의 유사도 계산
        unique_questions = set()
        for q in questions:
            # 문제 내용을 정규화하여 저장
            normalized = q.get("question", "").lower().strip()
            unique_questions.add(normalized)

        return len(unique_questions) / len(questions)


    async def calculate_example_coverage(
        self,
        questions: List[Dict[str, Any]]
    ) -> float:
        """예시 포함 비율 계산"""
        if not questions:
            return 0.0

        example_count = 0
        for q in questions:
            # 예시나 실제 사례가 포함된 문제 수 계산
            question_text = q.get("question", "").lower()
            if any(keyword in question_text for keyword in ["예를 들어", "예시", "사례", "for example", "such as"]):
                example_count += 1

        return example_count / len(questions)


    def identify_domain(
            self, 
            filename: str
    ) -> str:
        """파일명을 기반으로 도메인 식별"""
        filename_lower = filename.lower()

        if "aws" in filename_lower or "cloud" in filename_lower:
            return "클라우드/AWS"
        elif "dynamic" in filename_lower or "programming" in filename_lower or "algorithm" in filename_lower:
            return "알고리즘/프로그래밍"
        elif "심리" in filename_lower or "psychology" in filename_lower:
            return "심리학"
        elif "기술" in filename_lower or "tech" in filename_lower:
            return "기술"
        elif "강의" in filename_lower or "lecture" in filename_lower:
            return "교육/강의"
        else:
            return "기타"


    async def shuffle_choices_and_map_answer(
        self,
        question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """선택지 섞기 및 정답 매핑"""
        import random

        if not question.get("choices") or not question.get("answer"):
            return question

        choices = question["choices"]
        correct_answer = question["answer"]

        # 선택지와 정답을 함께 섞기
        choice_answer_pairs = list(zip(choices, [i for i in range(len(choices))]))
        random.shuffle(choice_answer_pairs)

        # 새로운 선택지와 정답 인덱스
        new_choices = [pair[0] for pair in choice_answer_pairs]
        correct_index = None
        for i, pair in enumerate(choice_answer_pairs):
            if pair[1] == correct_answer:
                correct_index = i
                break

        question["choices"] = new_choices
        question["answer"] = correct_index if correct_index is not None else correct_answer

        return question
