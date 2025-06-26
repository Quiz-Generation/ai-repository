"""
🎯 Quiz Generation Service
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..agent.quiz_generator import (
    QuizGeneratorAgent,
    QuizRequest,
    DifficultyLevel,
    QuestionType
)
from ..service.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)


class QuizService:
    """문제 생성 서비스"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        초기화
        Args:
            openai_api_key: OpenAI API 키
        """
        self.vector_service = VectorDBService()
        self.quiz_agent = QuizGeneratorAgent(openai_api_key)

    async def generate_quiz_from_file(
        self,
        file_id: str,
        num_questions: int = 5,
        difficulty: str = "medium",
        question_type: str = "multiple_choice",
        custom_topic: Optional[str] = None,
        category: Optional[str] = None,
        sub_category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        단일 파일 ID를 기반으로 문제 생성
        Args:
            file_id: 대상 파일 ID (단일)
            num_questions: 생성할 문제 수
            difficulty: 난이도 (easy/medium/hard)
            question_type: 문제 유형 (multiple_choice/true_false/short_answer/essay/fill_blank)
            custom_topic: 특정 주제 (선택사항)
            category: 대분류 (선택사항, 예: 컴퓨터 공학)
            sub_category: 소분류 (선택사항, 예: 데이터베이스)
        Returns:
            생성된 문제 데이터
        """
        try:
            logger.info(f"🚀 문제 생성 서비스 시작: {file_id}")

            # 1. 파일 ID로 문서 조회
            document_data = await self._get_document_by_file_id(file_id)
            if not document_data:
                return {
                    "success": False,
                    "error": f"파일 ID '{file_id}'에 해당하는 문서를 찾을 수 없습니다",
                    "file_id": file_id
                }

            # 2. 요청 객체 생성
            try:
                difficulty_enum = DifficultyLevel(difficulty.lower())
                question_type_enum = QuestionType(question_type.lower())
            except ValueError as e:
                return {
                    "success": False,
                    "error": f"잘못된 파라미터: {str(e)}",
                    "valid_difficulty": [d.value for d in DifficultyLevel],
                    "valid_question_types": [q.value for q in QuestionType]
                }

            # 3. 문제 생성 요청 구성
            quiz_request = QuizRequest(
                file_ids=[file_id],
                num_questions=num_questions,
                difficulty=difficulty_enum,
                question_type=question_type_enum,
                custom_topic=custom_topic,
                category=category,
                sub_category=sub_category,
                additional_instructions=[
                    "각 문제는 구체적인 예시나 실제 응용 사례를 포함해야 합니다.",
                    "문제는 서로 중복되지 않아야 하며, 각각 독립적인 개념을 다뤄야 합니다.",
                    "선택지의 경우, 명확한 정답과 그럴듯한 오답을 포함해야 합니다.",
                    "문제의 난이도는 일관성을 유지해야 합니다.",
                    "문제는 실제 학습 목표와 연관되어야 합니다."
                ]
            )

            # 4. AI 에이전트로 문제 생성
            logger.info("STEP_AGENT AI 에이전트 문제 생성 시작")
            result = await self.quiz_agent.generate_quiz(quiz_request, [document_data])

            # 5. 결과 후처리
            if result["success"]:
                # 메타데이터 추가
                result["meta"]["generation_timestamp"] = datetime.now().isoformat()
                result["meta"]["service_version"] = "1.0.0"
                result["meta"]["source_file"] = document_data.get("filename")
                result["meta"]["file_id"] = file_id
                result["meta"]["quality_metrics"] = {
                    "difficulty_consistency": self._calculate_difficulty_consistency(result["questions"]),
                    "question_uniqueness": self._calculate_question_uniqueness(result["questions"]),
                    "example_coverage": self._calculate_example_coverage(result["questions"])
                }

                logger.info(f"🎉 SUCCESS 문제 생성 완료: {result['meta']['generated_count']}개 문제")
            else:
                logger.error(f"ERROR 문제 생성 실패: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"ERROR 문제 생성 서비스 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_id": file_id,
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_difficulty_consistency(self, questions: List[Dict[str, Any]]) -> float:
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

    def _calculate_question_uniqueness(self, questions: List[Dict[str, Any]]) -> float:
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

    def _calculate_example_coverage(self, questions: List[Dict[str, Any]]) -> float:
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

    async def _get_document_by_file_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        """단일 파일 ID로 문서 내용 조회"""
        try:
            logger.info(f"STEP_VECTOR 파일 ID로 문서 조회: {file_id}")

            # 벡터 DB 초기화
            if not self.vector_service.vector_db:
                await self.vector_service.initialize_vector_db()

            # 모든 문서 조회 (충분히 큰 수)
            all_docs_result = await self.vector_service.get_all_documents(10000)

            if not all_docs_result["success"]:
                logger.error("ERROR 전체 문서 조회 실패")
                return None

            # 지정된 file_id에 해당하는 파일 찾기
            target_file = None
            for file_info in all_docs_result["files"]:
                if file_info["file_id"] == file_id:
                    target_file = file_info
                    break

            if not target_file:
                logger.warning(f"WARNING 지정된 파일 ID를 찾을 수 없음: {file_id}")
                return None

            # 해당 파일의 실제 문서 내용 조회
            filename = target_file["filename"]

            # 해당 파일의 문서 청크들 조회
            file_chunks = await self._get_file_chunks(file_id)

            # 청크들을 하나의 문서로 합치기
            combined_content = ""
            for chunk in file_chunks:
                combined_content += chunk.get("content", "") + "\n\n"

            # 문서 정보 구성
            document = {
                "file_id": file_id,
                "filename": filename,
                "content": combined_content.strip(),
                "language": target_file.get("language", "unknown"),
                "file_size": target_file.get("file_size", 0),
                "total_chunks": target_file.get("total_chunks", 0),
                "pdf_loader": target_file.get("pdf_loader", "unknown"),
                "upload_timestamp": target_file.get("upload_timestamp"),
                "domain": self._identify_domain(filename)
            }

            logger.info(f"SUCCESS 문서 조회: {filename} ({len(combined_content)}자)")
            return document

        except Exception as e:
            logger.error(f"ERROR 문서 조회 실패: {e}")
            return None

    async def _get_file_chunks(self, file_id: str) -> List[Dict[str, Any]]:
        """특정 파일의 모든 청크 조회"""
        try:
            # 벡터 DB에서 해당 file_id를 가진 모든 문서 조회
            all_documents = await self.vector_service.vector_db.get_all_documents(10000)

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

            logger.info(f"SUCCESS 파일 청크 조회: {file_id} -> {len(file_chunks)}개 청크")
            return file_chunks

        except Exception as e:
            logger.error(f"ERROR 파일 청크 조회 실패: {e}")
            return []

    async def get_available_files(self) -> Dict[str, Any]:
        """문제 생성 가능한 파일 목록 조회"""
        try:
            logger.info("STEP_FILES 사용 가능한 파일 목록 조회")

            # 벡터 DB에서 파일 목록 조회
            files_result = await self.vector_service.get_all_documents(1000)

            if not files_result["success"]:
                return {
                    "success": False,
                    "error": "파일 목록 조회 실패",
                    "files": []
                }

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
                        "domain": self._identify_domain(file_info["filename"])
                    }
                    suitable_files.append(suitable_file)

            return {
                "success": True,
                "message": f"문제 생성 가능한 파일 {len(suitable_files)}개 조회 완료",
                "total_files": len(suitable_files),
                "files": suitable_files,
                "supported_difficulties": [d.value for d in DifficultyLevel],
                "supported_question_types": [q.value for q in QuestionType]
            }

        except Exception as e:
            logger.error(f"ERROR 파일 목록 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "files": []
            }

    def _identify_domain(self, filename: str) -> str:
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