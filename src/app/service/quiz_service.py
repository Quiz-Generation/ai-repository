"""
🎯 Quiz Generation Service - 성능 최적화 버전
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from functools import lru_cache

from ..agent.quiz_generator import (
    QuizGeneratorAgent,
    QuizRequest,
    DifficultyLevel,
    QuestionType
)
from ..service.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)


class QuizService:
    """문제 생성 서비스 - 성능 최적화 버전"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        초기화
        Args:
            openai_api_key: OpenAI API 키
        """
        self.vector_service = VectorDBService()
        self.quiz_agent = QuizGeneratorAgent(openai_api_key)

        # 🔥 성능 최적화: 캐시 추가
        self._document_cache = {}
        self._file_list_cache = None
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5분 캐시

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
        단일 파일 ID를 기반으로 문제 생성 - 성능 최적화 버전
        """
        try:
            logger.info(f"🚀 문제 생성 서비스 시작: {file_id}")

            # 1. 캐시된 문서 조회 (성능 개선)
            document_data = await self._get_document_by_file_id_optimized(file_id)
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

            # 3. 단일 AI 호출로 모든 문제 생성 (성능 개선)
            logger.info(f"STEP_AGENT AI 에이전트 문제 생성 시작 ({num_questions}개 문제)")

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

            result = await self.quiz_agent.generate_quiz(quiz_request, [document_data])

            if not result["success"]:
                return {
                    "success": False,
                    "error": f"문제 생성 실패: {result.get('error')}",
                    "file_id": file_id
                }

            # 4. 결과 처리 및 품질 검사
            questions = result.get("questions", [])

            # 품질 검사 및 후처리
            processed_questions = []
            for q in questions:
                q["difficulty"] = difficulty  # 명시적으로 태깅
                if q.get("choices") and q.get("answer"):
                    q = self._shuffle_choices_and_map_answer(q)
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
                    "difficulty_consistency": self._calculate_difficulty_consistency(processed_questions),
                    "question_uniqueness": self._calculate_question_uniqueness(processed_questions),
                    "example_coverage": self._calculate_example_coverage(processed_questions)
                }
            }

            logger.info(f"🎉 SUCCESS 문제 생성 완료: {len(processed_questions)}개")

            return {
                "success": True,
                "questions": processed_questions,
                "meta": meta,
                "overall_difficulty": difficulty_enum.value
            }

        except Exception as e:
            logger.error(f"ERROR 문제 생성 서비스 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_id": file_id,
                "timestamp": datetime.now().isoformat()
            }

    async def _get_document_by_file_id_optimized(self, file_id: str) -> Optional[Dict[str, Any]]:
        """최적화된 문서 조회 - 캐싱 적용"""
        try:
            # 🔥 캐시 확인
            cache_key = f"doc_{file_id}"
            if cache_key in self._document_cache:
                cached_doc = self._document_cache[cache_key]
                if datetime.now().timestamp() - cached_doc.get("timestamp", 0) < self._cache_ttl:
                    logger.info(f"CACHE_HIT 문서 캐시 사용: {file_id}")
                    return cached_doc["data"]
                else:
                    del self._document_cache[cache_key]

            logger.info(f"STEP_VECTOR 파일 ID로 문서 조회: {file_id}")

            # 벡터 DB 초기화 (한 번만)
            if not self.vector_service.vector_db:
                await self.vector_service.initialize_vector_db()

            # 🔥 최적화: 직접 file_id로 조회 (가능한 경우)
            try:
                # 벡터 DB에서 file_id로 직접 조회 시도
                if (self.vector_service.vector_db and
                    hasattr(self.vector_service.vector_db, 'search_by_metadata')):
                    file_docs = await self.vector_service.vector_db.search_by_metadata(
                        {"file_id": file_id}, limit=1000
                    )
                else:
                    # 폴백: 기존 방식 사용
                    raise NotImplementedError("search_by_metadata 메서드가 지원되지 않습니다")

                if file_docs:
                    # 청크들을 하나의 문서로 합치기
                    combined_content = ""
                    file_chunks = []

                    for doc in file_docs:
                        combined_content += doc.content + "\n\n"
                        file_chunks.append({
                            "id": doc.id,
                            "content": doc.content,
                            "metadata": doc.metadata
                        })

                    # chunk_index 순서로 정렬
                    file_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))

                    # 문서 정보 구성
                    document = {
                        "file_id": file_id,
                        "filename": file_chunks[0]["metadata"].get("filename", "unknown"),
                        "content": combined_content.strip(),
                        "language": file_chunks[0]["metadata"].get("language", "unknown"),
                        "file_size": file_chunks[0]["metadata"].get("file_size", 0),
                        "total_chunks": len(file_chunks),
                        "pdf_loader": file_chunks[0]["metadata"].get("pdf_loader", "unknown"),
                        "upload_timestamp": file_chunks[0]["metadata"].get("upload_timestamp"),
                        "domain": self._identify_domain(file_chunks[0]["metadata"].get("filename", ""))
                    }

                    # 🔥 캐시에 저장
                    self._document_cache[cache_key] = {
                        "data": document,
                        "timestamp": datetime.now().timestamp()
                    }

                    logger.info(f"SUCCESS 문서 조회 (최적화): {document['filename']} ({len(combined_content)}자)")
                    return document

            except Exception as e:
                logger.warning(f"WARNING 최적화된 조회 실패, 기존 방식 사용: {e}")

            # 🔥 폴백: 기존 방식 (캐시된 파일 목록 사용)
            files_result = await self._get_cached_file_list()

            if not files_result["success"]:
                logger.error("ERROR 파일 목록 조회 실패")
                return None

            # 지정된 file_id에 해당하는 파일 찾기
            target_file = None
            for file_info in files_result["files"]:
                if file_info["file_id"] == file_id:
                    target_file = file_info
                    break

            if not target_file:
                logger.warning(f"WARNING 지정된 파일 ID를 찾을 수 없음: {file_id}")
                return None

            # 해당 파일의 문서 청크들 조회
            file_chunks = await self._get_file_chunks_optimized(file_id)

            # 청크들을 하나의 문서로 합치기
            combined_content = ""
            for chunk in file_chunks:
                combined_content += chunk.get("content", "") + "\n\n"

            # 문서 정보 구성
            document = {
                "file_id": file_id,
                "filename": target_file["filename"],
                "content": combined_content.strip(),
                "language": target_file.get("language", "unknown"),
                "file_size": target_file.get("file_size", 0),
                "total_chunks": target_file.get("total_chunks", 0),
                "pdf_loader": target_file.get("pdf_loader", "unknown"),
                "upload_timestamp": target_file.get("upload_timestamp"),
                "domain": self._identify_domain(target_file["filename"])
            }

            # 🔥 캐시에 저장
            self._document_cache[cache_key] = {
                "data": document,
                "timestamp": datetime.now().timestamp()
            }

            logger.info(f"SUCCESS 문서 조회: {target_file['filename']} ({len(combined_content)}자)")
            return document

        except Exception as e:
            logger.error(f"ERROR 문서 조회 실패: {e}")
            return None

    async def _get_cached_file_list(self) -> Dict[str, Any]:
        """캐시된 파일 목록 조회"""
        current_time = datetime.now().timestamp()

        # 캐시가 유효한지 확인
        if (self._file_list_cache and self._cache_timestamp and
            current_time - self._cache_timestamp < self._cache_ttl):
            logger.info("CACHE_HIT 파일 목록 캐시 사용")
            return self._file_list_cache

        # 캐시 갱신
        logger.info("CACHE_MISS 파일 목록 캐시 갱신")
        files_result = await self.vector_service.get_all_documents(1000)

        if files_result["success"]:
            self._file_list_cache = files_result
            self._cache_timestamp = current_time

        return files_result

    async def _get_file_chunks_optimized(self, file_id: str) -> List[Dict[str, Any]]:
        """최적화된 파일 청크 조회"""
        try:
            # 🔥 최적화: file_id로 직접 조회
            if hasattr(self.vector_service.vector_db, 'search_by_metadata'):
                file_docs = await self.vector_service.vector_db.search_by_metadata(
                    {"file_id": file_id}, limit=1000
                )
            else:
                # 폴백: 기존 방식 사용
                all_documents = await self.vector_service.vector_db.get_all_documents(10000)
                file_docs = [doc for doc in all_documents if doc.metadata.get("file_id") == file_id]

            file_chunks = []
            for doc in file_docs:
                chunk_data = {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                file_chunks.append(chunk_data)

            # chunk_index 순서로 정렬
            file_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))

            logger.info(f"SUCCESS 파일 청크 조회 (최적화): {file_id} -> {len(file_chunks)}개 청크")
            return file_chunks

        except Exception as e:
            logger.error(f"ERROR 파일 청크 조회 실패: {e}")
            return []

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

    async def get_available_files(self) -> Dict[str, Any]:
        """문제 생성 가능한 파일 목록 조회 - 캐싱 적용"""
        try:
            logger.info("STEP_FILES 사용 가능한 파일 목록 조회")

            # 캐시된 파일 목록 사용
            files_result = await self._get_cached_file_list()

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

    def _shuffle_choices_and_map_answer(self, question: Dict[str, Any]) -> Dict[str, Any]:
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