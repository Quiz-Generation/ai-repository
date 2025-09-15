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
from src.common.vector.connect import VectorDBService
from src.common.error import ErrorCode, JSendError

logger = logging.getLogger(__name__)


async def generate_quiz_from_file_streaming(
    request_id: str,
    user_idx: int,
    logger,
    vector_db: VectorDBService,
    file_id: str,
    num_questions: int = 5,
    difficulty: str = "medium",
    question_type: str = "multiple_choice",
    custom_topic: Optional[str] = None,
    category: Optional[str] = None,
    sub_category: Optional[str] = None
) -> None:
    """
    스트리밍 방식으로 문제 생성 (Redis 스트림으로 실시간 전송)
    """
    from src.common.redis.connect import (
        push_quiz_error_to_stream
    )
    
    try:
        logger.info(f"🚀 스트리밍 문제 생성 서비스 시작: {request_id} - {file_id}")
        
        # 1. 파일 ID로 문서 조회
        document_data = await _get_document_by_file_id(logger, vector_db, file_id)
        if not document_data:
            # 더 자세한 오류 메시지 제공
            error_message = f"파일 ID '{file_id}'에 해당하는 문서를 찾을 수 없습니다. "
            error_message += "사용 가능한 파일 목록을 확인하려면 /api/v2/quiz/available-files 엔드포인트를 사용하세요."
            
            await push_quiz_error_to_stream(
                request_id=request_id,
                error_message=error_message,
                user_idx=user_idx
            )
            return

        # 2. 요청 객체 생성
        try:
            difficulty_enum = DifficultyLevel(difficulty.lower())
            question_type_enum = QuestionType(question_type.lower())
        except ValueError as e:
            await push_quiz_error_to_stream(
                request_id=request_id,
                error_message=f"잘못된 파라미터: {str(e)}",
                user_idx=user_idx
            )
            return

        # 3. AI 에이전트 초기화
        import os
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            await push_quiz_error_to_stream(
                request_id=request_id,
                error_message="OpenAI API 키가 설정되지 않았습니다",
                user_idx=user_idx
            )
            return

        # 에이전트 캐시에서 재사용
        if not hasattr(generate_quiz_from_file_streaming, '_cached_agent'):
            generate_quiz_from_file_streaming._cached_agent = QuizGeneratorAgent(openai_api_key)
            logger.info("🔄 AI 에이전트 캐시 생성")

        quiz_agent = generate_quiz_from_file_streaming._cached_agent

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

        # 5. 스트리밍 방식으로 문제 생성
        logger.info(f"STEP_AGENT 스트리밍 AI 에이전트 문제 생성 시작 ({num_questions}개 문제)")
        
        # 스트리밍 문제 생성 호출
        result = await quiz_agent.generate_quiz_streaming(
            request_id=request_id,
            user_idx=user_idx,
            request=quiz_request,
            documents=[document_data]
        )
        
        if result["success"]:
            logger.info(f"✅ 스트리밍 문제 생성 완료: {request_id}")
        else:
            logger.error(f"❌ 스트리밍 문제 생성 실패: {request_id} - {result.get('error')}")
            
    except Exception as e:
        logger.error(f"❌ 스트리밍 문제 생성 서비스 실패: {request_id} - {e}")
        await push_quiz_error_to_stream(
            request_id=request_id,
            error_message=f"서비스 오류: {str(e)}",
            user_idx=user_idx
        )


async def generate_quiz_from_file(
    logger,
    vector_db: VectorDBService,
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
    """
    try:
        logger.info(f"🚀 문제 생성 서비스 시작: {file_id}")

        # 1. 파일 ID로 문서 조회
        document_data = await _get_document_by_file_id(logger, vector_db, file_id)
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

                # 3. AI 에이전트 초기화 (캐싱 적용)
        import os
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise JSendError(
                code=ErrorCode.Common.DEFAULT_ERROR[0],
                message="OpenAI API 키가 설정되지 않았습니다. OPENAI_API_KEY 환경변수를 설정해주세요."
            )

        # 🔥 최적화: 전역 캐시에서 에이전트 재사용
        if not hasattr(generate_quiz_from_file, '_cached_agent'):
            generate_quiz_from_file._cached_agent = QuizGeneratorAgent(openai_api_key)
            logger.info("🔄 AI 에이전트 캐시 생성")

        quiz_agent = generate_quiz_from_file._cached_agent

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
        logger.info(f"STEP_AGENT AI 에이전트 문제 생성 시작 ({num_questions}개 문제)")
        result = await quiz_agent.generate_quiz(quiz_request, [document_data])

        if not result["success"]:
            return {
                "success": False,
                "error": f"문제 생성 실패: {result.get('error')}",
                "file_id": file_id
            }

        # 6. 결과 처리 및 품질 검사
        questions = result.get("questions", [])

        # 품질 검사 및 후처리
        processed_questions = []
        for q in questions:
            q["difficulty"] = difficulty  # 명시적으로 태깅
            if q.get("choices") and q.get("answer"):
                q = _shuffle_choices_and_map_answer(q)
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
                "difficulty_consistency": _calculate_difficulty_consistency(processed_questions),
                "question_uniqueness": _calculate_question_uniqueness(processed_questions),
                "example_coverage": _calculate_example_coverage(processed_questions)
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
        logger.error(f"❌ 문제 생성 서비스 실패: {file_id} - {e}")
        return {
            "success": False,
            "error": f"서비스 오류: {str(e)}",
            "file_id": file_id
        }


async def get_available_files(
    logger,
    vector_db: VectorDBService
) -> Dict[str, Any]:
    """문제 생성 가능한 파일 목록 조회"""
    try:
        logger.info("STEP_FILES 사용 가능한 파일 목록 조회")

        # 벡터 DB에서 파일 목록 조회
        files_result = await vector_db.get_all_documents(1000)

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
                    "domain": _identify_domain(file_info["filename"])
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
        logger.error(f"❌ 파일 목록 조회 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "files": []
        }


async def _get_document_by_file_id(
    logger,
    vector_db: VectorDBService,
    file_id: str
) -> Optional[Dict[str, Any]]:
    """단일 파일 ID로 문서 내용 조회 (최적화된 버전)"""
    try:
        logger.info(f"STEP_VECTOR 파일 ID로 문서 조회: {file_id}")

        # 🔥 최적화: file_id로 직접 필터링하여 조회
        if hasattr(vector_db, 'vector_db') and vector_db.vector_db:
            # 벡터 DB에서 해당 file_id를 가진 문서들만 조회
            all_documents = await vector_db.vector_db.get_all_documents(10000)

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
                "domain": _identify_domain(target_file_info["filename"])
            }

            logger.info(f"SUCCESS 문서 조회: {target_file_info['filename']} ({len(combined_content)}자, {len(target_chunks)}개 청크)")
            return document
        else:
            # 기존 방식으로 fallback
            all_docs_result = await vector_db.get_all_documents(10000)
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

            file_chunks = await _get_file_chunks(vector_db, file_id)
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
                "domain": _identify_domain(target_file["filename"])
            }

            logger.info(f"SUCCESS 문서 조회: {target_file['filename']} ({len(combined_content)}자)")
            return document

    except Exception as e:
        logger.error(f"❌ 문서 조회 실패: {e}")
        return None


async def _get_file_chunks(
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

        logger.info(f"SUCCESS 파일 청크 조회: {file_id} -> {len(file_chunks)}개 청크")
        return file_chunks

    except Exception as e:
        logger.error(f"❌ 파일 청크 조회 실패: {e}")
        return []


def _calculate_difficulty_consistency(questions: List[Dict[str, Any]]) -> float:
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


def _calculate_question_uniqueness(questions: List[Dict[str, Any]]) -> float:
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


def _calculate_example_coverage(questions: List[Dict[str, Any]]) -> float:
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


def _identify_domain(filename: str) -> str:
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


def _shuffle_choices_and_map_answer(question: Dict[str, Any]) -> Dict[str, Any]:
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
