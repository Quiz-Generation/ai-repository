from datetime import datetime
from typing import Any, Dict, List, Tuple
from src.app.agent.prompt.quiz_prompt_manager import DifficultyLevel, QuestionType
from src.app.agent.quiz_generator import QuizGeneratorAgent
from src.app.agent.quiz_generator import QuizRequest
from src.common.milvus.connect import VectorDBService

async def get_difficulty_distribution(
        overall: str,
        total: int
    ) -> List[Tuple[str, int]]:
    # 비율: 쉬움 70/25/5, 중간 30/50/20, 어려움 10/30/60
    table = {
        "easy":   [0.7, 0.25, 0.05],
        "medium": [0.3, 0.5, 0.2],
        "hard":   [0.1, 0.3, 0.6],
    }
    ratio = table.get(overall, [0.3, 0.5, 0.2])
    easy = round(total * ratio[0])
    medium = round(total * ratio[1])
    hard = total - easy - medium
    return [("easy", easy), ("medium", medium), ("hard", hard)]




async def generate_quiz_from_file(
        logger,
        vector_db: VectorDBService,
        file_id: str,
        num_questions: int = 5,
        difficulty: str = "medium",
        question_type: str = "multiple_choice",
        category: str = "",
        sub_category: str = ""
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
            quiz_agent = QuizGeneratorAgent()

            # 1. 파일 ID로 문서 조회
            document_data = await vector_db.get_document_by_file_id(file_id)
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

            dist = await get_difficulty_distribution(difficulty_enum.value, num_questions)
            all_questions = []
            for diff, count in dist:
                if count <= 0:
                    continue
                quiz_request = QuizRequest(
                    file_ids=[file_id],
                    num_questions=count,
                    difficulty=DifficultyLevel(diff),
                    question_type=question_type_enum,
                    category=category,
                    sub_category=sub_category,
                    additional_instructions=[
                        f"이 문제들은 '{diff}' 난이도로 생성되어야 합니다. 각 문제의 difficulty 필드를 반드시 '{diff}'로 설정하세요.",
                        "각 문제는 구체적인 예시나 실제 응용 사례를 포함해야 합니다.",
                        "문제는 서로 중복되지 않아야 하며, 각각 독립적인 개념을 다뤄야 합니다.",
                        "선택지의 경우, 명확한 정답과 그럴듯한 오답을 포함해야 합니다.",
                        "문제의 난이도는 일관성을 유지해야 합니다.",
                        "문제는 실제 학습 목표와 연관되어야 합니다."
                    ]
                )
                logger.info(f"STEP_AGENT AI 에이전트 문제 생성 시작 (난이도: {diff}, 개수: {count})")
                result = await quiz_agent.generate_quiz(quiz_request, [document_data])
                if result["success"]:
                    for q in result.get("questions", []):
                        q["difficulty"] = diff  # 명시적으로 태깅
                        if q.get("choices") and q.get("answer"):
                            q = quiz_agent._shuffle_choices_and_map_answer(q)
                        all_questions.append(q)
                else:
                    logger.error(f"ERROR 문제 생성 실패: {result.get('error')}")

            # 문제 개수 맞추기(혹시 초과 생성 시)
            all_questions = all_questions[:num_questions]

            # id를 1부터 다시 부여
            for idx, q in enumerate(all_questions, 1):
                q["id"] = idx

            # 메타데이터 추가
            meta = {
                "generation_timestamp": datetime.now().isoformat(),
                "service_version": "1.0.0",
                "source_file": document_data.get("filename"),
                "file_id": file_id,
                "overall_difficulty": difficulty_enum.value,
                "generated_count": len(all_questions),
                "quality_metrics": {
                    "difficulty_consistency": quiz_agent.calculate_difficulty_consistency(all_questions),
                    "question_uniqueness": quiz_agent.calculate_question_uniqueness(all_questions),
                    "example_coverage": quiz_agent.calculate_example_coverage(all_questions)
                }
            }

            logger.info(f"🎉 SUCCESS 문제 생성 완료: {len(all_questions)}개 (분포: {dist})")

            return {
                "success": True,
                "questions": all_questions,
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