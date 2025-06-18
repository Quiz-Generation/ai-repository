"""
🤖 Quiz Generation AI Agent using LangGraph
"""
import logging
import os
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum
import json
import time

# tokenizers 병렬 처리 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.prompts import ChatPromptTemplate

# 🔥 프롬프트 관리자 임포트
from .prompt import QuizPromptManager
from .prompt.quiz_prompt_manager import DifficultyLevel, QuestionType

logger = logging.getLogger(__name__)


@dataclass
class QuizRequest:
    """문제 생성 요청"""
    file_ids: List[str]                 # 대상 파일 ID들
    num_questions: int = 5              # 생성할 문제 수
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    question_type: QuestionType = QuestionType.MULTIPLE_CHOICE
    custom_topic: Optional[str] = None  # 특정 주제 지정
    additional_instructions: Optional[List[str]] = None  # 추가 지시사항


class QuizState(TypedDict):
    """LangGraph 상태 관리"""
    # 입력
    request: QuizRequest
    documents: List[Dict[str, Any]]

    # 워크플로우 상태
    summary: str
    core_topics: List[str]
    keywords: List[str]
    generated_questions: List[Dict[str, Any]]

    # 메타데이터
    current_step: str
    errors: List[str]
    domain_context: Dict[str, Any]


class QuizGeneratorAgent:
    """문제 생성 AI 에이전트"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        초기화
        Args:
            openai_api_key: OpenAI API 키 (환경변수에서 자동 로드 가능)
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 직접 전달하세요.")

        # LLM 모델 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # 비용 효율적인 모델
            temperature=0.7,      # 창의성과 일관성의 균형
            api_key=self.openai_api_key
        )

        # LangGraph 워크플로우 구성
        self.workflow = self._create_workflow()

        # 🎯 프롬프트 관리자 초기화
        self.prompt_manager = QuizPromptManager()

        # 프롬프트 템플릿 초기화
        self.summary_template = ChatPromptTemplate.from_messages([
            ("system", "당신은 전문 교육 컨텐츠 분석가입니다."),
            ("human", "{prompt}")
        ])

        self.topic_template = ChatPromptTemplate.from_messages([
            ("system", "당신은 전문 교육과정 설계자입니다."),
            ("human", "{prompt}")
        ])

        self.keyword_template = ChatPromptTemplate.from_messages([
            ("system", "당신은 전문 시험 출제 전문가입니다."),
            ("human", "{prompt}")
        ])

        self.question_template = ChatPromptTemplate.from_messages([
            ("system", "{system_message}"),
            ("human", "{prompt}")
        ])

        self.validation_template = ChatPromptTemplate.from_messages([
            ("system", "당신은 전문 교육 컨텐츠 품질 검증 전문가입니다."),
            ("human", "{prompt}")
        ])

        # 체인 초기화
        self.summary_chain = self.summary_template | self.llm
        self.topic_chain = self.topic_template | self.llm
        self.keyword_chain = self.keyword_template | self.llm
        self.question_chain = self.question_template | self.llm
        self.validation_chain = self.validation_template | self.llm

    def _create_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(QuizState)

        # 병렬 처리 노드 추가
        workflow.add_node("parallel_processor", self._parallel_process)
        workflow.add_node("question_generator", self._generate_questions)

        # 워크플로우 순서 정의
        workflow.set_entry_point("parallel_processor")
        workflow.add_edge("parallel_processor", "question_generator")
        workflow.add_edge("question_generator", END)

        return workflow.compile()

    async def _parallel_process(self, state: QuizState) -> QuizState:
        """📄 병렬 처리: 문서 요약, 핵심 주제 추출, 키워드 추출"""
        try:
            parallel_start = time.time()
            logger.info("병렬 처리 시작")

            # 문서 내용 결합 및 전처리
            combined_content = ""
            domain_info = {}
            total_sentences = 0
            total_paragraphs = 0

            for doc in state["documents"]:
                filename = doc.get("filename", "Unknown")
                content = doc.get("content", "")

                # 문장과 단락 수 계산
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                total_sentences += len(sentences)
                total_paragraphs += len(paragraphs)

                combined_content += f"\n\n=== {filename} ===\n{content}"
                domain_info[filename] = {
                    "language": doc.get("language", "unknown"),
                    "file_size": doc.get("file_size", 0),
                    "chunk_count": doc.get("total_chunks", 0),
                    "sentence_count": len(sentences),
                    "paragraph_count": len(paragraphs)
                }

            # 문제 수 재계산
            base_questions = max(3, total_sentences // 4)  # 4문장당 1문제
            complexity_factor = min(1.5, 1 + (total_paragraphs / total_sentences))
            concept_factor = min(0.5, len(domain_info) * 0.1)

            recommended_questions = int(base_questions * complexity_factor * (1 + concept_factor))
            recommended_questions = min(max(5, recommended_questions), 20)  # 5-20개 사이로 제한

            # 문제 수 업데이트
            state["request"].num_questions = recommended_questions

            # 병렬 처리 태스크 정의
            async def summarize_documents():
                summary_prompt = self.prompt_manager.get_prompt("summary").format(
                    content=combined_content
                )
                return await self.summary_chain.ainvoke({"prompt": summary_prompt})

            async def extract_topics():
                topic_prompt = self.prompt_manager.get_prompt("topic").format(
                    content=combined_content,
                    difficulty=state["request"].difficulty.value,
                    num_questions=recommended_questions,
                    question_type=state["request"].question_type.value,
                    num_topics=recommended_questions + 3
                )
                return await self.topic_chain.ainvoke({"prompt": topic_prompt})

            async def extract_keywords():
                keyword_prompt = self.prompt_manager.get_prompt("keyword").format(
                    content=combined_content,
                    difficulty=state["request"].difficulty.value,
                    question_type=state["request"].question_type.value,
                    num_keywords=recommended_questions * 3
                )
                return await self.keyword_chain.ainvoke({"prompt": keyword_prompt})

            # 병렬 실행
            import asyncio
            topics_response, summary_response, keywords_response = await asyncio.gather(
                extract_topics(), summarize_documents(), extract_keywords()
            )
            logger.info(f"[전처리] 병렬 전체 소요 시간: {time.time() - parallel_start:.2f}초")
            summary = summary_response.content
            topics = [line.strip().lstrip('- •').strip() for line in topics_response.content.split('\n') if line.strip().startswith(('-', '•'))]
            keywords = [kw.strip() for kw in keywords_response.content.split(',') if kw.strip()]
            logger.info(f"[전처리] 완료 (총 소요 시간: {time.time() - parallel_start:.2f}초)")

            # 상태 업데이트
            state["summary"] = summary
            state["core_topics"] = topics
            state["keywords"] = keywords
            state["domain_context"] = {
                **domain_info,
                "total_sentences": total_sentences,
                "total_paragraphs": total_paragraphs,
                "recommended_questions": recommended_questions
            }
            state["current_step"] = "parallel_processor"

            logger.info(f"SUCCESS 병렬 처리 완료: {recommended_questions}개 문제 추천")
            return state

        except Exception as e:
            logger.error(f"ERROR 병렬 처리 실패: {e}")
            state["errors"].append(f"병렬 처리 실패: {str(e)}")
            return state

    async def _generate_questions(self, state: QuizState) -> QuizState:
        """❓ 4단계: 균형 잡힌 문제 생성"""
        try:
            generate_start = time.time()
            logger.info("STEP4 균형 잡힌 문제 생성 시작")

            request = state["request"]
            summary = state["summary"]
            topics = state["core_topics"]
            keywords = state["keywords"]

            # 추가 지시사항이 있는 경우 프롬프트에 추가
            additional_guide = ""
            if request.additional_instructions:
                additional_guide = "\n\n📝 **추가 지시사항**:\n" + "\n".join(f"- {instruction}" for instruction in request.additional_instructions)

            # 🎯 1단계: PDF 기반 문제 생성
            pdf_prompt = self.prompt_manager.get_prompt("question").format(
                summary=summary,
                topics="\n".join(f"- {topic}" for topic in topics),
                keywords="\n".join(f"- {keyword}" for keyword in keywords),
                num_questions=request.num_questions * 4,  # 요청 수의 4배로 생성
                difficulty=request.difficulty.value,
                question_type=request.question_type.value
            )

            # PDF 기반 문제 생성과 AI 기반 문제 생성을 병렬로 실행
            async def generate_pdf_questions():
                response = await self.question_chain.ainvoke({
                    "system_message": "당신은 전문 교육 컨텐츠 개발자입니다.",
                    "prompt": pdf_prompt
                })
                return self._parse_questions(response.content)

            async def generate_ai_questions():
                # AI 기반 문제 생성을 위한 프롬프트
                ai_prompt = self.prompt_manager.get_prompt("question").format(
                    summary=summary,
                    topics="\n".join(f"- {topic}" for topic in topics),
                    keywords="\n".join(f"- {keyword}" for keyword in keywords),
                    num_questions=request.num_questions * 3,  # 요청 수의 3배로 생성
                    difficulty=request.difficulty.value,
                    question_type=request.question_type.value
                )
                response = await self.question_chain.ainvoke({
                    "system_message": "당신은 전문 교육 컨텐츠 개발자입니다.",
                    "prompt": ai_prompt
                })
                return self._parse_questions(response.content)

            # 병렬 실행
            import asyncio
            pdf_questions, ai_questions = await asyncio.gather(
                generate_pdf_questions(),
                generate_ai_questions()
            )

            # 📊 최종 문제 목록 생성
            final_questions = pdf_questions + ai_questions

            # 🔄 문제 순서 섞기
            import random
            random.shuffle(final_questions)

            # 기본 품질 검사
            final_questions = self._basic_quality_check(final_questions)

            # 문제 수가 부족한 경우 재시도
            retry_count = 0
            while len(final_questions) < request.num_questions and retry_count < 3:  # 최대 3번까지 재시도
                logger.info(f"문제 수 부족 ({len(final_questions)}/{request.num_questions}), 추가 생성 시도 {retry_count + 1}")

                # 추가 문제 생성 (부족한 수의 3배로 생성)
                additional_prompt = self.prompt_manager.get_prompt("question").format(
                    summary=summary,
                    topics="\n".join(f"- {topic}" for topic in topics),
                    keywords="\n".join(f"- {keyword}" for keyword in keywords),
                    num_questions=(request.num_questions - len(final_questions)) * 3,
                    difficulty=request.difficulty.value,
                    question_type=request.question_type.value
                )

                response = await self.question_chain.ainvoke({
                    "system_message": "당신은 전문 교육 컨텐츠 개발자입니다.",
                    "prompt": additional_prompt
                })

                additional_questions = self._parse_questions(response.content)
                additional_questions = self._basic_quality_check(additional_questions)

                final_questions.extend(additional_questions)
                retry_count += 1

            # 최종 중복 제거 및 품질 검사 한 번 더
            final_questions = self._basic_quality_check(final_questions)

            # 문제 수 조정 (최종적으로 반드시 요청 수만큼만 반환)
            final_questions = final_questions[:request.num_questions]

            # ID 순차적으로 부여
            for i, question in enumerate(final_questions, 1):
                question["id"] = i

                # 다중선택 문제의 경우 보기 번호 추가
                if question.get("type") == "multiple_choice" and isinstance(question.get("options"), list):
                    numbered_options = []
                    for idx, opt in enumerate(question["options"], 1):
                        numbered_options.append(f"{idx}. {opt}")
                    question["options"] = numbered_options

                    # 정답도 번호로 변환
                    if "correct_answer" in question:
                        try:
                            answer_idx = [opt.replace(f"{idx}. ", "") for idx, opt in enumerate(numbered_options, 1)].index(question["correct_answer"]) + 1
                            question["correct_answer_number"] = answer_idx
                        except Exception:
                            question["correct_answer_number"] = None

            state["generated_questions"] = final_questions
            state["current_step"] = "question_generator"

            # 📊 분포 확인 로깅
            basic_count = sum(1 for q in final_questions if q.get("problem_level") == "basic")
            concept_count = sum(1 for q in final_questions if q.get("problem_level") == "concept")
            app_count = sum(1 for q in final_questions if q.get("problem_level") == "application")
            pdf_count = sum(1 for q in final_questions if q.get("source") != "ai_generated")
            ai_count = sum(1 for q in final_questions if q.get("source") == "ai_generated")

            logger.info(f"SUCCESS 문제 생성 완료: 총 {len(final_questions)}개")
            logger.info(f"- PDF 기반: {pdf_count}개")
            logger.info(f"- AI 기반: {ai_count}개")
            logger.info(f"- 기본 개념: {basic_count}개")
            logger.info(f"- 개념 연계: {concept_count}개")
            logger.info(f"- 응용 문제: {app_count}개")
            logger.info(f"[실행시간] 문제 생성 소요 시간: {time.time() - generate_start:.2f}초")

            return state

        except Exception as e:
            logger.error(f"ERROR 문제 생성 실패: {e}")
            state["errors"].append(f"문제 생성 실패: {str(e)}")
            return state

    def _basic_quality_check(self, questions: List[Dict]) -> List[Dict]:
        """기본적인 품질 검사 수행"""
        valid_questions = []
        seen_questions = set()

        for q in questions:
            try:
                # 필수 필드 확인
                if not all(k in q for k in ["question", "options", "correct_answer", "explanation"]):
                    continue

                # 중복 문제 제거 (유사도 기반, 기준 완화)
                question_text = q["question"].lower().strip()
                if any(self._is_similar(question_text, seen, threshold=0.9) for seen in seen_questions):  # 유사도 기준 상향
                    continue
                seen_questions.add(question_text)

                # 선택지 검증 (최소 2개 이상)
                if len(q["options"]) < 2:
                    continue

                # 정답이 선택지에 포함되어 있는지 확인
                if q["correct_answer"] not in q["options"]:
                    continue

                # 문제 수준 설정
                if "problem_level" not in q:
                    q["problem_level"] = "basic"

                valid_questions.append(q)
            except Exception as e:
                logger.warning(f"문제 품질 검사 중 오류 발생: {e}")
                continue

        return valid_questions

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.9) -> bool:
        """두 텍스트의 유사도 검사 (기준 상향)"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio() > threshold

    def _parse_questions(self, content: str) -> List[Dict]:
        """JSON 응답 파싱"""
        try:
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content.strip()

            questions_data = json.loads(json_content)
            return questions_data.get("questions", [])

        except json.JSONDecodeError as e:
            logger.error(f"ERROR JSON 파싱 실패: {e}")
            logger.error(f"LLM 응답 내용: {content[:500]}...")
            return []

    def smart_truncate(self, text, max_length=2000):
        """앞/중간/끝 샘플링 방식으로 텍스트를 자름"""
        if len(text) <= max_length:
            return text
        part = max_length // 3
        return text[:part] + text[len(text)//2:len(text)//2+part] + text[-part:]

    async def generate_quiz(self, request: QuizRequest, documents: List[Dict[str, Any]], use_combined_prompt: bool = False, use_sampling: bool = False) -> Dict[str, Any]:
        """
        문제 생성 메인 메서드 (문서별 전처리까지 완전 비동기 + 문제 생성 2문제씩 병렬)
        use_combined_prompt: 무시(항상 분리 방식)
        use_sampling: True면 샘플링(앞/중간/끝), False면 전체 결합
        """
        import time
        try:
            total_start = time.time()
            logger.info(f"🚀 문제 생성 AI 에이전트 시작 (문서별 전처리 완전 비동기, 문제 생성 병렬 2문제씩, use_sampling={use_sampling})")

            # 난이도 값 검증
            if not isinstance(request.difficulty, DifficultyLevel):
                try:
                    request.difficulty = DifficultyLevel(request.difficulty)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"잘못된 파라미터: '{request.difficulty}' is not a valid DifficultyLevel",
                        "valid_difficulty": [level.value for level in DifficultyLevel],
                        "valid_question_types": [qtype.value for qtype in QuestionType]
                    }

            # 문제 유형 값 검증
            if not isinstance(request.question_type, QuestionType):
                try:
                    request.question_type = QuestionType(request.question_type)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"잘못된 파라미터: '{request.question_type}' is not a valid QuestionType",
                        "valid_difficulty": [level.value for level in DifficultyLevel],
                        "valid_question_types": [qtype.value for qtype in QuestionType]
                    }

            preprocess_start = time.time()
            logger.info(f"[전처리] 시작 (문서별 완전 비동기, use_sampling={use_sampling})")

            import asyncio
            async def process_single_doc(doc):
                filename = doc.get("filename", "Unknown")
                content = doc.get("content", "")
                if use_sampling:
                    content = self.smart_truncate(content, 2000)
                summary_prompt = self.prompt_manager.get_prompt("summary").format(content=content)
                topic_prompt = self.prompt_manager.get_prompt("topic").format(
                    content=content,
                    difficulty=request.difficulty.value,
                    num_questions=request.num_questions,
                    question_type=request.question_type.value,
                    num_topics=request.num_questions + 3
                )
                keyword_prompt = self.prompt_manager.get_prompt("keyword").format(
                    content=content,
                    difficulty=request.difficulty.value,
                    question_type=request.question_type.value,
                    num_keywords=request.num_questions * 3
                )
                s_task = self.summary_chain.ainvoke({"prompt": summary_prompt})
                t_task = self.topic_chain.ainvoke({"prompt": topic_prompt})
                k_task = self.keyword_chain.ainvoke({"prompt": keyword_prompt})
                summary_resp, topic_resp, keyword_resp = await asyncio.gather(s_task, t_task, k_task)
                return {
                    "summary": summary_resp.content,
                    "topics": topic_resp.content,
                    "keywords": keyword_resp.content
                }

            doc_tasks = [process_single_doc(doc) for doc in documents]
            doc_results = await asyncio.gather(*doc_tasks)

            # 결과 합치기
            summary = "\n".join([r["summary"] for r in doc_results])
            topics = []
            for r in doc_results:
                topics.extend([line.strip().lstrip('- •').strip() for line in r["topics"].split('\n') if line.strip().startswith(('-', '•'))])
            keywords = []
            for r in doc_results:
                keywords.extend([kw.strip() for kw in r["keywords"].split(',') if kw.strip()])
            logger.info(f"[전처리] 완료 (총 소요 시간: {time.time() - preprocess_start:.2f}초)")

            # 2. 문제 생성: 2문제씩 5번 병렬
            generate_start = time.time()
            logger.info("[문제 생성] 시작 (2문제씩 5회 병렬)")
            batch_size = 2
            total_batches = (request.num_questions + batch_size - 1) // batch_size
            async def generate_questions_batch(batch_num):
                question_prompt = self.prompt_manager.get_prompt("question").format(
                    summary=summary,
                    topics="\n".join(f"- {topic}" for topic in topics),
                    keywords="\n".join(f"- {keyword}" for keyword in keywords),
                    num_questions=batch_size,
                    difficulty=request.difficulty.value,
                    question_type=request.question_type.value
                )
                response = await self.question_chain.ainvoke({
                    "system_message": "당신은 전문 교육 컨텐츠 개발자입니다.",
                    "prompt": question_prompt
                })
                return self._parse_questions(response.content)
            tasks = [generate_questions_batch(i) for i in range(total_batches)]
            results = await asyncio.gather(*tasks)
            questions = [q for batch in results for q in batch]
            logger.info(f"[문제 생성] 완료 (소요 시간: {time.time() - generate_start:.2f}초)")

            # 3. 후처리: 중복 제거, 품질 검사, 슬라이싱, 보기 번호 부여
            post_start = time.time()
            logger.info("[후처리] 시작")
            questions = self._basic_quality_check(questions)
            questions = questions[:request.num_questions]
            for i, question in enumerate(questions, 1):
                question["id"] = i
                if question.get("type") == "multiple_choice" and isinstance(question.get("options"), list):
                    numbered_options = []
                    for idx, opt in enumerate(question["options"], 1):
                        numbered_options.append(f"{idx}. {opt}")
                    question["options"] = numbered_options
                    if "correct_answer" in question:
                        try:
                            answer_idx = [opt.replace(f"{idx}. ", "") for idx, opt in enumerate(numbered_options, 1)].index(question["correct_answer"]) + 1
                            question["correct_answer_number"] = answer_idx
                        except Exception:
                            question["correct_answer_number"] = None
            logger.info(f"[후처리] 완료 (소요 시간: {time.time() - post_start:.2f}초)")

            total_end = time.time()
            logger.info(f"[실행시간] 전체 문제 생성 프로세스 소요 시간: {total_end - total_start:.2f}초")
            logger.info("🎉 SUCCESS 문제 생성 완료")

            return {
                "success": True,
                "request": {
                    "file_ids": request.file_ids,
                    "num_questions": request.num_questions,
                    "difficulty": request.difficulty.value,
                    "question_type": request.question_type.value
                },
                "process_info": {
                    "summary": summary,
                    "core_topics": topics,
                    "keywords": keywords
                },
                "questions": questions,
                "meta": {
                    "generated_count": len(questions),
                    "final_step": "generate_quiz"
                }
            }
        except Exception as e:
            logger.error(f"ERROR 문제 생성 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "request": {
                    "file_ids": request.file_ids,
                    "num_questions": request.num_questions,
                    "difficulty": request.difficulty.value,
                    "question_type": request.question_type.value
                }
            }