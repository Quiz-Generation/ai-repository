"""
LLM 팩토리 패턴 구현
다양한 LLM 모델을 추상화하여 쉽게 교체할 수 있도록 설계
"""
import os
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from enum import Enum
from lagnchain_fastapi_app.app.core.config import get_settings
from lagnchain_fastapi_app.app.schemas.quiz_schema import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)




class BaseLLMService(ABC):
    """LLM 서비스 추상 클래스"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider
        self.model_name = config.model_name

    @abstractmethod
    def generate_quiz(
        self,
        context: str,
        num_questions: int,
        difficulty: str,
        question_types: List[str],
        topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """퀴즈 생성 (추상 메서드)"""
        pass

    @abstractmethod
    def extract_topics(self, context: str) -> List[str]:
        """문서에서 주요 토픽 추출 (추상 메서드)"""
        pass

    @abstractmethod
    def validate_question_quality(self, question_data: Dict[str, Any]) -> bool:
        """문제 품질 검증 (추상 메서드)"""
        pass


class OpenAILLMService(BaseLLMService):
    """OpenAI GPT 기반 LLM 서비스"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._setup_client()

    def _setup_client(self):
        """OpenAI 클라이언트 설정"""
        try:
            import openai
            settings = get_settings()
            self.client = openai.OpenAI(
                api_key=self.config.api_key or settings.OPENAI_API_KEY
            )
            logger.info(f"OpenAI 클라이언트 초기화 완료: {self.model_name}")
        except ImportError:
            raise ImportError("OpenAI 패키지가 설치되지 않았습니다: pip install openai")
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            raise

    def generate_quiz(
        self,
        context: str,
        num_questions: int,
        difficulty: str,
        question_types: List[str],
        topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """OpenAI를 사용한 퀴즈 생성"""

        # 프롬프트 구성
        prompt = self._build_quiz_generation_prompt(
            context, num_questions, difficulty, question_types, topics
        )

        try:
            logger.info(f"OpenAI 퀴즈 생성 시작: {num_questions}문제, 난이도: {difficulty}")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "당신은 PDF 문서 기반 퀴즈 생성 전문가입니다. 주어진 컨텍스트를 바탕으로 고품질의 문제를 생성하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            # 응답 파싱
            result_text = response.choices[0].message.content
            if result_text is None:
                raise ValueError("OpenAI 응답이 비어있습니다")
            quiz_data = self._parse_quiz_response(result_text)

            logger.info(f"OpenAI 퀴즈 생성 완료: {len(quiz_data.get('questions', []))}문제")
            return {
                "questions": quiz_data.get("questions", []),
                "success": True,
                "model_used": self.model_name,
                "provider": "openai"
            }

        except Exception as e:
            logger.error(f"OpenAI 퀴즈 생성 실패: {e}")
            return {
                "questions": [],
                "success": False,
                "error": str(e),
                "model_used": self.model_name
            }

    def extract_topics(self, context: str) -> List[str]:
        """OpenAI를 사용한 주요 토픽 추출"""

        prompt = f"""
다음 텍스트에서 주요 토픽들을 추출하세요. 퀴즈 문제로 만들기 좋은 핵심 주제들을 찾아주세요.

텍스트:
{context[:3000]}...

JSON 형식으로 응답하세요:
{{
    "topics": ["토픽1", "토픽2", "토픽3"],
    "main_subjects": ["주요주제1", "주요주제2"]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "주어진 텍스트에서 퀴즈 문제 생성에 적합한 핵심 토픽을 추출하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            response_content = response.choices[0].message.content
            if response_content is None:
                raise ValueError("OpenAI 토픽 추출 응답이 비어있습니다")

            result = json.loads(response_content)
            topics = result.get("topics", []) + result.get("main_subjects", [])

            logger.info(f"추출된 토픽: {topics}")
            return topics[:10]  # 최대 10개

        except Exception as e:
            logger.error(f"토픽 추출 실패: {e}")
            return ["일반"]

    def validate_question_quality(self, question_data: Dict[str, Any]) -> bool:
        """문제 품질 검증"""

        # 기본 검증
        if not question_data.get("question") or len(question_data["question"]) < 10:
            return False

        if not question_data.get("correct_answer"):
            return False

        # 객관식 문제 검증
        if question_data.get("question_type") == "multiple_choice":
            options = question_data.get("options", [])
            if len(options) < 2 or question_data["correct_answer"] not in options:
                return False

        return True

    def _build_quiz_generation_prompt(
        self,
        context: str,
        num_questions: int,
        difficulty: str,
        question_types: List[str],
        topics: Optional[List[str]] = None
    ) -> str:
        """퀴즈 생성 프롬프트 구성"""

        difficulty_map = {
            "easy": "쉬움 (기본 개념 이해)",
            "medium": "보통 (응용 및 분석)",
            "hard": "어려움 (심화 분석 및 종합)"
        }

        type_instructions = {
            "multiple_choice": "4개 선택지가 있는 객관식",
            "short_answer": "간단한 주관식 (1-2문장 답변)",
            "fill_blank": "빈칸 채우기",
            "true_false": "참/거짓"
        }

        prompt = f"""
다음 PDF 문서 내용을 바탕으로 {num_questions}개의 퀴즈 문제를 생성하세요.

**문서 내용:**
{context[:4000]}

**생성 조건:**
- 난이도: {difficulty_map.get(difficulty, '보통')}
- 문제 수: {num_questions}개
- 문제 유형: {', '.join([type_instructions.get(qt, qt) for qt in question_types])}
{f"- 집중 토픽: {', '.join(topics)}" if topics else ""}

**중요 규칙:**
1. 반드시 주어진 문서 내용에 기반하여 문제 생성
2. 정답은 문서에서 명확히 찾을 수 있어야 함
3. 객관식은 정답 1개 + 그럴듯한 오답 3개
4. 각 문제마다 해설 포함
5. 문제 간 중복 피하기

다음 JSON 형식으로 응답:
{{
    "questions": [
        {{
            "question": "문제 내용",
            "question_type": "multiple_choice|short_answer|fill_blank|true_false",
            "options": ["선택지1", "선택지2", "선택지3", "선택지4"],  // 객관식만
            "correct_answer": "정답",
            "explanation": "해설",
            "difficulty": "{difficulty}",
            "topic": "관련 토픽"
        }}
    ]
}}
"""
        return prompt

    def _parse_quiz_response(self, response_text: str) -> Dict[str, Any]:
        """OpenAI 응답 파싱"""
        try:
            # JSON 부분만 추출
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("JSON 형식을 찾을 수 없습니다")

            json_text = response_text[start_idx:end_idx]
            return json.loads(json_text)

        except Exception as e:
            logger.error(f"응답 파싱 실패: {e}")
            # 파싱 실패 시 기본 구조 반환
            return {"questions": []}


class AnthropicLLMService(BaseLLMService):
    """Anthropic Claude 기반 LLM 서비스 (미래 확장용)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        logger.info("Anthropic LLM 서비스 (준비 중)")

    def generate_quiz(self, context: str, num_questions: int, difficulty: str, question_types: List[str], topics: Optional[List[str]] = None) -> Dict[str, Any]:
        return {"questions": [], "success": False, "error": "Anthropic 미구현"}

    def extract_topics(self, context: str) -> List[str]:
        return ["일반"]

    def validate_question_quality(self, question_data: Dict[str, Any]) -> bool:
        return True


class KoreanLocalLLMService(BaseLLMService):
    """한국어 로컬 LLM 서비스 (미래 확장용)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        logger.info("한국어 로컬 LLM 서비스 (준비 중)")

    def generate_quiz(self, context: str, num_questions: int, difficulty: str, question_types: List[str], topics: Optional[List[str]] = None) -> Dict[str, Any]:
        return {"questions": [], "success": False, "error": "한국어 로컬 모델 미구현"}

    def extract_topics(self, context: str) -> List[str]:
        return ["일반"]

    def validate_question_quality(self, question_data: Dict[str, Any]) -> bool:
        return True


class LLMFactory:
    """LLM 팩토리 클래스"""

    _services = {
        LLMProvider.OPENAI: OpenAILLMService,
        LLMProvider.ANTHROPIC: AnthropicLLMService,
        LLMProvider.KOREAN_LOCAL: KoreanLocalLLMService,
    }

    @classmethod
    def create_llm(cls, config: LLMConfig) -> BaseLLMService:
        """LLM 서비스 인스턴스 생성"""

        if config.provider not in cls._services:
            raise ValueError(f"지원하지 않는 LLM 제공업체: {config.provider}")

        service_class = cls._services[config.provider]
        return service_class(config)

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """사용 가능한 LLM 제공업체 목록"""
        return [provider.value for provider in cls._services.keys()]

    @classmethod
    def create_openai_gpt4o_mini(cls, api_key: Optional[str] = None, language: str = "ko") -> BaseLLMService:
        """OpenAI GPT-4o-mini 퀵 생성"""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            api_key=api_key,
            temperature=0.7,
            max_tokens=2000,
            language=language
        )
        return cls.create_llm(config)

    @classmethod
    def create_korean_local_model(cls, model_name: str = "kullm-polyglot-12.8b-v2") -> BaseLLMService:
        """한국어 로컬 모델 퀵 생성 (미래용)"""
        config = LLMConfig(
            provider=LLMProvider.KOREAN_LOCAL,
            model_name=model_name,
            language="ko"
        )
        return cls.create_llm(config)


# 전역 기본 LLM 서비스 (싱글톤 패턴)
_default_llm_service: Optional[BaseLLMService] = None


def get_default_llm_service() -> BaseLLMService:
    """기본 LLM 서비스 반환"""
    global _default_llm_service

    if _default_llm_service is None:
        settings = get_settings()
        _default_llm_service = LLMFactory.create_openai_gpt4o_mini(
            api_key=settings.OPENAI_API_KEY
        )
        logger.info("기본 LLM 서비스 초기화: OpenAI GPT-4o-mini")

    return _default_llm_service


def set_default_llm_service(service: BaseLLMService):
    """기본 LLM 서비스 설정"""
    global _default_llm_service
    _default_llm_service = service
    logger.info(f"기본 LLM 서비스 변경: {service.model_name}")


if __name__ == "__main__":
    # 테스트 코드
    print("=== LLM Factory 테스트 ===")

    # OpenAI 서비스 생성
    try:
        openai_service = LLMFactory.create_openai_gpt4o_mini()
        print(f"OpenAI 서비스 생성 성공: {openai_service.model_name}")
    except Exception as e:
        print(f"OpenAI 서비스 생성 실패: {e}")

    # 사용 가능한 제공업체 출력
    providers = LLMFactory.get_available_providers()
    print(f"사용 가능한 LLM 제공업체: {providers}")