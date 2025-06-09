#!/usr/bin/env python3
"""
📝 문제 유형별 올바른 형태 검증 테스트
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lagnchain_fastapi_app'))

from app.schemas.quiz_schema import QuestionType

def validate_question_format(question_text: str, question_type: QuestionType) -> bool:
    """문제 형태가 올바른지 검증"""

    question_lower = question_text.lower()

    if question_type == QuestionType.TRUE_FALSE:
        # OX 문제는 단정적 서술이어야 함
        forbidden_patterns = ["다음 중", "보기에서", "선택하", "중에서"]
        if any(pattern in question_lower for pattern in forbidden_patterns):
            return False
        # 문장이 단정적으로 끝나야 함
        return question_text.rstrip().endswith(('.', '다.', '된다.', '이다.', '한다.'))

    elif question_type == QuestionType.MULTIPLE_CHOICE:
        # 객관식 문제는 선택을 요구하는 형태여야 함
        valid_patterns = ["다음 중", "무엇인가", "어떤 것", "올바른 것은", "맞는 것은", "해당하는"]
        return any(pattern in question_lower for pattern in valid_patterns)

    elif question_type == QuestionType.SHORT_ANSWER:
        # 주관식 문제는 설명/정의를 요구하는 형태여야 함
        forbidden_patterns = ["다음 중", "보기에서", "선택하", "중에서", "어떤 것"]
        if any(pattern in question_lower for pattern in forbidden_patterns):
            return False

        valid_patterns = ["설명하세요", "정의하세요", "무엇인가", "무엇인지", "차이점", "이란", "의미"]
        return any(pattern in question_lower for pattern in valid_patterns)

    return True

def test_question_validation():
    """문제 형태 검증 테스트"""

    print("📝 문제 유형별 형태 검증 테스트")
    print("=" * 50)

    # 테스트 케이스들
    test_cases = [
        # 올바른 OX 문제
        ("합성곱 신경망(CNN)은 이미지의 시각적 특징을 추출하는 데 사용된다.", QuestionType.TRUE_FALSE, True),
        ("딥러닝 모델은 항상 더 많은 데이터를 필요로 한다.", QuestionType.TRUE_FALSE, True),

        # 잘못된 OX 문제 (객관식 형태)
        ("다음 중 CNN의 설명으로 옳은 것은?", QuestionType.TRUE_FALSE, False),

        # 올바른 객관식 문제
        ("구글 코랩에서 GPU를 무료로 사용할 수 있는 최대 시간은 얼마인가요?", QuestionType.MULTIPLE_CHOICE, True),
        ("다음 중 딥러닝에서 손실 함수의 역할은 무엇인가요?", QuestionType.MULTIPLE_CHOICE, True),

        # 올바른 주관식 문제
        ("딥러닝에서 'Representation Learning'의 의미를 설명하세요.", QuestionType.SHORT_ANSWER, True),
        ("Gradient Descent의 기본 원리를 설명하세요.", QuestionType.SHORT_ANSWER, True),
        ("활성화 함수란 무엇인지 정의하세요.", QuestionType.SHORT_ANSWER, True),

        # 잘못된 주관식 문제 (객관식 형태)
        ("다음 중 머신러닝의 정의를 가장 잘 설명한 것은?", QuestionType.SHORT_ANSWER, False),
        ("보기에서 올바른 딥러닝 개념을 선택하세요.", QuestionType.SHORT_ANSWER, False),
    ]

    total_tests = len(test_cases)
    passed_tests = 0

    for i, (question, q_type, expected) in enumerate(test_cases, 1):
        result = validate_question_format(question, q_type)
        status = "✅ PASS" if result == expected else "❌ FAIL"

        print(f"\n테스트 {i}/{total_tests}: {status}")
        print(f"  문제: {question[:60]}...")
        print(f"  유형: {q_type.value}")
        print(f"  예상: {expected}, 실제: {result}")

        if result == expected:
            passed_tests += 1
        else:
            print(f"  ⚠️  검증 실패!")

    print(f"\n📊 테스트 결과: {passed_tests}/{total_tests} 통과")

    if passed_tests == total_tests:
        print("✅ 모든 테스트 통과! 검증 로직이 올바르게 작동합니다.")
    else:
        print("⚠️ 일부 테스트 실패. 검증 로직 개선이 필요합니다.")

    print("\n🎯 이제 프롬프트에 다음 가이드라인이 추가되었습니다:")
    print("  • OX 문제: 단정적 서술 (예: '~이다.', '~한다.')")
    print("  • 객관식: 선택을 요구하는 형태 (예: '다음 중 ~는?')")
    print("  • 주관식: 설명/정의를 요구하는 형태 (예: '~을 설명하세요.')")
    print("  • 주관식에서 '다음 중', '보기에서' 등 절대 사용 금지!")

if __name__ == "__main__":
    test_question_validation()