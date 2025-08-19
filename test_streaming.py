#!/usr/bin/env python3
"""
Redis 스트림을 통한 문제 생성 스트리밍 테스트 스크립트
"""
import asyncio
import json
import time
from src.common.redis.connect import (
    push_quiz_batch_to_stream,
    push_quiz_completion_to_stream,
    push_quiz_error_to_stream,
    get_quiz_stream_messages
)

async def test_quiz_streaming():
    """문제 생성 스트리밍 테스트"""
    
    # 테스트용 요청 ID
    request_id = "test-request-123"
    
    print(f"🚀 문제 생성 스트리밍 테스트 시작: {request_id}")
    
    # 1. 시작 알림 전송
    print("📡 1. 시작 알림 전송...")
    await push_quiz_batch_to_stream(
        request_id=request_id,
        batch_num=0,
        questions=None,  # 문제가 아직 생성되지 않았으므로 None
        total_batches=1,
        status="started",
        metadata={
            "file_id": "test-file.pdf",
            "num_questions": 5,
            "difficulty": "medium",
            "question_type": "multiple_choice"
        }
    )
    
    # 2. 전처리 완료 알림 전송
    print("📡 2. 전처리 완료 알림 전송...")
    await push_quiz_batch_to_stream(
        request_id=request_id,
        batch_num=1,
        questions=None,  # 문제가 아직 생성되지 않았으므로 None
        total_batches=3,
        status="preprocessing_completed",
        metadata={
            "summary": "테스트 문서 요약...",
            "topics_count": 3,
            "keywords_count": 5
        }
    )
    
    # 3. 문제 생성 시작 알림 전송
    print("📡 3. 문제 생성 시작 알림 전송...")
    await push_quiz_batch_to_stream(
        request_id=request_id,
        batch_num=2,
        questions=None,  # 문제가 아직 생성되지 않았으므로 None
        total_batches=3,
        status="generation_started",
        metadata={
            "target_questions": 6,
            "batch_size": 3,
            "total_batches": 2
        }
    )
    
    # 4. 배치 1 완료 (3개 문제)
    print("📡 4. 배치 1 완료 알림 전송...")
    test_questions_1 = [
        {
            "id": 1,
            "question": "테스트 문제 1",
            "choices": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "테스트 설명 1"
        },
        {
            "id": 2,
            "question": "테스트 문제 2",
            "choices": ["A", "B", "C", "D"],
            "correct_answer": "B",
            "explanation": "테스트 설명 2"
        },
        {
            "id": 3,
            "question": "테스트 문제 3",
            "choices": ["A", "B", "C", "D"],
            "correct_answer": "C",
            "explanation": "테스트 설명 3"
        }
    ]
    
    await push_quiz_batch_to_stream(
        request_id=request_id,
        batch_num=3,
        questions=test_questions_1,
        total_batches=4,
        status="batch_completed",
        metadata={
            "batch_quality_score": 0.85
        }
    )
    
    # 5. 배치 2 완료 (2개 문제)
    print("📡 5. 배치 2 완료 알림 전송...")
    test_questions_2 = [
        {
            "id": 4,
            "question": "테스트 문제 4",
            "choices": ["A", "B", "C", "D"],
            "correct_answer": "D",
            "explanation": "테스트 설명 4"
        },
        {
            "id": 5,
            "question": "테스트 문제 5",
            "choices": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "테스트 설명 5"
        }
    ]
    
    await push_quiz_batch_to_stream(
        request_id=request_id,
        batch_num=4,
        questions=test_questions_2,
        total_batches=4,
        status="batch_completed",
        metadata={
            "batch_quality_score": 0.90
        }
    )
    
    # 6. 완료 알림 전송
    print("📡 6. 완료 알림 전송...")
    all_questions = test_questions_1 + test_questions_2
    await push_quiz_completion_to_stream(
        request_id=request_id,
        total_questions=len(all_questions),
        final_questions=all_questions,
        metadata={
            "total_time": 15.5,
            "avg_quality_score": 0.875,
            "failed_batches": 0
        }
    )
    
    print("✅ 모든 테스트 메시지 전송 완료!")
    
    # 잠시 대기 후 스트림 메시지 조회
    print("⏳ 3초 대기 후 스트림 메시지 조회...")
    await asyncio.sleep(3)
    
    # 스트림 메시지 조회
    print("📥 스트림 메시지 조회...")
    messages = await get_quiz_stream_messages(request_id, count=20)
    
    print(f"📊 조회된 메시지 수: {len(messages)}")
    for i, msg in enumerate(messages):
        print(f"\n--- 메시지 {i+1} ---")
        print(f"ID: {msg['message_id']}")
        print(f"데이터: {json.dumps(msg['data'], indent=2, ensure_ascii=False)}")
        
        # questions 데이터가 제대로 파싱되었는지 확인
        if 'questions' in msg['data'] and msg['data']['questions']:
            if isinstance(msg['data']['questions'], list):
                print(f"✅ questions 파싱 성공: {len(msg['data']['questions'])}개 문제")
            else:
                print(f"❌ questions 파싱 실패: {type(msg['data']['questions'])}")

async def test_error_streaming():
    """에러 상황 테스트"""
    
    request_id = "test-error-456"
    print(f"\n🚨 에러 상황 테스트: {request_id}")
    
    # 에러 알림 전송
    await push_quiz_error_to_stream(
        request_id=request_id,
        error_message="테스트 에러: OpenAI API 호출 실패",
        batch_num=2,
        metadata={
            "file_id": "error-file.pdf",
            "error_code": "API_ERROR"
        }
    )
    
    print("✅ 에러 메시지 전송 완료!")

if __name__ == "__main__":
    print("🧪 Redis 스트림 문제 생성 테스트 시작")
    
    # 테스트 실행
    asyncio.run(test_quiz_streaming())
    asyncio.run(test_error_streaming())
    
    print("\n🎉 모든 테스트 완료!")
