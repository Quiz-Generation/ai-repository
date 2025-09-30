#!/usr/bin/env python3
"""
Redis 스트림에서 실제 전송되는 문제 데이터를 확인하는 스크립트
"""
import asyncio
import json
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.common.redis.connect import get_quiz_stream_messages

async def check_redis_stream():
    """Redis 스트림에서 최근 메시지들을 확인"""
    print("🔍 Redis 스트림에서 최근 메시지들을 확인 중...")
    
    try:
        # 최근 10개 메시지 조회
        messages = await get_quiz_stream_messages(count=10)
        
        if not messages:
            print("❌ Redis 스트림에 메시지가 없습니다.")
            return
        
        print(f"✅ 총 {len(messages)}개의 메시지를 찾았습니다.\n")
        
        # 최근 3개 메시지만 상세 분석
        for i, msg in enumerate(messages[:3], 1):
            print(f"--- 최근 메시지 {i} ---")
            print(f"Message ID: {msg['message_id']}")
            
            data = msg['data']
            print(f"User IDX: {data.get('user_idx', 'N/A')}")
            print(f"Quizset IDX: {data.get('quizset_idx', 'N/A')}")
            print(f"Batch Num: {data.get('batch_num', 'N/A')}")
            print(f"Status: {data.get('status', 'N/A')}")
            
            # 문제 데이터가 있는 경우
            if 'questions' in data and data['questions']:
                questions = data['questions']
                print(f"문제 수: {len(questions)}개")
                
                # 각 문제의 제목만 출력 (중복 확인용)
                for j, question in enumerate(questions, 1):
                    question_text = question.get('question', 'N/A')
                    print(f"  {j}. {question_text[:80]}{'...' if len(question_text) > 80 else ''}")
            else:
                print("문제 데이터 없음")
            
            print()
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(check_redis_stream())
