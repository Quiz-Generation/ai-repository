# Python 3.12 베이스 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (PDF 처리 라이브러리용)
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app/ ./app/
COPY data/ ./data/

# 로그 디렉토리 생성
RUN mkdir -p /app/logs

# 데이터 디렉토리 권한 설정
RUN chmod -R 755 /app/data

# 포트 노출
EXPOSE 7000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7000/health || exit 1

# 애플리케이션 실행
CMD ["python", "app/main.py"]