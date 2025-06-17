from setuptools import setup, find_packages


setup(
    name="src",
    version="0.1.0",
    author="devjun",
    author_email="jyporse@naver.com",
    description="A FastAPI application for document processing and quiz generation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # FastAPI 관련
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",

        # PDF 처리 라이브러리들
        "PyMuPDF==1.23.8",
        "pdfplumber==0.10.3",
        "PyPDF2==3.0.1",
        "pdfminer.six==20221105",

        # 언어 감지
        "langdetect==1.0.9",

        # 임베딩 및 텍스트 처리
        "sentence-transformers==2.2.2",
        "langchain==0.1.0",
        "langchain-community==0.0.10",
        "langchain-openai==0.0.5",
        "langgraph==0.0.20",

        # 벡터 DB 및 임베딩 관련
        "transformers==4.35.0",
        "torch==2.2.0",
        "numpy==1.26.4",

        # 벡터 데이터베이스 클라이언트
        "pymilvus[model]==2.3.4",

        # 기타 유틸리티
        "python-dotenv==1.0.0",
        "pydantic==2.5.0",
        "pydantic-settings==2.1.0",
        "python-jose[cryptography]==3.3.0",
        "passlib[bcrypt]==1.7.4",

        # Additional utilities
        "requests==2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.5b2",
            "isort>=5.9.3",
            "flake8>=3.9.2",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "lagnchain=app.main:main",
        ],
    },
)