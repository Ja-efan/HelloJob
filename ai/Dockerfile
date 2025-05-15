# Python 경량 이미지
FROM python:3.12-slim

# 작업 디렉토리 설정 (변경 없이 진행할것.)
WORKDIR /root/hellojob

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gcc \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | bash

# 의존성 파일만 먼저 복사
COPY requirements.txt ./

# 패키지 설치
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# 그 후에 소스 코드 복사
COPY main.py ./
COPY ./app ./app
COPY start.sh ./

# start script 실행 권한 부여
RUN chmod +x start.sh

# 포트 노출
EXPOSE 8000

# 컨테이너 시작 시 실행할 명령 추가
CMD ["/bin/bash", "-c", "/root/hellojob/start.sh"]