# Python 3.11 기반 이미지 사용
FROM python:3.11-slim

# 필수 패키지 설치 (OpenGL 및 glib 포함)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 의존성 설치
COPY /ml_backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ML Backend 코드 복사
COPY /ml_backend .

# ML Backend 실행
CMD ["python", "app.py"]