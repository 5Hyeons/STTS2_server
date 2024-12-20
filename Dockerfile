# Nvidia CUDA 12.3.2 with cuDNN 9 Runtime Ubuntu 20.04 기반 이미지 사용
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu20.04

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 Python 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 설치
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

# 컨테이너 시작 시 기본 명령어 설정 (필요에 따라 수정 가능)
CMD ["bash"]