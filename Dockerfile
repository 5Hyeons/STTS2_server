# Nvidia CUDA 12.5.1 with cuDNN on Ubuntu 20.04
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

# 비대화 모드 설정
ENV DEBIAN_FRONTEND=noninteractive

#----------------------------------------------------
# 1) 시스템 패키지 업데이트 및 Python 3.11 설치 준비
#----------------------------------------------------
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Python 3.11 설치
RUN apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv

# 기존 python3 -> python3.11 연결 (선택)
# 꼭 update-alternatives를 쓰지 않고 심볼릭 링크를 걸어도 됩니다.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ensurepip로 pip 설치 (Python 3.11 전용)
RUN python3.11 -m ensurepip --upgrade
RUN python3.11 -m pip install --upgrade pip

#----------------------------------------------------
# 2) 작업 디렉토리 설정
#----------------------------------------------------
WORKDIR /app

#----------------------------------------------------
# 3) requirements.txt 복사 및 의존성 설치
#    (여기서 python3.11을 사용하도록 명시)
#----------------------------------------------------
COPY requirements.txt .

RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

#----------------------------------------------------
# 추가로 필요한 파일/명령이 있으면 COPY/RUN 등으로 추가
#----------------------------------------------------

#----------------------------------------------------
# 4) 컨테이너 시작 시 기본 명령어 설정
#----------------------------------------------------
CMD ["python3", "server_onnx.py"]