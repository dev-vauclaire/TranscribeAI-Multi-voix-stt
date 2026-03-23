#------------------------------------------------------------
# Dockerfile: PyTorch + WhisperX pour RTX 40/50 Series
# Optimisé pour CUDA 12.8 et architecture Blackwell (sm_100/sm_120)
#------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Désactiver les interactions lors de l'install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install base dependencies
RUN apt update --fix-missing && \
apt install -y --no-install-recommends \
    software-properties-common \
    gpg-agent \
    ca-certificates \
    wget \
    curl \
    git \
    lsb-release \
    ffmpeg && \
apt clean && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update && \
    apt install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3.11-distutils \
        build-essential \
        pkg-config && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip
RUN wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

RUN pip install --upgrade pip

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy utilities
COPY requirements.txt .

# Installation du reste des requirements (WhisperX, FastAPI, etc.)
RUN pip install -r requirements.txt

COPY config.py .
COPY models.py .
COPY utils.py .
COPY main.py .
COPY audios/audio_diarization_test.mp3 ./audios/
COPY audios/test.flac ./audios/

EXPOSE 5001

# Lancement du micro-service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001", "--workers", "1"]