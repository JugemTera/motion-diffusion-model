# motion-diffusion-model (MDM) runtime image
# Base: CUDA 11.0 + cuDNN8 (matches pytorch=1.7.1 cuda11.0 in environment.yml).
# Ubuntu 20.04 with CPython 3.7 built from source, matching the "Python 3.7" the repo is
# tested on (README "Getting started").
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Headless matplotlib (environment.yml pulls pyqt for a GUI backend; not needed here).
    MPLBACKEND=Agg

# --- System dependencies ------------------------------------------------------
# git: required for `pip install git+https://github.com/openai/CLIP.git`.
# ffmpeg/libgl1/libglib2.0: needed by moviepy, trimesh rendering and the SMPL fitting code.
# wget/unzip/curl: used by prepare/download_*.sh.
# lib*-dev: build inputs for the CPython 3.7 source build below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        unzip \
        ca-certificates \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        libffi-dev \
        libsqlite3-dev \
        libreadline-dev \
        libncursesw5-dev \
        uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Python 3.7 ---------------------------------------------------------------
# Python 3.7 is EOL: focal ships 3.8 and the deadsnakes PPA has dropped 3.7, so build
# CPython 3.7 from source into /usr/local (becomes the default `python`/`python3`/`pip`).
ARG PYTHON_VERSION=3.7.17
RUN cd /tmp \
    && wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
    && tar xzf "Python-${PYTHON_VERSION}.tgz" \
    && cd "Python-${PYTHON_VERSION}" \
    && ./configure --prefix=/usr/local --enable-shared --with-ensurepip=install \
        LDFLAGS="-Wl,-rpath,/usr/local/lib" \
    && make -j"$(nproc)" \
    && make altinstall \
    && ln -s /usr/local/bin/python3.7 /usr/local/bin/python3 \
    && ln -s /usr/local/bin/python3.7 /usr/local/bin/python \
    && ln -s /usr/local/bin/pip3.7 /usr/local/bin/pip3 \
    && ln -s /usr/local/bin/pip3.7 /usr/local/bin/pip \
    && cd / && rm -rf /tmp/Python-*

# pip 24.0 is the last release supporting 3.7; keep setuptools<69 for the same reason and
# so legacy sdists (chumpy) still build.
RUN python -m pip install --upgrade "pip==24.0" "setuptools<69" wheel

WORKDIR /workspace

# --- Python dependencies ------------------------------------------------------
# PyTorch 1.7.1 + CUDA 11.0 wheels from the PyTorch index (same versions as environment.yml).
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu110 \
        torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2

# Everything else (pins mirror environment.yml; see docker-shell/requirements.txt).
COPY docker-shell/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

# chumpy last: it imports numpy in setup.py, so disable build isolation.
RUN python -m pip install --no-build-isolation "chumpy==0.70"

# README "Setup conda env" post-install steps.
RUN python -m spacy download en_core_web_sm \
    && python -m pip install git+https://github.com/openai/CLIP.git

# Quick sanity check that the core stack imported cleanly and CUDA build is present.
RUN python -c "import torch, torchvision, clip, spacy, smplx, chumpy, transformers; \
print('torch', torch.__version__, '| cuda build', torch.version.cuda)"

CMD ["/bin/bash"]
