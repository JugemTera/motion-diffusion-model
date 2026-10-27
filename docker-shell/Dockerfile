# motion-diffusion-model (MDM) runtime image (CUDA 12.1 build for Hopper GPUs such as the
# H200 on ABCI 3.0).
#
# environment.yml pins Python 3.7 + PyTorch 1.7.1 (CUDA 11.0). Those PyTorch builds carry
# kernels only up to sm_80 and fail on H100/H200 (sm_90) with
# "CUDA error: no kernel image is available for execution on the device", so this image
# uses the +cu121 build of PyTorch 2.4.1 on Ubuntu 22.04 / Python 3.10 instead. The repo
# code (MDM, Guo et al. evaluators, CLIP ViT-B/32, spaCy, DistilBERT for DiP) runs unchanged.
#
# Build (the docker-shell directory is the context; nothing is copied in):
#   bash docker-shell/build.sh                  -> mdm:cu121
# Singularity on ABCI (after `docker push jugemtera/mdm:cu121`):
#   singularity build mdm.sif docker://jugemtera/mdm:cu121
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# --- System dependencies ------------------------------------------------------
# python3 (3.10 on jammy), git for `git+https://` pip requirements (CLIP),
# ffmpeg/libgl1/libglib2.0 for moviepy / trimesh rendering / the SMPL fitting code,
# wget/curl/unzip for prepare/download_*.sh (gdown is installed via pip).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip build-essential git wget curl unzip ca-certificates ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/bin/python

# Modern pip; keep setuptools<70 so legacy sdists (chumpy) still build.
RUN python -m pip install --upgrade "pip<25" "setuptools<70" wheel

# --- Python dependencies ------------------------------------------------------
# PyTorch 2.4.1 (+cu121) wheels from the PyTorch index; numpy 1.x (chumpy and parts of the
# repo use np.float etc., and torch 2.4 is built against numpy 1.x).
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install "numpy==1.23.5"

# Everything the repo imports: training / sampling / evaluation, moviepy 1.x for
# `moviepy.editor` (removed in 2.0), decorator<5 for moviepy, blobfile (training_loop),
# spaCy + en_core_web_sm (text processing), transformers (DistilBERT encoder for DiP),
# chumpy last with build isolation off (its setup.py imports numpy).
RUN pip install scipy==1.10.1 matplotlib==3.7.5 tqdm pyyaml h5py pandas joblib scikit-learn==1.3.2 tensorboard wandb \
        ftfy regex "git+https://github.com/openai/CLIP.git" smplx==0.1.28 trimesh==3.23.5 \
        imageio==2.31.6 imageio-ffmpeg "moviepy==1.0.3" "decorator<5" blobfile gdown spacy==3.7.5 "transformers==4.44.2" \
    && pip install --no-build-isolation "chumpy==0.70" \
    && python -m spacy download en_core_web_sm

WORKDIR /workspace

CMD ["/bin/bash"]
