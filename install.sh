#!/bin/bash
# Usage: 
#   bash install.sh [ENV_NAME]
# 
# This script sets up the environment for the VGM-TTS project.
# - Optionally specify a conda environment name as the first argument (default: soni).
# - Installs CUDA and required system dependencies.
# - Installs Miniconda and sets up a new Python 3.10 environment.
# - Installs the required Python packages for TTS/STT/TTT.
# - Downloads additional language models.

ENV_NAME=${1:-soni}

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y aria2 build-essential ffmpeg wget curl git vim cmake unzip cifs-utils tmux cudnn9-cuda-12 nvidia-cuda-toolkit
wget \
	https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \
	&& mkdir ~/.conda \
	&& bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b \
	&& rm -f Miniconda3-py310_23.5.2-0-Linux-x86_64.sh

CONDA_BIN=~/miniconda3/bin
"${CONDA_BIN}"/conda init
"${CONDA_BIN}"/conda create -n "${ENV_NAME}" python=3.10.12 -y

ENV_BIN=~/miniconda3/envs/"${ENV_NAME}"/bin

"${ENV_BIN}"/pip install so-vits-svc-fork==4.2.29 --no-deps
for req in requirements_stt.txt requirements_ttt.txt requirements_tts.txt; do
	"${ENV_BIN}"/pip install -r ${req} --no-cache-dir --resume-retries 10 --force-reinstall
done
"${ENV_BIN}"/python -m spacy download en_core_web_sm
"${ENV_BIN}"/python -c "import nltk; nltk.download('punkt_tab')"
