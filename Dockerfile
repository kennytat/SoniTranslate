FROM python:3.10.12

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y aria2 build-essential ffmpeg wget curl git vim cmake unzip

WORKDIR /app

ENV LD_LIBRARY_PATH=/usr/lib:/usr/lib64:/usr/local/lib:/usr/local/lib64/:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PATH="/usr/local/cuda/bin:/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ARG CACHE_DIR=/root/.cache/pip

RUN wget \
	https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \
	&& mkdir ~/.conda \
	&& bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b \
	&& rm -f Miniconda3-py310_23.5.2-0-Linux-x86_64.sh

RUN conda install -y nvidia/label/cuda-12.2.2::libcusparse
RUN conda install -y nvidia/label/cuda-12.2.2::cuda-toolkit
RUN mkdir -p /root/miniconda3/lib/python3.10/site-packages/data/checkpoints

COPY requirement*.txt ./
# Install dependencies with cache mounting
RUN --mount=type=cache,target=${CACHE_DIR} pip install --cache-dir=${CACHE_DIR} -r requirements_stt.txt
RUN --mount=type=cache,target=${CACHE_DIR} pip install --cache-dir=${CACHE_DIR} -r requirements_ttt.txt
RUN --mount=type=cache,target=${CACHE_DIR} pip install --cache-dir=${CACHE_DIR} -r requirements_tts.txt
RUN --mount=type=cache,target=${CACHE_DIR} pip install --cache-dir=${CACHE_DIR} -r requirements_extra.txt
RUN python -m spacy download en_core_web_sm
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121

RUN rm -rf /var/cache/apt/*

COPY . .
RUN chmod u+x run.sh

EXPOSE 6860
EXPOSE 8321
EXPOSE 7901
EXPOSE 3100

CMD /bin/bash run.sh --api --stt --tts --soni
## /bin/bash run.sh --api --stt --tts --soni