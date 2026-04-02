FROM python:3.10.12

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ffmpeg build-essential cmake unzip curl git vim cron \
        cudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements_stt.txt requirements_tts.txt requirements_ttt.txt ./
ARG CACHE_DIR=/root/.cache/pip
RUN --mount=type=cache,target=${CACHE_DIR} \
    pip install --upgrade pip && \
    pip install --cache-dir=${CACHE_DIR} so-vits-svc-fork==4.2.29 --no-deps && \
    for req in requirements_stt.txt requirements_ttt.txt requirements_tts.txt; do \
        pip install --cache-dir=${CACHE_DIR} -r ${req} --resume-retries 10; \
    done && \
    python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('punkt_tab')"

# Copy application code
COPY . .

# Cron: clean up temp files older than 60 min every hour
RUN echo "0 * * * * find /tmp -type f -mmin +60 -exec rm -f {} \;" > /etc/cron.d/cleanup-cron \
    && chmod 0644 /etc/cron.d/cleanup-cron

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Model share credentials (optional – prompted at runtime if missing)
ENV MODEL_SHAREID=""
ENV MODEL_PASSWORD=""

EXPOSE 6860 8321 7901 3100

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "app.py"]
