#!/bin/bash
CUDA_VISIBLE_DEVICES=0 celery -A tasks worker --pool=solo --loglevel=INFO -Q queue_stt,queue_diarization,queue_vtts,queue_xtts,queue_ptts &
CUDA_VISIBLE_DEVICES=0 celery -A tasks worker --pool=solo --loglevel=INFO -Q queue_vtts,queue_xtts,queue_ptts &
CUDA_VISIBLE_DEVICES=1 celery -A tasks worker --pool=solo --loglevel=INFO -Q queue_stt,queue_diarization,queue_vtts,queue_xtts,queue_ptts &
CUDA_VISIBLE_DEVICES=1 celery -A tasks worker --pool=solo --loglevel=INFO -Q queue_vtts,queue_xtts,queue_ptts &
