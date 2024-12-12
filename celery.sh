#!/bin/bash
device=$1
CUDA_VISIBLE_DEVICES=$device celery -A tasks worker --pool=solo --loglevel=INFO -Q queue_stt,queue_diarization,queue_vtts,queue_xtts,queue_ptts &
CUDA_VISIBLE_DEVICES=$device celery -A tasks worker --pool=solo --loglevel=INFO -Q queue_vtts,queue_xtts,queue_ptts &
