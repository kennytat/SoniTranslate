# tasks.py
from celery import Celery
from speech_to_text import STTClient
from vietTTS.vietTTS import VietTTS
from utils.tts_utils import piper_tts
from utils.xtts import XTTS
import gc
import torch

app = Celery('routed_tasks',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/1')

# Define queue configuration
app.conf.task_queues = {
    'queue_stt': {
        'exchange': 'default',
        'routing_key': 'default',
    },
    'queue_diarization': {
        'exchange': 'default',
        'routing_key': 'default',
    },
    'queue_vtts': {
        'exchange': 'default',
        'routing_key': 'default',
    },
    'queue_xtts': {
        'exchange': 'default',
        'routing_key': 'default',
    },
    'queue_ptts': {
        'exchange': 'default',
        'routing_key': 'default',
    },
}

# Define task routing
app.conf.task_routes = {
    'tasks.stt': {'queue': 'queue_stt'},
    'tasks.diarization': {'queue': 'queue_diarization'},
    'tasks.vtts': {'queue': 'queue_vtts'},
    'tasks.xtts': {'queue': 'queue_xtts'},
    'tasks.ptts': {'queue': 'queue_ptts'}
}

class Celery():
  def __init__(self):
    self.name = ""
    self.client = None

celery_task = Celery()

# Tasks
@app.task(name='tasks.stt', max_retries=10, autoretry_for=(Exception,), default_retry_delay=5)
def stt(audio_wav, language, batch_size, chunk_size):
    global celery_task
    if celery_task.name != "stt" or celery_task.client is None:
      celery_task.client = None
      gc.collect(); torch.cuda.empty_cache()
      celery_task.name = "stt"
      celery_task.client = STTClient(language)
    celery_task.client.init_model(language)
    print("audio_wav::", audio_wav)
    result = celery_task.client.transcribe(audio_wav, batch_size, chunk_size)
    return result

# Tasks
@app.task(name='tasks.diarization', max_retries=10, autoretry_for=(Exception,), default_retry_delay=5)
def diarize(audio_wav, transcript, min_speakers, max_speakers):
    global celery_task
    if celery_task.name != "stt" or celery_task.client is None:
      celery_task.client = None
      gc.collect(); torch.cuda.empty_cache()
      celery_task.name = "stt"
      celery_task.client = STTClient()
    result = celery_task.client.diarize(audio_wav, transcript, min_speakers, max_speakers)
    return result

@app.task(name='tasks.ptts', max_retries=10, autoretry_for=(Exception,), default_retry_delay=5)
def ptts(tts_text, tts_voice, tts_speed, filename):
    global celery_task
    if celery_task.name != "ptts":
      celery_task.client = None
      gc.collect(); torch.cuda.empty_cache()
      celery_task.name = "ptts"
      celery_task.client = None
    result = piper_tts(tts_text, tts_voice, tts_speed, filename)
    return result
  
@app.task(name='tasks.vtts', max_retries=10, autoretry_for=(Exception,), default_retry_delay=5)
def vtts(tts_text, filename, tts_voice, tts_speed):
    global celery_task
    if celery_task.name != "vtts" or celery_task.client is None:
      celery_task.client = None
      gc.collect(); torch.cuda.empty_cache()
      celery_task.name = "vtts"
      celery_task.client = VietTTS()
    result = celery_task.client.text_to_speech(tts_text, filename, tts_voice, tts_speed)
    return result
  
@app.task(name='tasks.xtts', max_retries=10, autoretry_for=(Exception,), default_retry_delay=5)
def xtts(tts_text, filename, tts_voice, tts_speed, language):
    global celery_task
    if celery_task.name != "xtts" or celery_task.client is None:
      celery_task.client = None
      gc.collect(); torch.cuda.empty_cache()
      celery_task.name = "xtts"
      celery_task.client = XTTS()
    result = celery_task.client.text_to_speech(tts_text, filename, tts_voice, tts_speed, language)
    return result