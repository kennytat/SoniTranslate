# tasks.py
from celery import Celery
from speech_to_text import STTClient
from vietTTS.vietTTS import VietTTS
from utils.tts_utils import piper_tts
from utils.xtts import XTTS
    
app = Celery('routed_tasks',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/1')

# Define task routing
app.conf.task_routes = {
    'tasks.stt': {'queue': 'queue_stt'},
    'tasks.diarization': {'queue': 'queue_diarization'},
    'tasks.vtts': {'queue': 'queue_vtts'},
    'tasks.xtts': {'queue': 'queue_xtts'},
    'tasks.ptts': {'queue': 'queue_ptts'}
}

stt_client = None
vtts_client = None
xtts_client = None

# Tasks
@app.task(name='tasks.stt')
def stt(audio_wav, language, batch_size, chunk_size):
    global stt_client
    if stt_client is None:
        stt_client = STTClient(language)
    stt_client.init_model(language)
    result = stt_client.transcribe(audio_wav, batch_size, chunk_size)
    return result

# Tasks
@app.task(name='tasks.diarization')
def diarize(audio_wav, transcript, min_speakers, max_speakers):
    global stt_client
    if stt_client is None:
        stt_client = STTClient()
    result = stt_client.diarize(audio_wav, transcript, min_speakers, max_speakers)
    return result

@app.task(name='tasks.ptts')
def ptts(tts_text, tts_voice, tts_speed, filename):
    result = piper_tts(tts_text, tts_voice, tts_speed, filename)
    return result
  
@app.task(name='tasks.vtts')
def vtts(tts_text, filename, tts_voice, tts_speed):
    global vtts_client
    if vtts_client is None:
        vtts_client = VietTTS()
    result = vtts_client.text_to_speech(tts_text, filename, tts_voice, tts_speed)
    return result
  
@app.task(name='tasks.xtts')
def xtts(tts_text, filename, tts_voice, tts_speed, language):
    global xtts_client
    if xtts_client is None:
        xtts_client = XTTS()
    result = stt_client.text_to_speech(tts_text, filename, tts_voice, tts_speed, language)
    return result