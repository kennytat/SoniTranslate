# tasks.py
from celery import Celery
from speech_to_text import STTClient
app = Celery('routed_tasks',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/1')


# Define task routing
app.conf.task_routes = {
    'tasks.stt': {'queue': 'queue_stt'},
    'tasks.vtts': {'queue': 'queue_vtts'},
    'tasks.xtts': {'queue': 'queue_xtts'}
}

stt_client = None

# Tasks
@app.task(name='tasks.stt')
def stt(audio_wav, language, batch_size, chunk_size):
    global stt_client
    if stt_client is None:
        stt_client = STTClient(language)
    stt_client.init_model(language)
    result = stt_client.transcribe(audio_wav, batch_size, chunk_size)
    return result

@app.task(name='tasks.vtts')
def vtts():
    return "Processing in vtts"
  
@app.task(name='tasks.xtts')
def xtts():
    return "Processing in xtts"
