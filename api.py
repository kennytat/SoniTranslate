from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends, status, Security
from fastapi.responses import FileResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import whisperx
import torch
from text_to_speech import TTSClient, voice_conversion
from pydantic import BaseModel
import time
from utils.tts_utils import edge_tts_voices_list, piper_tts_voices_list
from natsort import natsorted
from utils.language_configuration import LANGUAGES, EXTRA_ALIGN, INVERTED_LANGUAGES
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from typing import Annotated
load_dotenv()

## ----- Configuration -----
temp_dir = os.getenv("APP_TEMP_DIR", os.path.join(tempfile.gettempdir(), "vgm-translate"))
if torch.cuda.is_available():
    device = "cuda"
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    torch.cuda.set_device(int(CUDA_VISIBLE_DEVICES.split(',')[0]))
else:
    device = "cpu"
compute_type = "float32" if device == "cpu" else "float16"
input_language = os.getenv("INPUT_LANG", "en")
output_language = os.getenv("OUTPUT_LANG", "vi")
tts_method = os.getenv("TTS_METHOD", "VietTTS")
basic_auth_admin = os.getenv("BASIC_AUTH_ADMIN", "admin")
basic_auth_password = os.getenv("BASIC_AUTH_PASSWORD", "password")
## ----- Configuration - end -----

def current_milli_time():
    return round(time.time() * 1000)

def get_tts_list(method=tts_method, language="vi"):
  print("method::", method, language)
  match method:
    case 'VietTTS':
      list_vtts = natsorted([voice for voice in os.listdir(os.path.join("model","vits")) if os.path.isdir(os.path.join("model","vits", voice))], key=lambda x: (x.count(os.sep), os.path.dirname(x), os.path.basename(x)))
      list_tts = list_vtts
    case 'EdgeTTS':
      list_etts = edge_tts_voices_list()
      list_tts = [ x for x in list_etts if x.startswith(LANGUAGES[language])]
    case 'PiperTTS':
      list_ptts = piper_tts_voices_list()
      list_tts = [ x for x in list_ptts if x.startswith(LANGUAGES[language])]
    case 'XTTS':
      list_xtts = natsorted([voice for voice in os.listdir(os.path.join("model","viXTTS","voices"))], key=lambda x: (x.count(os.sep), os.path.dirname(x), os.path.basename(x)))
      list_tts = list_xtts
    case _:
      list_tts = ['default']
  return list_tts

security = HTTPBasic()

def get_current_username(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    correct_username = secrets.compare_digest(credentials.username, basic_auth_admin)
    correct_password = secrets.compare_digest(credentials.password, basic_auth_password)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
  
class TTSDict(BaseModel):
    text: str
    language: str
    voice: str
     
class Whisper:
    def __init__(self, whisper_model="", device="", compute_type=compute_type, language=None):
        self.model = whisperx.load_model(
            whisper_arch=whisper_model,
            device=device,
            compute_type=compute_type,
            language=language,
            )

    def stt(self, file_path="", batch_size=16, chunk_size=5):
        try:
          audio_bytes = whisperx.load_audio(file_path)
          result = self.model.transcribe(audio_bytes, batch_size=batch_size, chunk_size=chunk_size, print_progress=True)
          speech = ", ".join([ segment["text"].strip() for segment in result["segments"]])
          return speech
        except Exception as e:
          print('Error stt::', e)
          return ""
        
stt_client = Whisper(whisper_model="large-v2", device=device, language=input_language)
tts_client = TTSClient()
tts_client.init_tts_client(client=tts_method)

def speech_to_text(file_path) -> str:
		return stt_client.stt(file_path=file_path)

def text_to_audio(tts_text="", tts_voice="", tts_speed=1, language="vi", t2s_method=tts_method):
    file_path = os.path.join(temp_dir, f"{current_milli_time()}.wav")
    tts_client.make_voice_gradio(tts_text=tts_text, tts_voice=tts_voice, tts_speed=tts_speed, filename=file_path, language=language, t2s_method=t2s_method)
    return file_path

## Start fast api server  
app = FastAPI()
# Add middleware to manage sessions
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")
origins = [ "*" ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/{_:path}")
async def catch_all(request: Request):
    return HTTPException(status_code=500, detail=f"Method not support!!")

@app.post("/stt")
async def stt(file: UploadFile = File(...), username: str = Depends(get_current_username)):
    try:
        contents = await file.read()
        tmp_file_path = os.path.join(temp_dir, file.filename)
        with open(tmp_file_path, "wb") as temp_file:
            temp_file.write(contents)
        result_text = speech_to_text(file_path=tmp_file_path)
        os.unlink(tmp_file_path)
        return {"text": result_text}                                   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
             
@app.post("/tts")
async def tts(body: TTSDict, username: str = Depends(get_current_username)) -> FileResponse:
		try:
			file_path = text_to_audio(tts_text=body.text, tts_voice=body.voice, tts_speed=1, language=body.language, t2s_method=tts_method)
			return FileResponse(file_path)
		except Exception as e:
				raise HTTPException(status_code=500, detail=f"An error occurred: {e}") 

@app.post("/voice")
async def voice_list(username: str = Depends(get_current_username)):
		try:
			list = get_tts_list(method=tts_method, language=output_language)
			return list
		except Exception as e:
				raise HTTPException(status_code=500, detail=f"An error occurred: {e}") 
          
if __name__ == "__main__":
    os.system(f'rm -rf {temp_dir}/*')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8321)
