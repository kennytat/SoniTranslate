import os
import sys
import re
import subprocess
from pathlib import Path
import atexit
import argparse
import shutil
import tempfile
import gradio as gr
import torch
import torchaudio

# torchaudio 2.0+ compatibility shims
if not hasattr(torchaudio, 'AudioMetaData'):
    torchaudio.AudioMetaData = type('AudioMetaData', (), {})
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile', 'sox']

# PyTorch 2.6+ defaults torch.load(weights_only=True), which breaks loading
# Pyannote/WhisperX checkpoints (omegaconf types). Lightning passes weights_only=None
# so we must force False whenever it's not explicitly False.
_torch_load_orig = torch.load
def _torch_load_patched(*args, **kwargs):
    if kwargs.get("weights_only") is not False:
        kwargs["weights_only"] = False
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_patched

from fastapi import FastAPI, HTTPException, Form, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
import asyncio
import sqlite3
from passlib.hash import bcrypt
import uvicorn
from itsdangerous import URLSafeSerializer
import aiosqlite
import sys
from utils.utils import new_dir_now, encode_filename, is_windows_path, convert_to_wsl_path, find_all_media_files, find_most_matching_prefix, youtube_download

if sys.platform == "darwin":
    import whisper_mlx as whisperx
else:
    import whisperx
from whisperx.utils import get_writer

from dotenv import load_dotenv
load_dotenv()
total_input = []
total_output = []

FAULT_TEXT = [
  "Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn"
]

# --- STT defaults (Whisper / faster-whisper): FP16 probe + VRAM-aware model size ---


def _cuda_vram_free_total():
    """(free_bytes, total_bytes) for CUDA device 0; falls back to total-only if mem_get_info fails."""
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        return int(free_b), int(total_b)
    except Exception:
        t = int(torch.cuda.get_device_properties(0).total_memory)
        return t, t


def _cuda_float16_ops_ok():
    """True if FP16 runs on device 0; synchronize so failures are not missed asynchronously."""
    try:
        with torch.cuda.device(0):
            x = torch.ones(1, dtype=torch.float16, device="cuda")
            y = x * x
            torch.cuda.synchronize()
            _ = y.item()
        return True
    except Exception:
        return False


# Total VRAM at or below this (bytes) is treated as 6GB-class → default Whisper to "small".
_WHISPER_TOTAL_MAX_FOR_SMALL_DEFAULT = 7 * 1024**3  # ≤7 GiB covers typical 6GB cards; 8GB stays above
# Min *free* VRAM to default the Whisper dropdown to large-v3 (prior code used ~9 GB total for FP16).
_WHISPER_LARGE_V3_MIN_FREE_FP16 = 9_000_000_000
# float32 roughly doubles weight RAM vs FP16; require more free VRAM before defaulting to large-v3.
_WHISPER_LARGE_V3_MIN_FREE_FP32 = 12_000_000_000


def _default_whisper_model_cuda(compute_type, free_vram_b, total_vram_b):
    if total_vram_b <= _WHISPER_TOTAL_MAX_FOR_SMALL_DEFAULT:
        return "small"
    need = (
        _WHISPER_LARGE_V3_MIN_FREE_FP32
        if compute_type == "float32"
        else _WHISPER_LARGE_V3_MIN_FREE_FP16
    )
    return "large-v3" if free_vram_b >= need else "medium"


# Check GPU
if torch.cuda.is_available():
    device = "cuda"
    list_compute_type = ['float16', 'float32']
    compute_type_default = 'float16'
    if not _cuda_float16_ops_ok():
        compute_type_default = 'float32'
        list_compute_type = ['float32']
    free_vram_b, total_vram_b = _cuda_vram_free_total()
    CUDA_MEM = total_vram_b
    whisper_model_default = _default_whisper_model_cuda(
        compute_type_default, free_vram_b, total_vram_b
    )
elif torch.backends.mps.is_available():
    device = "mps"
    list_compute_type = ['float32']
    compute_type_default = 'float32'
    whisper_model_default = 'large-v3'
else:
    device = "cpu"
    list_compute_type = ['float32']
    compute_type_default = 'float32'
    whisper_model_default = 'medium'

LANGUAGES = {
    'Automatic detection': 'Automatic detection',
    'Arabic (ar)': 'ar',
    'Cantonese (yue)': 'yue',
    'Chinese (zh)': 'zh',
    'Czech (cs)': 'cs',
    'Danish (da)': 'da',
    'Dutch (nl)': 'nl',
    'English (en)': 'en',
    'Finnish (fi)': 'fi',
    'French (fr)': 'fr',
    'German (de)': 'de',
    'Greek (el)': 'el',
    'Hebrew (he)': 'he',
    'Hungarian (hu)': 'hu',
    'Italian (it)': 'it',
    'Japanese (ja)': 'ja',
    'Korean (ko)': 'ko',
    'Persian (fa)': 'fa',
    'Polish (pl)': 'pl',
    'Portuguese (pt)': 'pt',
    'Russian (ru)': 'ru',
    'Spanish (es)': 'es',
    'Turkish (tr)': 'tr',
    'Ukrainian (uk)': 'uk',
    'Urdu (ur)': 'ur',
    'Vietnamese (vi)': 'vi',
    'Hindi (hi)': 'hi',
}

  
class ExitHooks(object):
    def __init__(self):
        self.exit_code = None
        self.exception = None

    def hook(self):
        self._orig_exit = sys.exit
        sys.exit = self.exit
        sys.excepthook = self.exc_handler

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args):
        self.exception = exc

hooks = ExitHooks()
hooks.hook()

gradio_temp_dir = os.getenv("GRADIO_TEMP_DIR", os.path.join(tempfile.gettempdir(), "gradio-vgm-stt"))
Path(gradio_temp_dir).mkdir(parents=True, exist_ok=True)
gradio_temp_processing_dir = os.path.join(gradio_temp_dir, "processing_dir")
        
class CONFIG():
    """Configurations"""
    # ckpt
    os_tmp = Path(os.getenv("APP_TEMP_DIR", os.path.join(tempfile.gettempdir(), "STT")))


class Whisper:
    def __init__(self, whisper_model="", device="", compute_type=compute_type_default, language='en'):
        self.current_model = whisper_model
        self.current_language = language
        self.model = whisperx.load_model(
            whisper_arch=whisper_model,
            device=device,
            compute_type=compute_type,
            language=None if language == 'Automatic detection' else language,
            ) if device != 'mps' else None

    def stt(self, file_path="", batch_size=16, chunk_size=24):
        try:
          if device == 'mps':
            result = whisperx.transcribe(file_path, path_or_hf_repo=f"mlx-community/whisper-{self.current_model}-mlx")
          else:
            audio_bytes = whisperx.load_audio(file_path)
            result = self.model.transcribe(audio_bytes, batch_size=batch_size, chunk_size=chunk_size, print_progress=True)
            result["segments"] = [seg for seg in result["segments"] if not any(ft in seg["text"] for ft in FAULT_TEXT)]
          return result
        except Exception as e:
          print('Error stt::', e)
          return ""
        
class STT():
  def __init__(self):
    self.stt_client = Whisper(whisper_model=whisper_model_default, device=device)
    self.local_input_dirs = []
    self.local_input_temp_pairs = []
    
  def handle_link_input(self, media_inputs, link_inputs):
    # print("media::", media_inputs)
    media_inputs = media_inputs if media_inputs and len(media_inputs) > 0 else []
    # print("links::", link_inputs)
    link_inputs = link_inputs.split(',')
    if link_inputs is not None and len(link_inputs) > 0 and link_inputs[0] != '':
      for url in link_inputs:
        url = url.strip().rstrip("/")
        # print('testing url::', url.startswith( 'https://www.youtube.com' ))
        ## Handle online link
        if url.startswith('https://'):
          try:
            media_info = yt_dlp.YoutubeDL().extract_info(url, download=False)
            download_path = f"{os.path.join(youtube_temp_dir, media_info['title'])}.mp4"
            youtube_download(url, download_path)
            media_inputs.append(download_path) 
          except Exception as e:
            print('Error downloading youtube video::', e)
            gr.Error(f"Error downloading from link: {url}")
        ## Handle local link
        else:
          osPath = url if not is_windows_path(url) else convert_to_wsl_path(url)
          if os.path.isfile(osPath):
            media_inputs.append(osPath)
          elif os.path.isdir(osPath):
            tmp_dir = os.path.join(gradio_temp_processing_dir, os.path.basename(osPath))
            print("tmp_dir::", tmp_dir, osPath)
            self.local_input_dirs.append(tmp_dir)
            files = find_all_media_files(osPath)
            print(f"media found in directory:: '{osPath}' | ", files)
            if len(files) > 0:
              for file in files:
                tmp_file = os.path.join(gradio_temp_processing_dir, re.sub(r'[\'\"]', '', file.replace(os.path.dirname(osPath),"").strip('/')))
                subprocess.run(["mkdir", "-p", os.path.dirname(tmp_file)], capture_output=True, text=True)
                if not os.path.exists(tmp_file):
                  subprocess.run(["touch", tmp_file], capture_output=True, text=True)
                  self.local_input_temp_pairs.append({
                    "origin": file,
                    "temp": tmp_file
                  })
                  media_inputs.append(tmp_file)
            else:
              gr.Warning(f"No media files found in: {osPath}")
    return media_inputs, ""
        
  def speech_to_text(
    self,
    input_files,
    whisper_model,
    LANGUAGE,
    batch_size,
    chunk_size
    ):
        
      output_dir_name = new_dir_now()
      output_dir_path = os.path.join(CONFIG.os_tmp, output_dir_name)
      Path(output_dir_path).mkdir(parents=True, exist_ok=True)
      print("stt called::",   input_files, whisper_model, LANGUAGE, batch_size, chunk_size)
      input_files = input_files or []
      file_list = [getattr(f, "name", f) if not isinstance(f, str) else f for f in input_files]
      results_list = []
      LANGUAGE = LANGUAGES[LANGUAGE]
      print("Start transcribing source language::")
      if self.stt_client.current_model != whisper_model or self.stt_client.current_language != LANGUAGE:
        self.stt_client = Whisper(whisper_model=whisper_model, device=device, language=LANGUAGE)
      global total_input
      global total_output
      total_input = input_files
      for index, file_path in enumerate(file_list):
        try:
          print('file_path::',file_path)
          tmp_dir = os.path.join(output_dir_path, encode_filename(file_path))
          archive_path = os.path.join(Path(output_dir_path).absolute(), os.path.splitext(os.path.basename(file_path))[0])
          if file_path.startswith(gradio_temp_processing_dir) and len(self.local_input_dirs) > 0:
            origin_entry = next((obj for obj in self.local_input_temp_pairs if obj['temp'] == file_path), None)
            origin_path = origin_entry['origin'] if origin_entry else None
            output_dir_path = os.path.splitext(origin_path)[0]
            tmp_dir = os.path.splitext(origin_path)[0]
            archive_path = Path(output_dir_path).absolute()
          output_format = "all"
          Path(tmp_dir).mkdir(parents=True, exist_ok=True)
          processing_file_path = origin_path if file_path.startswith(gradio_temp_processing_dir) else file_path
          result = self.stt_client.stt(file_path=processing_file_path, batch_size=batch_size, chunk_size=chunk_size)
          writer_args = {"highlight_words": False, "max_line_count": None, "max_line_width": None}
          writer = get_writer(output_format, tmp_dir)
          writer(result, file_path, writer_args)
          print(f'Done:: {index}/{len(file_list)} task::', file_path)
          print("archive_path::", archive_path)
          shutil.make_archive(archive_path, 'zip', tmp_dir)   
          results_list.append(f"{archive_path}.zip")
          total_output.append(f"{archive_path}.zip")
          
          # copy_output_dir = os.getenv('COPY_OUTPUT_DIR', '')
          # if copy_output_dir and os.path.isdir(copy_output_dir):
          #   subprocess.run(["cp", f"{archive_path}.zip", copy_output_dir], capture_output=True, text=True)

          ## Remove tmp files
          shutil.rmtree(tmp_dir, ignore_errors=True)
          os.remove(file_path)
        except:
            print("Skip error file while stt: {}".format(file_path))
      print("[DONE] {} tasks: {}".format(len(results_list), results_list))
      return results_list

  def web_interface(self, port):
    css = """
    .btn-active {background-color: "orange"}
    #logout_btn {
      align-self: self-end;
      width: 65px;
    }
    """
    app = gr.Blocks(title="VGM Speech To Text", theme=gr.themes.Default(), css=css)
    with app:
        with gr.Row():
          with gr.Column():
            gr.Markdown("# VGM Speech To Text")
          if os.getenv('ENABLE_AUTH', '') == "true":
            with gr.Column():
              gr.Button("Logout", link="/logout", size="sm", icon=None, elem_id="logout_btn")
        with gr.Tabs():
            with gr.Tab("STT"):
                with gr.Row():
                    with gr.Column():
                        input_files = gr.Files(label="Upload audio file(s)", file_types=["audio"])
                        with gr.Row():
                          link_input = gr.Textbox(label="OS Path",info="Example: M:\\warehouse\\video.mp3", placeholder="Path goes here, seperate by comma...", scale=5)        
                          link_btn = gr.Button("Submit", size="sm", scale=1)
                        with gr.Row():
                          WHISPER_MODEL = gr.Dropdown(['tiny', 'base', 'base.en', 'small','small.en', 'medium', 'medium.en', 'large-v3'], value=whisper_model_default, label="Whisper model",  scale=1)
                          LANGUAGE = gr.Dropdown(list(LANGUAGES.keys()), value='English (en)',label = 'Language', scale=1)
                        with gr.Row():
                          batch_size = gr.Slider(minimum=1, maximum=32, value=16, label="Batch size", step=1, scale=1)
                          chunk_size = gr.Slider(minimum=2, maximum=30, value=24, label="Chunk size", step=1, scale=1)
                    with gr.Column():
                        def update_output_list():
                          global total_input
                          global total_output
                          return total_output if len(total_output) < len(total_input) else []
                        with gr.Row():
                          files_output = gr.Files(label="PROGRESS BAR")
                        with gr.Row():
                          tmp_output = gr.Files(label="Audio Files Output", every=10, value=update_output_list) #gr.Video()                     
                        with gr.Row():
                          ## Clear Button
                          def reset_param():
                            global total_input
                            global total_output
                            total_input = []
                            total_output = []
                            return gr.update(label="PROGRESS BAR", visible=True), gr.update(label="Audio Files Output", visible=True)
                          clear_btn = gr.ClearButton(components=[input_files, files_output])
                          clear_btn.click(reset_param,[],[files_output,tmp_output])
                          def update_output_visibility():
                            return gr.update(label="Audio Files Output"),gr.update(visible=False)
                          btn = gr.Button(value="Generate!", variant="primary")
                          link_btn.click(self.handle_link_input, inputs=[input_files, link_input], outputs=[input_files, link_input])
                          btn.click(self.speech_to_text,
                                  inputs=[input_files, WHISPER_MODEL,LANGUAGE,batch_size, chunk_size],
                                  outputs=[files_output], concurrency_limit=1).then(
                          fn=update_output_visibility,
                          inputs=[],
                          outputs=[files_output,tmp_output]
                          )
    app.queue()
    return app

@atexit.register
def cleanup_tmp():
  if hooks.exit_code is not None:
      print("atexit call:: death by sys.exit(%d)" % hooks.exit_code)
  elif hooks.exception is not None:
      print("atexit call:: death by exception: %s" % hooks.exception)
  else:
      print("atexit call:: natural death")
      print("closing app:: cleanup_tmp")
      if os.path.exists( CONFIG.os_tmp): shutil.rmtree( CONFIG.os_tmp)
      if hasattr(sys, '_MEIPASS2') and sys._MEIPASS2 and os.path.exists(sys._MEIPASS2): shutil.rmtree(sys._MEIPASS2)



## Fast API Initialization
root = FastAPI()
# Secret key for session management
SECRET_KEY = "your-secret-key"
serializer = URLSafeSerializer(SECRET_KEY)
root.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
root.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Function to create SQLite connection
async def create_connection():
    return await aiosqlite.connect('db/auth.db')

async def init_database():
  conn = await create_connection()
  cursor = await conn.cursor()
  await cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
        )
    ''')
  await conn.commit()
  await conn.close()

# Function to fetch user ID by username
async def get_user_id(username):
    conn = await create_connection()
    cursor = await conn.cursor()
    await cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = await cursor.fetchone()
    await conn.close()
    return row
  
# Dependency to check if the user is logged in
def is_authenticated(request: Request):
    token = request.cookies.get("token")
    # print('is_authenticated:', token)
    if token:
        username = serializer.loads(token)
        user_id = asyncio.run(get_user_id(username))
        # print('is_authenticated:', user_id[0])
        if user_id:
            return user_id[0]
    return None
  
# Routes
@root.get("/", response_class=HTMLResponse)
async def home(request: Request):
    token = request.cookies.get("token")
    if token:
        username = serializer.loads(token)
        # Check if user exists in the database (session management)
        conn = await create_connection()
        cursor = await conn.cursor()
        await cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = await cursor.fetchone()
        if row:
            return RedirectResponse(url="/app")
    return RedirectResponse(url="/login")
  

@root.get("/signup", response_class=HTMLResponse)
async def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})
  
@root.post("/signup")
async def signup(username: str = Form(...), password: str = Form(...)):
    hashed_password = bcrypt.hash(password)
    try:
        conn = await create_connection()
        cursor = await conn.cursor()
        await cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        await conn.commit()
        await conn.close()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    return RedirectResponse(url="/", status_code=303)

@root.get("/login", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})
  
@root.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = await create_connection()
    cursor = await conn.cursor()
    await cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = await cursor.fetchone()
    await conn.close()
    if row and bcrypt.verify(password, row[0]):
        token = serializer.dumps(username)
        response = RedirectResponse(url="/app", status_code=303)
        response.set_cookie(key="token", value=token)
        return response
    error = "Wrong username or password"
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

@root.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("token")
    return response
      
if __name__ == "__main__":
    ## Download model if not exist
    # os.system('/bin/sh update_model.sh')
    ## Initialise app
    print("Application running on::", sys.platform)
    os.makedirs( CONFIG.os_tmp, exist_ok=True)
    os.system(f'rm -rf { CONFIG.os_tmp}/*')
    os.system(f'rm -rf /tmp/gradio-vgm/*')
    host = "localhost"
    port = 3100
    stt = STT()
    app = stt.web_interface(port)
    if os.getenv('ENABLE_AUTH', '') == "true":
      print("Starting Authentication:")
      root = gr.mount_gradio_app(root, app, path="/app", auth_dependency=is_authenticated)
      asyncio.run(init_database())
      uvicorn.run(root, host="0.0.0.0", port=port)
    else:
      auth_user = os.getenv('AUTH_USER', '')
      auth_pass = os.getenv('AUTH_PASS', '')
      app.launch(
        auth=(auth_user, auth_pass) if auth_user != '' and auth_pass != '' else None,
        footer_links=["gradio", "settings"],
        debug=False,
        inbrowser=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=port,
        share=False)
    sys.exit()