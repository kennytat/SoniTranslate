from dotenv import load_dotenv
import os
import sys
import gc
from natsort import natsorted
from pathlib import Path
import atexit
from datetime import datetime
import argparse
import shutil
import tempfile
import librosa
import math
import torch  # isort:skip
import torch.multiprocessing as mp
torch.manual_seed(42)
import soundfile as sf
import json
import re
import unicodedata
from types import SimpleNamespace
import joblib
from joblib import Parallel, delayed
from pydub import AudioSegment
from queue import Queue
import gradio as gr
import numpy as np
import regex
from vietTTS.models import DurationNet, SynthesizerTrn
from vietTTS.utils import normalize, num_to_str, read_number, pad_zero, encode_filename, new_dir_now, file_to_paragraph, txt_to_paragraph, combine_wav_segment
# from vietTTS.upsample import Predictor
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
from text_to_speech import make_voice_gradio
from utils.tts_utils import edge_tts_voices_list, piper_tts_voices_list
from utils.language_configuration import LANGUAGES
from ovc_voice_main import OpenVoice
load_dotenv()


## Exit Hooks called when app terminating
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

class WavStruct():
    def __init__(self, wav_path, start_time):
        self.wav_path = wav_path
        self.start_time = start_time
        
class CONFIG():
    """Configurations"""
    # ckpt
    os_tmp = Path(os.path.join(tempfile.gettempdir(), "tts"))
    empty_wav = Path(os.path.join(f'{os_tmp}', "test.wav"))
    tts_ckpt_dir = Path(os.path.join(os.getcwd(), "model", "vits"))
    svc_ckpt_dir = Path(os.path.join(os.getcwd(), "model", "svc"))
    # salt = Path(os.path.join(os.getcwd(), "model","tts", "salt.salt"))
    # key = "^VGMAI*607#"

list_etts = edge_tts_voices_list()
list_gtts = ['default']
list_ptts = piper_tts_voices_list()
list_vtts = natsorted([voice for voice in os.listdir(os.path.join("model","vits")) if os.path.isdir(os.path.join("model","vits", voice))], key=lambda x: (x.count(os.sep), os.path.dirname(x), os.path.basename(x)))
list_xtts = natsorted([voice for voice in os.listdir(os.path.join("model","viXTTS","voices"))], key=lambda x: (x.count(os.sep), os.path.dirname(x), os.path.basename(x)))
list_svc = natsorted((["None"] + [voice for voice in os.listdir(os.path.join("model","svc")) if os.path.isdir(os.path.join("model","svc", voice))]), key=lambda x: (x.count(os.sep), os.path.dirname(x), os.path.basename(x)))
list_rvc = natsorted((["None"] + [voice for voice in os.listdir(os.path.join("model","rvc")) if voice.endswith('.pth')]), key=lambda x: (x.count(os.sep), os.path.dirname(x), os.path.basename(x))) 
list_ovc = natsorted((["None"] + [voice for voice in os.listdir(os.path.join("model","openvoice","target_voice")) if os.path.isdir(os.path.join("model","openvoice","target_voice", voice))]), key=lambda x: (x.count(os.sep), os.path.dirname(x), os.path.basename(x)))

class TTS():
  def __init__(self):
    self.upsampler = None
          
  def tts(self, text, output_file, tts_voice, speed = 1, desired_duration = 0, start_time = 0):
      # tts_voice_ckpt_dir = os.path.join(CONFIG.tts_ckpt_dir, tts_voice)
      # print("selected TTS voice:", tts_voice_ckpt_dir)
      try:
        print("Starting TTS {}".format(output_file), desired_duration, start_time)
        ### Get hifigan path
        # config_file = os.path.join(tts_voice_ckpt_dir,"config.json")
        # with open(config_file, "rb") as f:
        #   hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
        # sample_rate = hps.data.sampling_rate
        # print("tts text::", text)

        if re.sub(r'^sil\s+','',text).isnumeric():
            silence_duration = int(re.sub(r'^sil\s+','',text)) * 1000
            print("Got integer::", text, silence_duration) 
            print("\n\n\n ==> Generating {} seconds of silence at {}".format(silence_duration, output_file))
            second_of_silence = AudioSegment.silent(duration=silence_duration) # or be explicit
            second_of_silence = second_of_silence.set_frame_rate(16000)
            second_of_silence.export(output_file, format="wav")
        else:
          # duration_net, generator = self.load_models(tts_voice_ckpt_dir, hps)
          make_voice_gradio(text, tts_voice, speed, output_file, self.tts_lang, self.tts_method)
        
          ## For tts with timeline
          if desired_duration > 0:
            try:
              duration_true = desired_duration
              duration_tts = librosa.get_duration(path=output_file)

              # porcentaje
              porcentaje = duration_tts / duration_true
              print("change speed::", porcentaje, duration_tts, duration_true)
              # Smooth and round
              porcentaje = math.floor(porcentaje * 10000) / 10000
              porcentaje = 0.8 if porcentaje <= 0.8 else porcentaje + 0.005
              porcentaje = 1.5 if porcentaje >= 1.5 else porcentaje    
            except Exception as e:
              porcentaje = 1.0 
              print('An exception occurred:', e)
            # apply aceleration or opposite to the audio file in audio2 folder
            name, ext = os.path.splitext(output_file)
            tmp_file = f"{name}-tmp{ext}"
            os.system(f"ffmpeg -y -loglevel panic -i {output_file} -filter:a atempo={porcentaje} {tmp_file}")
            os.system(f"mv {tmp_file} {output_file}")
          gc.collect(); torch.cuda.empty_cache()
      except Exception as error:
        print("tts error::", text, "\n", error)
      return WavStruct(output_file, start_time)

  def upsampling(self, file):
    # if not self.upsampler:
    #   self.upsampler = Predictor()
    #   self.upsampler.setup(model_name="speech")
    # audio_data, sample_rate = sf.read(file.wav_path)
    # source_duration = len(audio_data) / sample_rate
    # data = self.upsampler.predict(
    #     file.wav_path,
    #     ddim_steps=50,
    #     guidance_scale=3.5,
    #     seed=42
    # )
    # ## Trim duration to match source duration
    # target_samples = int(source_duration * 48000)
    # sf.write(file.wav_path, data=data[:target_samples], samplerate=48000)
    return file
    
  def start_svc_voice(self, input_dir, model_dir):
    print("start svc_voice::", input_dir, model_dir)
    model_path = os.path.join(model_dir, "G.pth")
    config_path = os.path.join(model_dir, "config.json")
    output_dir = f'{input_dir}.out'
    os.system(f'svc infer -re -m {model_path} -c {config_path} {input_dir}')
    if os.path.exists(input_dir): shutil.rmtree(input_dir, ignore_errors=True)
    shutil.move(output_dir, input_dir)
    gc.collect(); torch.cuda.empty_cache()

  def start_ovc_voice(self, input_dir, tts_voice, open_voice):
    print("start open_voice::", input_dir, tts_voice, open_voice)
    output_dir = f"{input_dir}-out"
    os.system(f"mkdir -p {output_dir}")
    ov = OpenVoice()
    for file in sorted(Path(input_dir).glob("*.wav")):
      file_output = str(file).replace(input_dir, output_dir)
      print("openvoice::", file_output)
      ov.convert_voice(file, tts_voice, file_output, open_voice)
    del ov
    if os.path.exists(input_dir): shutil.rmtree(input_dir, ignore_errors=True)
    shutil.move(output_dir, input_dir)
    gc.collect(); torch.cuda.empty_cache()
      
  def synthesize(self, output_dir_name, input, is_file, speed, method):
      print("start synthesizing::", output_dir_name, input, is_file, speed)
      filepath = ""
      paragraphs = ""
      file_name_only = ""
      basename, ext = os.path.splitext(os.path.basename(input))
      if is_file:
        file_name_only = Path(basename)
        filepath = encode_filename(input)
        paragraphs = file_to_paragraph(input)
      else:
        filepath = "{}".format(new_dir_now())
        file_name_only = encode_filename(filepath)
        paragraphs = txt_to_paragraph(input)
        
      # print("paragraphs::", paragraphs)
      ### Put segments in temp dir for concatnation later
      tmp_dirname = os.path.join(CONFIG.os_tmp, output_dir_name, filepath)
      # print("filename::", filepath, paragraphs, file_name_only, tmp_dirname)
      Path(tmp_dirname).mkdir(parents=True, exist_ok=True)
      final_name = "{}.wav".format(file_name_only)
      final_output = os.path.join(CONFIG.os_tmp, output_dir_name, final_name)
      log_output = None
      print("Output Temp: ", final_output)
      temp_output = ''

      # process_list = []
      queue_list = Queue()
      # wav_list = []
      results = []
      for (no, para) in enumerate(paragraphs):
          # print("Processing para::", no, para)
          name = "{}.wav".format(pad_zero(no, 5))
          # print("Prepare normalized text: ", para.text)
          temp_output = os.path.join(tmp_dirname, name)
          print("paragraph::", para.text.strip(), temp_output, para.total_duration, para.start_time)
          queue_list.put((para.text.strip(), temp_output, para.total_duration, para.start_time))
          
      # print("Parallel processing {} tasks".format(len(process_list)))
      print("Queue list:: ", queue_list.qsize())
      CUDA_MEM = int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None
      N_JOBS = os.getenv('TTS_JOBS', round(CUDA_MEM*0.5/1000000000) if CUDA_MEM else 1)
      N_JOBS = N_JOBS if self.tts_method != "XTTS" else 1
      
      print("Start TTS:: concurrency =", N_JOBS)
      with joblib.parallel_config(backend="loky", prefer="threads", n_jobs=int(N_JOBS)):
        results = Parallel(verbose=100)(delayed(self.tts)(text, output_file, self.tts_voice, speed, total_duration, start_silence) for (text, output_file, total_duration, start_silence) in queue_list.queue)
      
      if os.getenv('UPSAMPLING_ENABLE', '') == "true":  
        print("Start Upsampling::")
        with joblib.parallel_config(backend="loky", prefer="threads", n_jobs=1):
          results = Parallel(verbose=100)(delayed(self.upsampling)(file) for (file) in results)
        self.upsampler = None; gc.collect(); torch.cuda.empty_cache()
      
      ## TTS Done - Start converting voice
      print("TTS Done::")
      if self.vc_method == "SVC":
        svc_voice_ckpt_dir = os.path.join(CONFIG.svc_ckpt_dir, self.vc_voice)
        print("Start Voice Convertion::")
        self.start_svc_voice(tmp_dirname, svc_voice_ckpt_dir)

      if self.vc_method == "OpenVoice":
        print("Start Voice Convertion::")
        self.start_ovc_voice(tmp_dirname, self.tts_voice, self.vc_voice)    
        
      ## Return join or split output files  
      if method == 'join':
        result_path, log_path = combine_wav_segment(results, final_output)
        print("combine_wav_segment result::", result_path, log_path)
        final_output = result_path
        log_output = log_path
      if method == 'split':
        archive_path = re.sub(r'\.wav$', '', final_output)
        shutil.make_archive(archive_path, 'zip', tmp_dirname)
        final_output = "{}.zip".format(archive_path)
      print("final_output_path::", final_output)
      
      ## Remove tmp files
      if os.path.exists(tmp_dirname): shutil.rmtree(tmp_dirname, ignore_errors=True)
      if input.startswith('/tmp'):
        os.remove(input)
      return (final_output, log_output)

  def speak(
    self,
    input_files,
    input_text,
    tts_lang,
    tts_voice,
    vc_voice,
    speed=1,
    method="join",
    tts_method="VietTTS",
    vc_method="None"
    ):
      self.tts_lang = LANGUAGES[tts_lang]
      self.tts_voice = tts_voice
      self.vc_voice = vc_voice
      self.tts_method = tts_method
      self.vc_method = vc_method
      output_dir_name = new_dir_now()
      output_dir_path = os.path.join(CONFIG.os_tmp, output_dir_name)
      Path(output_dir_path).mkdir(parents=True, exist_ok=True)
      print("start speak_fn:", tts_voice)
      results_list = []
      result_text = CONFIG.empty_wav
      logs_list = []
      ## Process input_text first
      if input_text:
        try:
            print('input_text::', input_text)
            output_temp_file, log_temp_file = self.synthesize(output_dir_name, input_text, False, speed, method)
            if log_temp_file:
              logs_list.append(log_temp_file)
            if method == 'join':
              result_text = output_temp_file
            if method == 'split':
              results_list.append(output_temp_file)
        except Exception as e:
            print("Skip error file while synthesizing input_text::", e)
      ## Process input_files     
      if input_files:
        print("got input files::",input_files)
        file_list = [f.name for f in input_files]
        for file_path in file_list:
            try:
                print('file_path::',file_path)
                output_temp_file, log_temp_file = self.synthesize(output_dir_name, file_path, True, speed, method)
                results_list.append(output_temp_file)
                if log_temp_file:
                  logs_list.append(log_temp_file)
            except:
                print("Skip error file while synthesizing doc: {}".format(file_path))
      print("[DONE] {} tasks: {}".format(len(results_list), results_list))
      return results_list, result_text, logs_list

  def refresh_model(self, tts_method):
    if tts_method == "SVC":
      vc_list = [voice for voice in os.listdir(os.path.join("model","svc")) if os.path.isdir(os.path.join("model","svc", voice))]
    if tts_method == "OpenVoice":
      vc_list = [voice for voice in os.listdir(os.path.join("model","openvoice","target_voice")) if os.path.isdir(os.path.join("model","openvoice","target_voice", voice))]
    return gr.update(choices=vc_list)
  
  def create_open_voice(self, file_path, model_name):
    ov = OpenVoice()
    ov.create_voice(file_path, model_name)
    gr.Info(f'Created voice: {model_name}')
    del ov
    return None, None
   
  def web_interface(self, port):
    css = """
    .btn-active {background-color: "orange"}
    #logout_btn {
      align-self: self-end;
      width: 65px;
    }
    """
    # title="VGM Text To Speech",
    # description = "A vietnamese text-to-speech by VGM speakers."
    app = gr.Blocks(title="VGM Text To Speech", theme=gr.themes.Default(), css=css)
    with app:
        with gr.Row():
          with gr.Column():
            gr.Markdown("# VGM Text To Speech")
          with gr.Column():
            gr.Button("Logout", link="/logout", size="sm", icon=None, elem_id="logout_btn", visible=True if os.getenv('ENABLE_AUTH', '') == "true" else False)
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Row():
                    with gr.Column():
                        input_files = gr.Files(label="Upload .doc|.docx|.txt|.srt file(s)", file_types=[".doc", ".docx", ".txt", ".srt"])
                        textbox = gr.Textbox(label="Text for synthesize")
                        with gr.Row():
                          tts_lang = gr.Dropdown(['Arabic (ar)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='Vietnamese (vi)',label = 'Target language', scale=1)
                          tts_voice = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker', visible=True, elem_id="tts_voice")
                          vc_voice = gr.Dropdown(choices=list_svc, value=list_svc[0], label='VC Speaker', visible=False, elem_id="vc_voice")
                        duration_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1, step=0.02, label='Speed')
                        method = gr.Radio(label="Method", value="join", choices=["join","split"])
                    with gr.Column():
                        files_output = gr.Files(label="Files Audio Output")
                        audio_output = gr.Audio(label="Text Audio Output", elem_id="tts-audio")
                        logs_output = gr.Files(label="Error Audio Logs")
                        with gr.Row():
                          clear_btn = gr.ClearButton([input_files,textbox,files_output,audio_output,logs_output], value="Refresh")
                          btn = gr.Button(value="Generate!", variant="primary")
            with gr.TabItem("Settings"):
                with gr.Column():
                  with gr.Accordion("T2S - VC Method", open=False):
                    with gr.Row():
                      tts_method = gr.Dropdown(["GTTS", "EdgeTTS", "PiperTTS","VietTTS","XTTS"], label='T2S', value="VietTTS", visible=True, elem_id="tts_method",interactive=True)
                      vc_method = gr.Dropdown(["None", "SVC", "RVC", "OpenVoice"], label='Voice Conversion', value="None", visible=True, elem_id="vc_method",interactive=True)

                      ## update t2s method
                      def update_t2s_list(method, language):
                        # print("method::", method, language, media_input)
                        match method:
                          case 'VietTTS':
                            list_tts = list_vtts
                          case 'EdgeTTS':
                            list_tts = [ x for x in list_etts if x.startswith(LANGUAGES[language])]
                          case 'PiperTTS':
                            list_tts = [ x for x in list_ptts if x.startswith(LANGUAGES[language])]
                          case 'XTTS':
                            list_tts = list_xtts
                          case _:
                            list_tts = list_gtts
                        return gr.update(choices=list_tts, value=list_tts[0])
                    tts_method.change(update_t2s_list, [tts_method, tts_lang], tts_voice)
                    tts_lang.change(update_t2s_list, [tts_method, tts_lang], tts_voice)
                  with gr.Accordion("Open Voice", visible=False) as open_voice_accordion:
                    ov_file = gr.File(label="Upload audio file", file_types=["audio"])
                    ov_name = gr.Textbox(label="Model Name")
                    ov_btn = gr.Button(value="Create Voice", variant="primary")
                    
                def update_voice_conversion(method):
                  visible = True if method != "None" else False
                  return  gr.update(visible=visible), gr.update(visible=method=='OpenVoice')
                vc_method.change(update_voice_conversion, [vc_method], [vc_voice, open_voice_accordion])    
                    
        ## Run function
        clear_btn.click(self.refresh_model, inputs=[tts_method], outputs=[vc_voice])
        ov_btn.click(self.create_open_voice, inputs=[ov_file, ov_name], outputs=[ov_file, ov_name])
        btn.click(self.speak,
                inputs=[input_files, textbox, tts_lang, tts_voice, vc_voice, duration_slider, method, tts_method, vc_method],
                outputs=[files_output, audio_output, logs_output], concurrency_limit=1)
                                              
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
      if sys._MEIPASS2 and os.path.exists(sys._MEIPASS2): shutil.rmtree(sys._MEIPASS2)
  sys.exit()


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
    CONFIG.empty_wav.touch(exist_ok=True)
    ## Set torch multiprocessing
    mp.set_start_method('spawn', force=True)
    host = "localhost"
    port = 7904
    tts = TTS()
    app = tts.web_interface(port)
    if os.getenv('ENABLE_AUTH', '') == "true":
      root = gr.mount_gradio_app(root, app, path="/app", auth_dependency=is_authenticated)
      asyncio.run(init_database())
      uvicorn.run(root, host="0.0.0.0", port=port)
    else:
      auth_user = os.getenv('AUTH_USER', '')
      auth_pass = os.getenv('AUTH_PASS', '')
      app.launch(
        auth=(auth_user, auth_pass) if auth_user != '' and auth_pass != '' else None,
        show_api=False,
        debug=False,
        inbrowser=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=port,
        share=False)   
    sys.exit()