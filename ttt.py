from dotenv import load_dotenv
import os
import sys
import glob
from pathlib import Path
import atexit
import argparse
import shutil
import tempfile
import json
import re
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from pydub import AudioSegment
from queue import Queue
import gradio as gr
# from vietTTS.models import DurationNet, SynthesizerTrn
from vietTTS.utils import encode_filename, new_dir_now, file_to_paragraph, txt_to_paragraph, combine_wav_segment
# from vietTTS.upsample import Predictor
from utils.utils import is_path, new_dir_now, save_texts_to_file, get_llm_models, segments_to_parquet
from utils.language_configuration import LANGUAGES
# from utils.logging_setup import logger
from translate_segments import translate_text, grammar_correction
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

class CONFIG():
    """Configurations"""
    # ckpt
    os_tmp = Path(os.getenv("APP_TEMP_DIR", os.path.join(tempfile.gettempdir(), "TTT")))
    gradio_temp_dir = os.getenv("GRADIO_TEMP_DIR", "/tmp/gradio-vgm")
    # salt = Path(os.path.join(os.getcwd(), "model","tts", "salt.salt"))
    # key = "^VGMAI*607#"

# Function to save settings to a JSON file
def save_settings(settings, filename='user_settings.json'):
  if settings:
    with open(filename, 'w') as f:
        json.dump(settings, f)

# Function to load settings from a JSON file
def load_settings(filename='user_settings.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

user_settings=load_settings()

get_local_storage = """
function() {
  globalThis.setStorage = (key, value) => {
    localStorage.setItem(key, JSON.stringify(value));
  };
  globalThis.getStorage = (key, value) => {
    return JSON.parse(localStorage.getItem(key));
  };

  const s2t_method = getStorage("s2t_method");
  const t2t_method = getStorage("t2t_method");
  const t2s_method = getStorage("t2s_method");
  const vc_method = getStorage("vc_method");
  const llm_url = getStorage("llm_url");
  const llm_model = getStorage("llm_model");
  const llm_temp = getStorage("llm_temp");
  const llm_k = getStorage("llm_k");
  const max_speakers = getStorage("max_speakers");

  const tts_voice00 = getStorage("tts_voice00");
  const tts_speed00 = getStorage("tts_speed00");
  const vc_voice00 = getStorage("vc_voice00");

  const tts_voice01 = getStorage("tts_voice01");
  const tts_speed01 = getStorage("tts_speed01");
  const vc_voice01 = getStorage("vc_voice01");

  const tts_voice02 = getStorage("tts_voice02");
  const tts_speed02 = getStorage("tts_speed02");
  const vc_voice02 = getStorage("vc_voice02");

  const tts_voice03 = getStorage("tts_voice03");
  const tts_speed03 = getStorage("tts_speed03");
  const vc_voice03 = getStorage("vc_voice03");

  const tts_voice04 = getStorage("tts_voice04");
  const tts_speed04 = getStorage("tts_speed04");
  const vc_voice04 = getStorage("vc_voice04");

  const tts_voice05 = getStorage("tts_voice05");
  const tts_speed05 = getStorage("tts_speed05");
  const vc_voice05 = getStorage("vc_voice05");

  const match_length = getStorage("match_length");
  const match_start = getStorage("match_start");

  const SOURCE_LANGUAGE = getStorage("SOURCE_LANGUAGE");
  const TARGET_LANGUAGE = getStorage("TARGET_LANGUAGE");
  
  const WHISPER_MODEL_SIZE = getStorage("WHISPER_MODEL_SIZE");
  const compute_type = getStorage("compute_type");
  const batch_size = getStorage("batch_size");
  const chunk_size = getStorage("chunk_size");
  
  return [
    t2t_method || "LLM",
    llm_url || "http://localhost:8081/v1",
    llm_model,
    llm_temp || 0.3,
    llm_k || 5,
    SOURCE_LANGUAGE || "English (en)",
    TARGET_LANGUAGE || "Vietnamese (vi)"
  ];
}
"""

class TTT():
  def __init__(self):
    pass
      
  def pre_process(self, output_dir_name, input, SOURCE_LANGUAGE, TARGET_LANGUAGE, progress=gr.Progress()):
      print("pre_process::", output_dir_name, input, self.t2t_method)
      filepath = ""
      paragraphs = ""
      file_name_only = ""
      basename, ext = os.path.splitext(os.path.basename(input))
      
      if is_path(input):
        file_name_only = Path(basename)
        filepath = encode_filename(input)
        paragraphs = file_to_paragraph(input)
      else:
        filepath = "{}".format(new_dir_now())
        file_name_only = encode_filename(filepath)
        paragraphs = txt_to_paragraph(input)
        ext = ".txt"
      
      Path(os.path.join(CONFIG.os_tmp, output_dir_name)).mkdir(parents=True, exist_ok=True)
      paragraphs = [{"text": para.text, "start": para.start, "end": para.end} for para in paragraphs]  
      print("paragraphs::", paragraphs)

      final_name = f"{file_name_only}-translated{ext}"
      final_output = os.path.join(CONFIG.os_tmp, output_dir_name, final_name)
      print("Output Temp: ", final_output)

      progress(0.10, desc="Translating...")
      translated_segments = translate_text(paragraphs, SOURCE_LANGUAGE, TARGET_LANGUAGE, self.t2t_method, self.llm_url, self.llm_model, self.llm_temp, self.llm_k)
      segments_to_parquet(translated_segments, f'{final_output}.parquet')
      progress(0.70, desc="Grammar correction...")
      paragraphs = grammar_correction(paragraphs, translated_segments, SOURCE_LANGUAGE, TARGET_LANGUAGE, self.llm_url, self.llm_model, self.llm_temp, self.llm_k)
      segments_to_parquet(paragraphs, f'{final_output}-correct.parquet')
      print("translated segments::", paragraphs)
      
      # Export translated text to file
      save_texts_to_file(paragraphs, final_output)
      if input.startswith('/tmp'):
        os.remove(input)
      return final_output

  def run(
    self,
    input_files,
    input_text,
    t2t_method, llm_url, llm_model, llm_temp, llm_k, SOURCE_LANGUAGE, TARGET_LANGUAGE
    ):
      self.t2t_method = t2t_method
      self.llm_url = llm_url
      self.llm_model = llm_model
      self.llm_temp = llm_temp
      self.llm_k = llm_k
      SOURCE_LANGUAGE = LANGUAGES[SOURCE_LANGUAGE]
      TARGET_LANGUAGE = LANGUAGES[TARGET_LANGUAGE]

      output_dir_name = new_dir_now()
      output_dir_path = os.path.join(CONFIG.os_tmp, output_dir_name)
      Path(output_dir_path).mkdir(parents=True, exist_ok=True)
      results_list = []
      ## Process input_text first
      if input_text:
        try:
            print('input_text::', input_text)
            output_temp_file = self.pre_process(output_dir_name, input_text, SOURCE_LANGUAGE, TARGET_LANGUAGE)
            results_list.append(output_temp_file)
        except Exception as e:
            print("Skip error file while translate input_text::", e)
      ## Process input_files     
      if input_files:
        print("got input files::",input_files)
        file_list = [f.name for f in input_files]
        for file_path in file_list:
            try:
                print('file_path::',file_path)
                output_temp_file = self.pre_process(output_dir_name, file_path, SOURCE_LANGUAGE, TARGET_LANGUAGE)
                results_list.append(output_temp_file)
            except:
                print("Skip error file while translate doc: {}".format(file_path))
      print("[DONE] {} tasks: {}".format(len(results_list), results_list))
      return results_list

    
  def web_interface(self, port):
    css = """
    .btn-active {background-color: "orange"}
    #logout_btn {
      align-self: self-end;
      width: 65px;
    }
    .sample-button {
      min-width: 100px;
    }
    """
    # title="VGM Text To Text",
    # description = "A vietnamese text-to-text tool."
    app = gr.Blocks(title="VGM Text To Text", theme=gr.themes.Default(), css=css)
    with app:
        with gr.Row():
          with gr.Column():
            gr.Markdown("# VGM Text To Text")
          with gr.Column():
            gr.Button("Logout", link="/logout", size="sm", icon=None, elem_id="logout_btn", visible=True if os.getenv('ENABLE_AUTH', '') == "true" else False)
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Row():
                    with gr.Column():
                        input_files = gr.Files(label="Upload .doc|.docx|.txt|.srt file(s)", file_types=[".doc", ".docx", ".txt", ".srt"])
                        input_text = gr.Textbox(label="Text for synthesize")
                        with gr.Row():
                          SOURCE_LANGUAGE = gr.Dropdown(['Automatic detection', 'Arabic (ar)', 'Cantonese (yue)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='English (en)',label = 'Source language', info="This is the original language of the video", scale=1)
                          SOURCE_LANGUAGE.change(None, SOURCE_LANGUAGE, None, js="(v) => setStorage('SOURCE_LANGUAGE',v)")
                          TARGET_LANGUAGE = gr.Dropdown(['Arabic (ar)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='Vietnamese (vi)',label = 'Target language', info="Select the target language for translation", scale=1)
                          TARGET_LANGUAGE.change(None, TARGET_LANGUAGE, None, js="(v) => setStorage('TARGET_LANGUAGE',v)")                        
                    with gr.Column():
                        files_output = gr.Files(label="Files Output")
                        with gr.Row():
                          clear_btn = gr.ClearButton([input_files,files_output], value="Refresh")
                          btn = gr.Button(value="Generate!", variant="primary")
            with gr.TabItem("Settings"):
                with gr.Column():
                  with gr.Row():
                    t2t_method = gr.Dropdown(["Google", "LLM"], label='T2T', value=user_settings['t2t'], visible=True, elem_id="t2t_method",interactive=True)
                  with gr.Accordion("LLM Settings", open=True):
                    with gr.Row():
                      llm_url = gr.Textbox(label="LLM Endpoint", placeholder="LLM Endpoint goes here...", value=user_settings['llm_url'], elem_id="llm_url", scale=5)
                      llm_model = gr.Dropdown(label="LLM Model", choices=user_settings['llm_models'], value=user_settings['llm_model'], elem_id="llm_model",scale=5)        
                      llm_temp = gr.Slider(0.1, 1, value=0.6, step=0.1, label="Temparature",scale=5, interactive=True)
                      llm_k = gr.Slider(1, 100, value=5, step=1, label="K", scale=5, interactive=True)
                      llm_refresh = gr.Button("Refresh", scale=2)
                      ## Config LLM Settings
                      def update_llm_model(llm_url):
                        models = get_llm_models(llm_url)
                        if models and len(models) > 0:
                          user_settings['llm_url'] = llm_url
                          user_settings['llm_models'] = models
                          user_settings['llm_model'] = models[0]
                        if 't2s' in user_settings:
                          save_settings(settings=user_settings)
                        return gr.update(choices=models)
                      llm_url.blur(update_llm_model, [llm_url], [llm_model])
                      llm_url.change(None, llm_url, None, js="(v) => setStorage('llm_url',v)")
                      llm_model.change(None, llm_model, None, js="(v) => setStorage('llm_model',v)")
                      llm_temp.change(None, llm_temp, None, js="(v) => setStorage('llm_temp',v)")
                      llm_k.change(None, llm_k, None, js="(v) => setStorage('llm_k',v)")
                      llm_refresh.click(update_llm_model, [llm_url], [llm_model])  
                    
        ## Run function
        btn.click(self.run,
                inputs=[input_files, input_text, t2t_method, llm_url, llm_model, llm_temp, llm_k, SOURCE_LANGUAGE, TARGET_LANGUAGE],
                outputs=[files_output], concurrency_limit=1)
                
        app.load(
            None,
            inputs=None,
            outputs=[
            t2t_method,
            llm_url,
            llm_model,
            llm_temp,
            llm_k,
            SOURCE_LANGUAGE,
            TARGET_LANGUAGE,
              ],
            js=get_local_storage,
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
    ## Initialise app
    print("Application running on::", sys.platform)
    os.makedirs( CONFIG.os_tmp, exist_ok=True)
    os.system(f'rm -rf { CONFIG.os_tmp}/*')
    os.system(f'rm -rf { CONFIG.gradio_temp_dir}/*')
    ## Set torch multiprocessing
    # mp.set_start_method('spawn', force=True)
    host = "localhost"
    port = 3200
    ttt = TTT()
    app = ttt.web_interface(port)
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