import os
import sys
import subprocess
from pathlib import Path
import atexit
import argparse
import shutil
import tempfile
import gradio as gr
from utils import new_dir_now, encode_filename
import torch
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
from dotenv import load_dotenv
load_dotenv()
total_input = []
total_output = []

# Check GPU
if torch.cuda.is_available():
    device = "cuda"
    list_compute_type = ['float16', 'float32']
    compute_type_default = 'float16'
    CUDA_MEM = int(torch.cuda.get_device_properties(0).total_memory)
    whisper_model_default = 'large-v3' if CUDA_MEM > 9000000000 else 'medium'
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

        
class CONFIG():
    """Configurations"""
    # ckpt
    os_tmp = Path(os.path.join(tempfile.gettempdir(), "STT"))

def STT(
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
    file_list = [f.name for f in input_files]
    results_list = []
    LANGUAGE = LANGUAGES[LANGUAGE]
    print("Start transcribing source language::")
    global total_input
    global total_output
    total_input = input_files
    for index, file_path in enumerate(file_list):
      try:
        print('file_path::',file_path)
        tmp_dir = os.path.join(output_dir_path, encode_filename(file_path))
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        whisper_args = ['whisperx', '--model', whisper_model, '--no_align', '--batch_size', str(batch_size),'--chunk_size', str(chunk_size), file_path ,'-o', tmp_dir]
        if LANGUAGE != 'Automatic detection':
          whisper_args.extend(['--language', LANGUAGE])
        subprocess.run(whisper_args)
        print(f'Done:: {index}/{len(file_list)} task::', file_path)
        archive_path = os.path.join(Path(output_dir_path).absolute(), os.path.splitext(os.path.basename(file_path))[0])
        shutil.make_archive(archive_path, 'zip', tmp_dir)   
        results_list.append(f"{archive_path}.zip")
        total_output.append(f"{archive_path}.zip")
        ## Remove tmp files
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.remove(file_path)
      except:
          print("Skip error file while stt: {}".format(file_path))
    print("[DONE] {} tasks: {}".format(len(results_list), results_list))
    return results_list

def web_interface(port):
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
          with gr.TabItem("STT"):
              with gr.Row():
                  with gr.Column():
                      input_files = gr.Files(label="Upload audio file(s)", file_types=["audio"])
                      with gr.Row():
                        WHISPER_MODEL = gr.Dropdown(['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'], value=whisper_model_default, label="Whisper model",  scale=1)
                        LANGUAGE = gr.Dropdown(list(LANGUAGES.keys()), value='English (en)',label = 'Language', scale=1)
                      with gr.Row():
                        batch_size = gr.Slider(1, 32, value=16, label="Batch size", step=1)
                        chunk_size = gr.Slider(2, 30, value=5, label="Chunk size", step=1)
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
                        clear_btn = gr.ClearButton([input_files,files_output])
                        clear_btn.click(reset_param,[],[files_output,tmp_output])
                        def update_output_visibility():
                          return gr.update(label="Audio Files Output"),gr.update(visible=False)
                        btn = gr.Button(value="Generate!", variant="primary")
                        btn.click(STT,
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
    host = "localhost"
    port = 3100
    app = web_interface(port)
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
        show_api=False,
        debug=False,
        inbrowser=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=port,
        share=False)
    sys.exit()