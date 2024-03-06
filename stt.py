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
  """
  app = gr.Blocks(title="VGM Speech To Text", theme=gr.themes.Default(), css=css)
  with app:
      gr.Markdown("# VGM Speech To Text")
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
  auth_user = os.getenv('AUTH_USER', '')
  auth_pass = os.getenv('AUTH_PASS', '')
  app.queue().launch(
    auth=(auth_user, auth_pass) if auth_user != '' and auth_pass != '' else None,
    show_api=False,
    debug=False,
    inbrowser=True,
    show_error=True,
    server_name="0.0.0.0",
    server_port=port,
    share=False)

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
      
if __name__ == "__main__":
    ## Download model if not exist
    # os.system('/bin/sh update_model.sh')
    ## Initialise app
    print("Application running on::", sys.platform)
    os.makedirs( CONFIG.os_tmp, exist_ok=True)
    os.system(f'rm -rf { CONFIG.os_tmp}/*')
    
    host = "localhost"
    port = 3100
    ## Parser argurment
    parser = argparse.ArgumentParser(description="VGM STT application")
    parser.add_argument("-pf", "--platform", help="STT Platform, default to desktop", default="web")
    parser.add_argument("-m", "--model", help="Custom path for model directory, default to current folder")
    parser.add_argument("-f", "--file", help="Input file for STT")
    parser.add_argument("-o", "--output", help="Output directory")
    args = parser.parse_args()
    ## Change ckpt_dir path if provided
    if args.model:
        CONFIG.STT_ckpt_dir = args.model
        print("ckpt_dir:",  CONFIG.STT_ckpt_dir)
    ## Execute app
    if args.platform == "web" and args.file:
        raise TypeError("Could not STT from WEB and CLI at same time")
    elif args.platform == "cli" and args.file and args.text:
        raise TypeError("Could not STT-CLI text and file at same time")
    elif args.platform == "web":
        web_interface(port)
    elif args.platform == "desktop":
        pass
        # start_desktop_interface(host, port)
    elif (args.platform == "cli" and args.file) or (args.platform == "cli" and args.text):
        pass
        # STT(file=args.file, text=args.text, voice=args.voice, speed=args.speed, method=args.method, output=args.output)
    else:
        raise TypeError("Not enough or wrong argument, please try again")
    sys.exit()