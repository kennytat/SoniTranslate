import os
import sys
import gc
import subprocess
from pathlib import Path
from pydub import AudioSegment
import atexit
import argparse
import shutil
import tempfile
import gradio as gr
from utils import new_dir_now, encode_filename
import torch
from vietTTS.upsample import Predictor
import soundfile as sf
from utils import new_dir_now

total_input = []
total_output = []
upsampler = None
# Check GPU
CUDA_MEM = int(torch.cuda.get_device_properties(0).total_memory)
if torch.cuda.is_available():
    device = "cuda"
    list_compute_type = ['float16', 'float32']
    compute_type_default = 'float16'
    whisper_model_default = 'large-v3' if CUDA_MEM > 9000000000 else 'medium'
else:
    device = "cpu"
    list_compute_type = ['float32']
    compute_type_default = 'float32'
    whisper_model_default = 'medium'

  
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
    os_tmp = Path(os.path.join(tempfile.gettempdir(), "upsample"))

def split_audio(input_file="", extension="", output_folder="", chunk_duration=10000):
    print("split_audio called:",input_file, extension, output_folder, chunk_duration)
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Get the duration of the audio in milliseconds
    audio_duration = len(audio)

    # Calculate the number of chunks
    num_chunks = audio_duration // chunk_duration

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    output_files = []
    # Split the audio into chunks
    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = (i + 1) * chunk_duration

        # Extract the chunk
        chunk = audio[start_time:end_time]

        # Save the chunk to a new file
        output_file = os.path.join(output_folder, f"chunk_{i + 1}{extension}")
        # print("output_file::", output_file)
        chunk.export(output_file, format=extension.replace(".",""))
        output_files.append(output_file)
    return output_files

def join_audio(chunk_files=[], extension="", output_file=""):
    # Get a list of all audio files in the chunks folder
    # print(chunk_files)
    # Initialize an empty AudioSegment
    joined_audio = AudioSegment.silent(duration=0)

    # Concatenate each chunk to the joined_audio
    for chunk_file in chunk_files:
        chunk = AudioSegment.from_file(chunk_file)
        joined_audio += chunk

    # Export the joined audio to the output file
    joined_audio.export(output_file, format=extension.replace(".",""))
    
def upsampling(filepath):
  global upsampler
  if not upsampler:
    upsampler = Predictor()
    upsampler.setup(model_name="speech")
  print("upsampling:", filepath)
  audio_data, sample_rate = sf.read(filepath)
  source_duration = len(audio_data) / sample_rate
  data = upsampler.predict(
      filepath,
      ddim_steps=50,
      guidance_scale=3.5,
      seed=42
  )
  ## Trim duration to match source duration
  target_samples = int(source_duration * 48000)
  sf.write(filepath, data=data[:target_samples], samplerate=48000)
  return filepath

def start(input_files):
    output_dir_name = new_dir_now()
    output_dir_path = os.path.join(CONFIG.os_tmp, output_dir_name)
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    print("stt called::",   input_files)
    file_list = [f.name for f in input_files]
    results_list = []
    print("Start upsampling::")
    global total_input
    global total_output
    total_input = input_files
    for index, file_path in enumerate(file_list):
      try:
        print('file_path::',file_path)
        tmp_dir = os.path.join(output_dir_path, encode_filename(file_path))
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        basename, ext = os.path.splitext(file_path)
        output_file = os.path.join(Path(output_dir_path).absolute(), os.path.basename(file_path))
        
        # is_video = True if is_video_or_audio(file_path) == 'video' else False
        is_video = True if os.path.splitext(os.path.basename(file_path.strip()))[1] == '.mp4' else False
        if is_video:
          video_path = file_path
          audio_path = f"{os.path.splitext(output_file)[0]}.wav"
          subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn" ,"-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path])
        else:
          audio_path = file_path
        ## Split audio
        split_audio_array = split_audio(audio_path, ext, tmp_dir, 10000)
        ## Upsample audio
        for audio in split_audio_array:
          upsampling(audio)
        global upsampler
        upsampler = None; gc.collect(); torch.cuda.empty_cache()
        ## Join audio
        join_audio(split_audio_array, ext, audio_path)
        ## Combined if is video
        if is_video:
          subprocess.run(["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-map", "0:v", "-map", "1:a", "-shortest", output_file])
        else:
          shutil.copy(audio_path, output_file)    
        print(f'Done:: {index}/{len(file_list)} task::', file_path)  
        results_list.append(output_file)
        total_output.append(output_file)
        ## Remove tmp files
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.remove(file_path)
      except:
          print("Skip error file while upsampling: {}".format(file_path))
    print("[DONE] {} tasks: {}".format(len(results_list), results_list))
    return results_list

def web_interface(port):
  css = """
  .btn-active {background-color: "orange"}
  """
  app = gr.Blocks(title="VGM Audio Enhancer", theme=gr.themes.Default(), css=css)
  with app:
      gr.Markdown("# VGM Audio Enhancer")
      with gr.Tabs():
          with gr.TabItem("Audio Enhancer"):
              with gr.Row():
                  with gr.Column():
                      input_files = gr.Files(label="Upload media file(s)", file_types=["audio","video"])
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
                        btn.click(start,
                                inputs=[input_files],
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
    port = 3110
    ## Parser argurment
    parser = argparse.ArgumentParser(description="VGM Audio Enhance application")
    parser.add_argument("-pf", "--platform", help="STT Platform, default to desktop", default="web")
    parser.add_argument("-m", "--model", help="Custom path for model directory, default to current folder")
    parser.add_argument("-f", "--file", help="Input file for STT")
    parser.add_argument("-o", "--output", help="Output directory")
    args = parser.parse_args()
    # ## Change ckpt_dir path if provided
    # if args.model:
    #     CONFIG.model = args.model
    #     print("ckpt_dir:",  CONFIG.model)
    ## Execute app
    if args.platform == "web" and args.file:
        raise TypeError("Could not start Upsample from WEB and CLI at same time")
    elif args.platform == "web":
        web_interface(port)
    elif args.platform == "desktop":
        pass
        # start_desktop_interface(host, port)
    elif args.platform == "cli" and args.file:
        pass
        # STT(file=args.file, text=args.text, voice=args.voice, speed=args.speed, method=args.method, output=args.output)
    else:
        raise TypeError("Not enough or wrong argument, please try again")
    sys.exit()