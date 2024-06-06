from dotenv import load_dotenv
import subprocess
import json
import yt_dlp
from pathlib import Path,PureWindowsPath, PurePosixPath
import joblib
from joblib import Parallel, delayed
import gradio as gr
import whisperx
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH as DAMT, DEFAULT_ALIGN_MODELS_HF as DAMHF
from IPython.utils import capture
import torch
import torch.multiprocessing as mp
from gtts import gTTS
import librosa
import math
import gc
from tqdm import tqdm
import os
from audio_segments import create_translated_audio
from text_to_speech import make_voice_gradio
from translate_segments import translate_text
import time
import shutil
import logging
import tempfile
from vietTTS.utils import concise_srt
# from vietTTS.upsample import Predictor
import soundfile as sf
from utils.language_configuration import LANGUAGES, EXTRA_ALIGN, INVERTED_LANGUAGES
from utils.utils import new_dir_now, segments_to_srt, srt_to_segments, segments_to_txt, is_video_file, is_audio_file, is_windows_path, convert_to_wsl_path, find_all_media_files, find_most_matching_prefix, youtube_download, get_llm_models
# from utils.logging_setup import logger
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
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
from spell_check import SpellCheck
from utils.tts_utils import edge_tts_voices_list, piper_tts_voices_list
from ovc_voice_main import OpenVoice
from pydub import AudioSegment
load_dotenv()

total_input = []
total_output = []
upsampler = None
gradio_temp_dir = os.getenv("GRADIO_TEMP_DIR", "/tmp/gradio-vgm")
gradio_temp_processing_dir = os.path.join(gradio_temp_dir, "processing_dir")
srt_temp_dir = os.path.join(tempfile.gettempdir(), "vgm-translate", 'srt')
Path(srt_temp_dir).mkdir(parents=True, exist_ok=True)
youtube_temp_dir = os.path.join(tempfile.gettempdir(), "vgm-translate", 'youtube')
Path(youtube_temp_dir).mkdir(parents=True, exist_ok=True) 

import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

sys.stdout = Logger("output.log")

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()

def submit_file_func(file):
    print(file.name)
    return file.name, file.name

## UI Config
MAX_TTS = 6
theme="Taithrah/Minimal"
css = """
.btn-active {background-color: "orange"}
#logout_btn {
  align-self: self-end;
  width: 65px;
}
"""
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
  const TRANSLATE_AUDIO_TO = getStorage("TRANSLATE_AUDIO_TO");
  
  const WHISPER_MODEL_SIZE = getStorage("WHISPER_MODEL_SIZE");
  const compute_type = getStorage("compute_type");
  const batch_size = getStorage("batch_size");
  const chunk_size = getStorage("chunk_size");

  return [
    s2t_method || "Whisper",
    t2t_method || "LLM",
    t2s_method || "VietTTS",
    vc_method || "None",
    llm_url || "http://infer-2.vgm.chat,http://infer-3.vgm.chat",
    llm_model,
    llm_temp || 0.3,
    llm_k || 3000,
    max_speakers || 1,
    tts_voice00,
    tts_speed00 || 1,
    vc_voice00,
    tts_voice01,
    tts_speed01 || 1,
    vc_voice01,
    tts_voice02,
    tts_speed02 || 1,
    vc_voice02,
    tts_voice03,
    tts_speed03 || 1,
    vc_voice03,
    tts_voice04 || 1,
    tts_speed04,
    vc_voice04,
    tts_voice05,
    tts_speed05 || 1,
    vc_voice05,
    match_length || true,
    match_start || true,
    SOURCE_LANGUAGE || "English (en)",
    TRANSLATE_AUDIO_TO || "Vietnamese (vi)",
    WHISPER_MODEL_SIZE || "large-v3",
    compute_type || "float16",
    batch_size || 16,
    chunk_size || 5
  ];
}
"""

title = "<strong><font size='7'>VGM Translate</font></strong>"

description = """
### 🎥 **Translate videos easily with VGM Translate!** 📽️

🎥 Upload a video or provide a video link. 📽️
🎥 Upload SRT File for skiping S2T & T2T 📽️
 - SRT Format: "<media name>-<target-language>-SPEAKER.srt" - Example: "video-vi-SPEAKER.srt"
 - See the tab labeled 'Help' for instructions on how to use it. Let's start having fun with video translation! 🚀🎉
"""

tutorial = """

# 🔰 **Instructions for use:**

1. 📤 **Upload a video** on the first tab or 🌐 **use a video link** on the second tab.

2. 🌍 Choose the language in which you want to **translate the video**.

3. 🗣️ Specify the **number of people speaking** in the video and **assign each one a text-to-speech voice** suitable for the translation language.

4. 🚀 Press the '**Translate**' button to obtain the results.


# 🎤 How to Use RVC and RVC2 Voices 🎶

The goal is to apply a RVC (Retrieval-based Voice Conversion) to the generated TTS (Text-to-Speech) 🎙️

1. In the `Custom Voice RVC` tab, download the models you need 📥 You can use links from Hugging Face and Google Drive in formats like zip, pth, or index. You can also download complete HF space repositories, but this option is not very stable 😕

2. Now, go to `Replace voice: TTS to RVC` and check the `enable` box ✅ After this, you can choose the models you want to apply to each TTS speaker 👩‍🦰👨‍🦱👩‍🦳👨‍🦲

3. Adjust the F0 method that will be applied to all RVCs 🎛️

4. Press `APPLY CONFIGURATION` to apply the changes you made 🔄

5. Go back to the video translation tab and click on 'Translate' ▶️ Now, the translation will be done applying the RVCs 🗣️

Tip: You can use `Test RVC` to experiment and find the best TTS or configurations to apply to the RVC 🧪🔍

"""

# Check GPU
if torch.cuda.is_available():
    device = "cuda"
    list_compute_type = ['float16', 'float32']
    compute_type_default = 'float16'
    CUDA_MEM = int(torch.cuda.get_device_properties(0).total_memory)
    whisper_model_default = 'large-v3' if CUDA_MEM > 9000000000 else 'medium'
else:
    device = "cpu"
    list_compute_type = ['float32']
    compute_type_default = 'float32'
    whisper_model_default = 'medium'

list_etts = edge_tts_voices_list()
list_gtts = ['default']
list_ptts = piper_tts_voices_list()
list_vtts = [voice for voice in os.listdir(os.path.join("model","vits")) if os.path.isdir(os.path.join("model","vits", voice))]
list_svc = [voice for voice in os.listdir(os.path.join("model","svc")) if os.path.isdir(os.path.join("model","svc", voice))]
list_rvc = [voice for voice in os.listdir(os.path.join("model","rvc")) if voice.endswith('.pth')]
list_ovc = [voice for voice in os.listdir(os.path.join("model","openvoice","target_voice")) if os.path.isdir(os.path.join("model","openvoice","target_voice", voice))]


# models, index_paths = upload_model_list()

f0_methods_voice = ["pm", "harvest", "crepe", "rmvpe"]

from rvc_voice_main import RVCClassVoices
rvc_voices = RVCClassVoices()

from svc_voice_main import SVCClassVoices
svc_voices = SVCClassVoices()

# Function to save settings to a JSON file
def save_settings(settings, filename='user_settings.json'):
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
 
class Main():
    def __init__(self):
        self.create_ui()
 
    def handle_link_input(self, link_inputs):
      
      print("links::", link_inputs)
      link_inputs = link_inputs.split(',')
      media_inputs = []
      self.input_dirs = []
      if link_inputs is not None and len(link_inputs) > 0 and link_inputs[0] != '':
        for url in link_inputs:
          url = url.strip()
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
              print("tmp_dir::", tmp_dir)
              self.input_dirs.append(tmp_dir)
              files = find_all_media_files(osPath)
              print(f"media found in directory:: {osPath} | ", files)
              if len(files) > 0:
                for file in files:
                  tmp_file = os.path.join(gradio_temp_processing_dir, file.replace(os.path.dirname(osPath),"").strip('/'))
                  subprocess.run(["mkdir", "-p", os.path.dirname(tmp_file)], capture_output=True, text=True)
                  shutil.copy(file, tmp_file)
                  media_inputs.append(tmp_file)
              else:
                gr.Warning(f"No media files found in: {osPath}")
      return media_inputs, ""
      

    def batch_preprocess(self,
      media_inputs,
      srt_inputs,
      s2t_method,
      t2t_method,
      t2s_method,
      vc_method,
      llm_url,
      llm_model,
      llm_temp,
      llm_k,
      match_length,
      match_start,
      YOUR_HF_TOKEN,
      preview=False,
      WHISPER_MODEL_SIZE="large-v3",
      batch_size=16,
      chunk_size=5,
      compute_type="float16",
      SOURCE_LANGUAGE= "Automatic detection",
      TRANSLATE_AUDIO_TO="English (en)",
      min_speakers=1,
      max_speakers=2,
      tts_voice00="en-AU-WilliamNeural-Male",
      tts_voice01="en-CA-ClaraNeural-Female",
      tts_voice02="en-GB-ThomasNeural-Male",
      tts_voice03="en-GB-SoniaNeural-Female",
      tts_voice04="en-NZ-MitchellNeural-Male",
      tts_voice05="en-GB-MaisieNeural-Female",
      tts_speed00=1,
      tts_speed01=1,
      tts_speed02=1,
      tts_speed03=1,
      tts_speed04=1,
      tts_speed05=1,
      vc_voice00=None,
      vc_voice01=None,
      vc_voice02=None,
      vc_voice03=None,
      vc_voice04=None,
      vc_voice05=None,
      AUDIO_MIX_METHOD='Adjusting volumes and mixing audio',
      progress=gr.Progress(),
    ):
      ## =====================Asign to self============================
      self.s2t_method = s2t_method
      self.t2t_method = t2t_method
      self.t2s_method = t2s_method
      self.vc_method = vc_method
      self.llm_url = llm_url
      self.llm_model = llm_model
      self.llm_temp = llm_temp
      self.llm_k = llm_k
      self.match_length = match_length
      self.match_start = match_start
      self.YOUR_HF_TOKEN = YOUR_HF_TOKEN
      self.preview = preview
      self.WHISPER_MODEL_SIZE=WHISPER_MODEL_SIZE
      self.batch_size=batch_size
      self.chunk_size=chunk_size
      self.compute_type=compute_type
      self.min_speakers=min_speakers
      self.max_speakers=max_speakers
      self.tts_voice00=tts_voice00
      self.tts_voice01=tts_voice01
      self.tts_voice02=tts_voice02
      self.tts_voice03=tts_voice03
      self.tts_voice04=tts_voice04
      self.tts_voice05=tts_voice05
      self.tts_speed00=tts_speed00
      self.tts_speed01=tts_speed01
      self.tts_speed02=tts_speed02
      self.tts_speed03=tts_speed03
      self.tts_speed04=tts_speed04
      self.tts_speed05=tts_speed05
      self.vc_voice00=vc_voice00
      self.vc_voice01=vc_voice01
      self.vc_voice02=vc_voice02
      self.vc_voice03=vc_voice03
      self.vc_voice04=vc_voice04
      self.vc_voice05=vc_voice05
      self.AUDIO_MIX_METHOD=AUDIO_MIX_METHOD
      
      ## =================================================================

      ## Move all srt files to srt tempdir
      media_inputs = media_inputs if media_inputs is not None else []
      media_inputs = media_inputs if isinstance(media_inputs, list) else [media_inputs]
      output = []

      os.system(f"rm -rf {srt_temp_dir}/*")
      os.system(f"rm -rf {youtube_temp_dir}/*")
      
      # path_inputs = [item.strip() for item in path_inputs.split(',')]
      # print("path_inputs::", path_inputs)
      # if path_inputs is not None and len(path_inputs) > 0 and path_inputs[0] != '':
      #   for media_path in path_inputs:
      #     media_path = media_path.strip()
      #     print("media_path::", media_path)
      #     if is_windows_path(media_path):
      #       window_path = PureWindowsPath(media_path)
      #       path_arr = [item for item in window_path.parts]
      #       path_arr[0] = re.sub(r'\:\\','',path_arr[0].lower())
      #       wsl_path = str(PurePosixPath('/mnt', *path_arr))
      #       print("wsl_path::", wsl_path)
      #       if os.path.exists(wsl_path):
      #         media_inputs.append(wsl_path)
      #       else:
      #         raise Exception(f"Path not exist:: {wsl_path}")
      #     else:
      #       if os.path.exists(media_path):
      #         media_inputs.append(media_path)
      #       else:
      #         raise Exception(f"Path not exist:: {media_path}")
              

                
      if srt_inputs is not None and len(srt_inputs)> 0:
        for srt in srt_inputs:
          os.system(f"mv {srt.name} {srt_temp_dir}/")
      global total_input
      global total_output
      print("process total files::", len(media_inputs))
      media_inputs = [file for file in media_inputs if os.path.isfile(file)]
      if media_inputs is not None and len(media_inputs)> 0:
        total_input = media_inputs
        for media in media_inputs:
          result = self.translate_from_media(media, self.input_dirs, SOURCE_LANGUAGE, TRANSLATE_AUDIO_TO, progress)
          total_output.append(result)
          output.append(result)
      return output

    def upsampling(self, file):
      # global upsampler
      # if not upsampler:
      #   upsampler = Predictor()
      #   upsampler.setup(model_name="speech")
      # filepath = os.path.join("audio2", file[0])
      # print("upsampling:", filepath)
      # audio_data, sample_rate = sf.read(filepath)
      # source_duration = len(audio_data) / sample_rate
      # data = upsampler.predict(
      #     filepath,
      #     ddim_steps=50,
      #     guidance_scale=3.5,
      #     seed=42
      # )
      # ## Trim duration to match source duration
      # target_samples = int(source_duration * 48000)
      # sf.write(filepath, data=data[:target_samples], samplerate=48000)
      return file

    def tts(self, segment, TRANSLATE_AUDIO_TO, speaker_to_voice, speaker_to_speed):
        text = segment['text']
        start = segment['start']
        end = segment['end']
        duration_true = end - start

        try:
            speaker = segment['speaker']
            print("speaker::", speaker)
        except KeyError:
            segment['speaker'] = "SPEAKER_99"
            speaker = segment['speaker']
            print(f"NO SPEAKER DETECT IN SEGMENT: Create blank segment --- {segment['start'], segment['text']}")

        # make the tts audio
        filename = f"audio/{start}.wav"

        if speaker in speaker_to_voice and speaker_to_voice[speaker] != 'None':
            make_voice_gradio(text, speaker_to_voice[speaker], speaker_to_speed[speaker], filename, TRANSLATE_AUDIO_TO, self.t2s_method)
        elif speaker == "SPEAKER_99":
            second_of_silence = AudioSegment.silent(duration=duration_true) # or be explicit
            second_of_silence = second_of_silence.set_frame_rate(16000)
            second_of_silence.export(filename, format="wav")

        # duration
        if os.path.isfile(filename):
          try:
            
            duration_tts = librosa.get_duration(path=filename)
            # porcentaje
            porcentaje = duration_tts / duration_true
            print("change speed::", porcentaje, duration_tts, duration_true)
            # Smooth and round
            porcentaje = math.floor(porcentaje * 10000) / 10000
            porcentaje = 0.8 if porcentaje <= 0.8 else porcentaje + 0.005
            porcentaje = 1.5 if porcentaje >= 1.5 else porcentaje
            porcentaje = 1.0 if not self.match_length else porcentaje     
          except Exception as e:
            porcentaje = 1.0 
            print('An exception occurred:', e)
          # apply aceleration or opposite to the audio file in audio2 folder
          os.system(f"ffmpeg -y -loglevel panic -i {filename} -filter:a atempo={porcentaje} audio2/{filename}")
        gc.collect(); torch.cuda.empty_cache()
        # duration_create = librosa.get_duration(filename=f"audio2/{filename}")
        return (filename, speaker) 
      
    def translate_from_media(self,
        media_input,
        input_dirs = [],
        SOURCE_LANGUAGE= "Automatic detection",
        TRANSLATE_AUDIO_TO="Vietnamese (vi)",
        progress=gr.Progress(),
        ):
        print("processing::", media_input)
        if self.YOUR_HF_TOKEN == "" or self.YOUR_HF_TOKEN == None:
          self.YOUR_HF_TOKEN = os.getenv("YOUR_HF_TOKEN")
          if self.YOUR_HF_TOKEN == None:
            print('No valid token')
            return "No valid token"
          else:
            os.environ["YOUR_HF_TOKEN"] = self.YOUR_HF_TOKEN

        media_input = media_input if isinstance(media_input, str) else media_input.name
        # media_input = '/home/vgm/Desktop/WE KNOW LOVE 09 17 23 - ANAHEIM CHURCH.mp4'
        # print(media_input)

        if "SET_LIMIT" == os.getenv("DEMO"):
          self.preview=True
          print("DEMO; set preview=True; The generation is **limited to 10 seconds** to prevent errors with the CPU. If you use a GPU, you won't have any of these limitations.")
          self.AUDIO_MIX_METHOD='Adjusting volumes and mixing audio'
          print("DEMO; set Adjusting volumes and mixing audio")
          self.WHISPER_MODEL_SIZE="medium"
          print("DEMO; set whisper model to medium")
        print("LANGUAGES::", TRANSLATE_AUDIO_TO)
        TRANSLATE_AUDIO_TO = LANGUAGES[TRANSLATE_AUDIO_TO]
        SOURCE_LANGUAGE = LANGUAGES[SOURCE_LANGUAGE]


        if not os.path.exists('audio'):
            os.makedirs('audio')

        if not os.path.exists('audio2/audio'):
            os.makedirs('audio2/audio')

        # Check GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float32" if device == "cpu" else self.compute_type

        temp_dir = os.path.join(tempfile.gettempdir(), "vgm-translate", new_dir_now())
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        
        is_video = is_video_file(media_input)
        # is_video = True if os.path.splitext(os.path.basename(media_input.strip()))[1] == '.mp4' else False
        
        OutputFile = os.path.join(temp_dir, 'Video.mp4') if is_video else os.path.join(temp_dir, 'Audio.mp3')
        file_name, file_extension = os.path.splitext(os.path.basename(media_input.strip().replace(' ','_')))
        mix_audio = os.path.join(temp_dir, f"{file_name}.mp3") 
        media_output_name = f"{file_name}-{TRANSLATE_AUDIO_TO}{file_extension}"
        media_output = os.path.join(temp_dir, media_output_name)
        source_media_output_basename = os.path.join(temp_dir, f'{file_name}-{SOURCE_LANGUAGE}')
        target_media_output_basename = os.path.join(temp_dir, f'{file_name}-{TRANSLATE_AUDIO_TO}') 
        audio_wav = f"{source_media_output_basename}.wav"
        audio_webm = f"{source_media_output_basename}.webm"
        translated_output_file = os.path.join(temp_dir, f"{target_media_output_basename}.wav")
        
        # os.system("rm -rf Video.mp4")
        # os.system("rm -rf audio_origin.webm")
        # os.system("rm -rf audio_origin.wav")

        progress(0.15, desc=f"Processing media...")
        if os.path.exists(media_input):
            if is_video:
              if self.preview:
                  print('Creating a preview video of 10 seconds, to disable this option, go to advanced settings and turn off preview.')
                  os.system(f'ffmpeg -y -i "{media_input}" -ss 00:00:20 -t 00:00:10 -c:v libx264 -c:a aac -strict experimental {OutputFile}')
              else:
                  # Check if the file ends with ".mp4" extension
                  if media_input.endswith(".mp4"):
                      destination_path = OutputFile
                      shutil.copy(media_input, destination_path)
                  else:
                      print("File does not have the '.mp4' extension. Converting video.")
                      os.system(f'ffmpeg -y -i "{media_input}" -c:v libx264 -c:a aac -strict experimental {OutputFile}')
            else:
              if self.preview:
                  print('Creating a preview video of 10 seconds, to disable this option, go to advanced settings and turn off preview.')
                  os.system(f'ffmpeg -y -i "{media_input}" -ss 00:00:20 -t 00:00:10 -strict experimental {OutputFile}')
              else:
                  # Check if the file ends with ".mp4" extension
                  if media_input.endswith(".mp3"):
                      destination_path = OutputFile
                      shutil.copy(media_input, destination_path)
                  else:
                      print("File does not have the '.mp3' extension. Converting audio.")
                      os.system(f"ffmpeg -y -i '{media_input}' -strict experimental '{OutputFile}'")   

            for i in range (120):
                time.sleep(1)
                print('process media...')
                if os.path.exists(OutputFile):
                    time.sleep(1)
                    os.system(f"ffmpeg -y -i '{OutputFile}' -vn -acodec pcm_s16le -ar 44100 -ac 2 '{audio_wav}'")
                    time.sleep(1)
                    break
                if i == 119:
                  print('Error processing media')
                  return

            for i in range (120):
                time.sleep(1)
                print('process audio...')
                if os.path.exists(audio_wav):
                    break
                if i == 119:
                  print("Error can't create the audio")
                  return

        else:
            print("path not found::", media_input)
        #     if self.preview:
        #         print('Creating a preview from the link, 10 seconds to disable this option, go to advanced settings and turn off preview.')
        #         #https://github.com/yt-dlp/yt-dlp/issues/2220
        #         mp4_ = f'yt-dlp -f "mp4" --downloader ffmpeg --downloader-args "ffmpeg_i: -ss 00:00:20 -t 00:00:10" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {media_input}'
        #         wav_ = "ffmpeg -y -i {OutputFile} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_wav}"
        #         os.system(mp4_)
        #         os.system(wav_)
        #     else:
        #         mp4_ = f'yt-dlp -f "mp4" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {media_input}'
        #         wav_ = f'python -m yt_dlp --output {audio_wav} --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --extract-audio --audio-format wav {media_input}'

        #         os.system(wav_)

        #         for i in range (120):
        #             time.sleep(1)
        #             print('process audio...')
        #             if os.path.exists(audio_wav) and not os.path.exists(audio_webm):
        #                 time.sleep(1)
        #                 os.system(mp4_)
        #                 break
        #             if i == 119:
        #               print('Error donwloading the audio')
        #               return

        print("Set file complete.")
        progress(0.30, desc="Speech to Text...")

        SOURCE_LANGUAGE = None if SOURCE_LANGUAGE == 'Automatic detection' else SOURCE_LANGUAGE

        # 1. Transcribe with original whisper (batched)
        print("Start transcribing source language::")
        with capture.capture_output() as cap:
          model = whisperx.load_model(
              self.WHISPER_MODEL_SIZE,
              device,
              compute_type=self.compute_type,
              language=SOURCE_LANGUAGE,
              )
          del cap
        audio = whisperx.load_audio(audio_wav)
        result = model.transcribe(self.WHISPER_MODEL_SIZE, audio, batch_size=self.batch_size, chunk_size=self.chunk_size, print_progress=True)
        gc.collect(); torch.cuda.empty_cache(); del model
        print("Transcript complete::", len(result["segments"]))

        ## =================================================================
        # # 2. Align whisper output for source language
        # print("Start aligning source language::")
        # progress(0.45, desc="Aligning source language...")
        # """
        # Aligns speech segments based on the provided audio and result metadata.

        # Parameters:
        # - audio (array): The audio data in a suitable format for alignment.
        # - result (dict): Metadata containing information about the segments
        #     and language.

        # Returns:
        # - result (dict): Updated metadata after aligning the segments with
        #     the audio. This includes character-level alignments if
        #     'return_char_alignments' is set to True.

        # Notes:
        # - This function uses language-specific models to align speech segments.
        # - It performs language compatibility checks and selects the
        #     appropriate alignment model.
        # - Cleans up memory by releasing resources after alignment.
        # """
        # DAMHF.update(DAMT)  # lang align
        # if (
        #     not result["language"] in DAMHF.keys()
        #     and not result["language"] in EXTRA_ALIGN.keys()
        # ):
        #     logger.warning(
        #         "Automatic detection: Source language not compatible with align"
        #     )
        #     raise ValueError(
        #         f"Detected language {result['language']}  incompatible, "
        #         "you can select the source language to avoid this error."
        #     )
        # if (
        #     result["language"] in EXTRA_ALIGN.keys()
        #     and EXTRA_ALIGN[result["language"]] == ""
        # ):
        #     lang_name = (
        #         INVERTED_LANGUAGES[result["language"]]
        #         if result["language"] in INVERTED_LANGUAGES.keys()
        #         else result["language"]
        #     )
        #     logger.warning(
        #         "No compatible wav2vec2 model found "
        #         f"for the language '{lang_name}', skipping alignment."
        #     )
        #     return result

        # model_a, metadata = whisperx.load_align_model(
        #     language_code=result["language"],
        #     device=os.environ.get("SONITR_DEVICE"),
        #     model_name=None
        #     if result["language"] in DAMHF.keys()
        #     else EXTRA_ALIGN[result["language"]],
        # )
        # result = whisperx.align(
        #     result["segments"],
        #     model_a,
        #     metadata,
        #     audio,
        #     os.environ.get("SONITR_DEVICE"),
        #     return_char_alignments=True,
        #     print_progress=False,
        # )
        # del model_a
        # gc.collect()
        # torch.cuda.empty_cache()  # noqa
    ## =================================================================

        if result['segments'] == []:
            print('No active speech found in audio')
            return

        # 3. Assign speaker labels
        print("Start Diarizing::")
        progress(0.50, desc="Diarizing...")
        if self.max_speakers > 1:
          with capture.capture_output() as cap:
            diarize_model = "pyannote/speaker-diarization-3.1" ## "pyannote/speaker-diarization-3.1" "pyannote/speaker-diarization@2.1"
            diarize_model = whisperx.DiarizationPipeline(model_name=diarize_model, use_auth_token=self.YOUR_HF_TOKEN, device=device)
            del cap
          diarize_segments = diarize_model(
              audio_wav,
              min_speakers=self.min_speakers,
              max_speakers=self.max_speakers)
          result_diarize = whisperx.assign_word_speakers(diarize_segments, result)
          gc.collect(); torch.cuda.empty_cache(); del diarize_model
        else:
          result_diarize = result
          result_diarize['segments'] = [{**item, 'speaker': "SPEAKER_00"} for item in result_diarize['segments']]

        # Mapping speakers to voice variables
        speaker_to_voice = {
            'SPEAKER_00': self.tts_voice00,
            'SPEAKER_01': self.tts_voice01,
            'SPEAKER_02': self.tts_voice02,
            'SPEAKER_03': self.tts_voice03,
            'SPEAKER_04': self.tts_voice04,
            'SPEAKER_05': self.tts_voice05
        }
        speaker_to_speed = {
            'SPEAKER_00': self.tts_speed00,
            'SPEAKER_01': self.tts_speed01,
            'SPEAKER_02': self.tts_speed02,
            'SPEAKER_03': self.tts_speed03,
            'SPEAKER_04': self.tts_speed04,
            'SPEAKER_05': self.tts_speed05
        }
        speaker_to_vc = {
            'SPEAKER_00': self.vc_voice00,
            'SPEAKER_01': self.vc_voice01,
            'SPEAKER_02': self.vc_voice02,
            'SPEAKER_03': self.vc_voice03,
            'SPEAKER_04': self.vc_voice04,
            'SPEAKER_05': self.vc_voice05
        }
        result_diarize['segments'] = [{**item, 'voice': speaker_to_voice[item['speaker']] if item['speaker'] != None else "", 'speed': speaker_to_speed[item['speaker']]} for item in result_diarize['segments']]
        print("Diarize complete::", result_diarize['segments'][0])


        # 4. Spell checking
        if SOURCE_LANGUAGE == "en":
          print("Start spell checking::")
          progress(0.55, desc="Spell checking...")
          try:
            checker = SpellCheck()
            for line in tqdm(range(len(result_diarize['segments']))):
              try:
                text = result_diarize['segments'][line]['text']
                result_diarize['segments'][line]['text'] = checker.correct(text)
              except Exception as e:
                pass 
          except Exception as e:
            print('Error initialize spell check::', e)
          del checker  
        
        # 4. Translate to target language
        print("Start translating::")
        progress(0.6, desc="Translating...")
        if TRANSLATE_AUDIO_TO == "zh":
            TRANSLATE_AUDIO_TO = "zh-CN"
        if TRANSLATE_AUDIO_TO == "he":
            TRANSLATE_AUDIO_TO = "iw"
        # print("os.path.splitext(media_input)[0]::", os.path.splitext(media_input)[0])
        ## Write source segment and srt,txt to file

        with open(f'{source_media_output_basename}.json', 'a', encoding='utf-8') as srtFile:
          srtFile.write(json.dumps(result_diarize['segments']))
        segments_to_srt(result_diarize['segments'], f'{source_media_output_basename}-origin.srt')
        result_diarize['segments'] = concise_srt(result_diarize['segments'], 375 if self.t2t_method == "LLM" else 500)
        segments_to_txt(result_diarize['segments'], f'{source_media_output_basename}.txt')
        segments_to_srt(result_diarize['segments'], f'{source_media_output_basename}.srt')
        target_srt_inputpath = os.path.join(tempfile.gettempdir(), "vgm-translate", 'srt', f'{file_name}-{TRANSLATE_AUDIO_TO}-SPEAKER.srt')
        if os.path.exists(target_srt_inputpath):
          # Start convert from srt if srt found
          print("srt file exist::", target_srt_inputpath)
          result_diarize['segments'] = srt_to_segments(target_srt_inputpath)
          result_diarize['segments'] = concise_srt(result_diarize['segments'])
        else:
          # Start translate if srt not found
          result_diarize['segments'] = translate_text(result_diarize['segments'], TRANSLATE_AUDIO_TO, self.t2t_method, self.llm_url, self.llm_model, self.llm_temp, self.llm_k)
          print("translated segments::", result_diarize['segments'])
        ## Write target segment and srt to file
        segments_to_srt(result_diarize['segments'], f'{target_media_output_basename}.srt')
        segments_to_txt(result_diarize['segments'], f'{target_media_output_basename}.txt')
        with open(f'{target_media_output_basename}.json', 'a', encoding='utf-8') as srtFile:
          srtFile.write(json.dumps(result_diarize['segments']))
        # ## Sort segments by speaker
        # result_diarize['segments'] = sorted(result_diarize['segments'], key=lambda x: x['speaker'])
        print("Translation complete")

        # 5. TTS target language
        progress(0.7, desc="Text_to_speech...")
        audio_files = []
        speakers_list = []


        
        N_JOBS = os.getenv('TTS_JOBS', round(CUDA_MEM*0.5/1000000000) if CUDA_MEM else 1)
        print("Start TTS:: concurrency =", N_JOBS)
        with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(N_JOBS) if self.max_speakers == 1 else 1):
          tts_results = Parallel(verbose=100)(delayed(self.tts)(segment, TRANSLATE_AUDIO_TO, speaker_to_voice, speaker_to_speed) for (segment) in tqdm(result_diarize['segments']))
        
        if os.getenv('UPSAMPLING_ENABLE', '') == "true":
          progress(0.75, desc="Upsampling...")
          print("Start Upsampling::")
          with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=1):
            tts_results = Parallel(verbose=100)(delayed(self.upsampling)(file) for (file) in tts_results)
          global upsampler
          upsampler = None; gc.collect(); torch.cuda.empty_cache()
        # tts_results = []
        # for segment in tqdm(result_diarize['segments']):
        #   tts_result = tts(segment, speaker_to_voice, TRANSLATE_AUDIO_TO, t2s_method, match_length)
        #   tts_results.append(tts_result)
          
        audio_files = [result[0] for result in tts_results]
        speakers_list = [result[1] for result in tts_results]
        print("audio_files:",len(audio_files))
        print("speakers_list:",len(speakers_list))
        
        # 6. Convert to target voices
        if self.vc_method == 'SVC':
            progress(0.80, desc="Applying SVC customized voices...")
            print("start SVC::")
            svc_voices(speakers_list, audio_files, speaker_to_vc)
            
        if self.vc_method == 'RVC':
            progress(0.80, desc="Applying RVC customized voices...")
            print("start RVC::")
            rvc_voices(speakers_list, audio_files, speaker_to_vc)

        if self.vc_method == 'OpenVoice':
            progress(0.80, desc="Applying OVC customized voices...")
            print("start OVC::")
            ov = OpenVoice()
            ov.batch_convert(tts_results, speaker_to_voice, speaker_to_vc)
            del ov
            
        # replace files with the accelerates
        os.system("mv -f audio2/audio/*.wav audio/")

        os.system(f"rm -rf {translated_output_file}")
        
        # 7. Join target language audio files
        progress(0.85, desc="Creating final translated media...")
        create_translated_audio(result_diarize, audio_files, translated_output_file, self.match_start)

        # # 8. Transribe target language for smaller chunk
        # print("Start transcribing target language::")
        # progress(0.90, desc="Transcribing target language...")
        # with capture.capture_output() as cap:
        #   model = whisperx.load_model(
        #       WHISPER_MODEL_SIZE,
        #       device,
        #       compute_type=compute_type,
        #       language= SOURCE_LANGUAGE,
        #       )
        #   del cap
        # audio = whisperx.load_audio(translated_output_file)
        # result = model.transcribe(WHISPER_MODEL_SIZE, audio, batch_size=batch_size, chunk_size=chunk_size)
        # ## Write target segment and srt to file
        # segments_to_srt(result['segments'], f'{target_media_output_basename}.srt')
        # gc.collect(); torch.cuda.empty_cache(); del model
        # print("Transcribe target language complete::", len(result["segments"]),result["segments"])

        # 8. Combine final audio and video
        print("Mixing source and target voices::")
        progress(0.95, desc="Mixing final video...")
        os.system(f"rm -rf {mix_audio}")  
        # TYPE MIX AUDIO
        if self.AUDIO_MIX_METHOD == 'Adjusting volumes and mixing audio':
            # volume mix
            os.system(f'ffmpeg -y -i {audio_wav} -i {translated_output_file} -filter_complex "[0:0]volume=0.15[a];[1:0]volume=1.90[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio}')
        else:
            try:
                # background mix
                os.system(f'ffmpeg -i {audio_wav} -i {translated_output_file} -filter_complex "[1:a]asplit=2[sc][mix];[0:a][sc]sidechaincompress=threshold=0.003:ratio=20[bg]; [bg][mix]amerge[final]" -map [final] {mix_audio}')
            except:
                # volume mix except
                os.system(f'ffmpeg -y -i {audio_wav} -i {translated_output_file} -filter_complex "[0:0]volume=0.25[a];[1:0]volume=1.80[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio}')

        print("Mixing target audio and video::")
        os.system(f"rm -rf {media_output}")
        if is_video:
          os.system(f"ffmpeg -i {OutputFile} -i {mix_audio} -c:v copy -c:a aac -map 0:v -map 1:a -shortest {media_output}")
        os.remove(OutputFile)
        if media_input.startswith('/tmp'):
          os.remove(media_input)
        ## Archve all files and return output
        archive_path = os.path.join(Path(temp_dir).parent.absolute(), os.path.splitext(os.path.basename(media_output))[0])
        shutil.make_archive(archive_path, 'zip', temp_dir)
        shutil.rmtree(temp_dir)
        final_output = f"{archive_path}.zip"
        
        ### Copy to temporary directory
        try:
          target_dir = os.getenv('COPY_OUTPUT_DIR', '')
          if target_dir and os.path.isdir(target_dir):
            if media_input.startswith(gradio_temp_processing_dir) and len(input_dirs) > 0:
              most_matching_prefix = find_most_matching_prefix(input_dirs, media_input)
              target_dir = os.path.join(target_dir, os.path.dirname(media_input).replace(os.path.dirname(most_matching_prefix),"").strip('/')) if os.path.isdir(most_matching_prefix) else os.path.join(target_dir, "/".join(os.path.dirname(media_input).split('/')[3:]).strip('/'))
            subprocess.run(["mkdir", "-p", target_dir], capture_output=True, text=True)
            subprocess.run(["cp", final_output, target_dir], capture_output=True, text=True)
            # os.system(f"rm -rf '{final_output}'")
        except:
          print('copy to target dir failed')
        return final_output
      
    def create_open_voice(self, file_path, model_name):
      ov = OpenVoice()
      ov.create_voice(file_path, model_name)
      gr.Info(f'Created voice: {model_name}')
      del ov
      return None, None  
       
    def create_ui(self):
        self.app = gr.Blocks(title="VGM Translate", theme=theme, css=css)
        with self.app:
          with gr.Row():
            with gr.Column():
              gr.Markdown(title)
            if os.getenv('ENABLE_AUTH', '') == "true":
              with gr.Column():
                gr.Button("Logout", link="/logout", size="sm", icon=None, elem_id="logout_btn")
          gr.Markdown(description)
          with gr.Tabs():
            with gr.Tab("Audio Translation for a Video"):
                with gr.Row():
                    with gr.Column():
                        media_input = gr.Files(label="VIDEO|AUDIO", file_types=['audio','video'])
                        with gr.Row():
                          link_input = gr.Textbox(label="YT Link or OS Path",info="Example: M:\\warehouse\\video.mp4,https://www.youtube.com/watch?v=DrG2c1vxGwU", placeholder="URL goes here, seperate by comma...", scale=5)        
                          link_btn = gr.Button("Submit", size="sm", scale=1)
                        srt_input = gr.Files(label="SRT(Optional)", file_types=['.srt'])
                        # gr.ClearButton(components=[media_input,link_input,srt_input], size='sm')
                        with gr.Row():
                          match_length = gr.Checkbox(label="Enable",container=False, value=False, info='Match speech length of original language?', interactive=True)
                          match_length.change(lambda x: x, match_length, None, js="(x) => setStorage('match_length',x)")
                          match_start = gr.Checkbox(label="Enable",container=False, value=True, info='Match speech start time of origin language?', interactive=True)
                          match_start.change(lambda x: x, match_start, None, js="(x) => setStorage('match_start',x)")
                        ## media_input change function
                        # link = gr.HTML()
                        # media_input.change(submit_file_func, media_input, [media_input, link], show_progress='full')

                        with gr.Row():
                          SOURCE_LANGUAGE = gr.Dropdown(['Automatic detection', 'Arabic (ar)', 'Cantonese (yue)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='English (en)',label = 'Source language', info="This is the original language of the video", scale=1)
                          SOURCE_LANGUAGE.change(None, SOURCE_LANGUAGE, None, js="(v) => setStorage('SOURCE_LANGUAGE',v)")
                          TRANSLATE_AUDIO_TO = gr.Dropdown(['Arabic (ar)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='Vietnamese (vi)',label = 'Target language', info="Select the target language for translation", scale=1)
                          TRANSLATE_AUDIO_TO.change(None, TRANSLATE_AUDIO_TO, None, js="(v) => setStorage('TRANSLATE_AUDIO_TO',v)")
                        # line_ = gr.HTML("<hr>")
                        gr.Markdown("Select how many people are speaking in the video.")
                        min_speakers = gr.Slider(1, MAX_TTS, value=1, step=1, label="min_speakers", visible=False)
                        max_speakers = gr.Slider(1, MAX_TTS, value=1, step=1, label="Max speakers", elem_id="max_speakers")
                        max_speakers.change(None, max_speakers, None, js="(v) => setStorage('max_speakers',v)")
                        gr.Markdown("Select the voice you want for each speaker.")
                        with gr.Row() as tts_voice00_row:
                          tts_voice00 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 1', visible=True, elem_id="tts_voice00")
                          tts_speed00 = gr.Slider(0.5, 1.5, value=1, label="TTS Speed 1", step=0.02, elem_id="tts_speed00", interactive=True)
                          vc_voice00 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='VC Speaker 1', visible=False, elem_id="vc_voice00")
                          tts_voice00.change(None, tts_voice00, None, js="(v) => setStorage('tts_voice00',v)")
                          tts_speed00.change(None, tts_speed00, None, js="(v) => setStorage('tts_speed00',v)")
                          vc_voice00.change(None, vc_voice00, None, js="(v) => setStorage('vc_voice00',v)")
                        with gr.Row(visible=False) as tts_voice01_row:
                          tts_voice01 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 2', visible=True, elem_id="tts_voice01")
                          tts_speed01 = gr.Slider(0.5, 1.5, value=1, label="TTS Speed 2", step=0.02, elem_id="tts_speed01", interactive=True)
                          vc_voice01 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='VC Speaker 2', visible=False, elem_id="vc_voice01")
                          tts_voice01.change(None, tts_voice01, None, js="(v) => setStorage('tts_voice01',v)")
                          tts_speed01.change(None, tts_speed01, None, js="(v) => setStorage('tts_speed01',v)")
                          vc_voice01.change(None, vc_voice01, None, js="(v) => setStorage('vc_voice01',v)")
                        with gr.Row(visible=False) as tts_voice02_row:
                          tts_voice02 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 3', visible=True, elem_id="tts_voice02")
                          tts_speed02 = gr.Slider(0.5, 1.5, value=1, label="TTS Speed 3", step=0.02, elem_id="tts_speed02", interactive=True)
                          vc_voice02 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='VC Speaker 3', visible=False, elem_id="vc_voice02")
                          tts_voice02.change(None, tts_voice02, None, js="(v) => setStorage('tts_voice02',v)")
                          tts_speed02.change(None, tts_speed02, None, js="(v) => setStorage('tts_speed02',v)")
                          vc_voice02.change(None, vc_voice02, None, js="(v) => setStorage('vc_voice02',v)")
                        with gr.Row(visible=False) as tts_voice03_row:
                          tts_voice03 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 4', visible=True, elem_id="tts_voice03")
                          tts_speed03 = gr.Slider(0.5, 1.5, value=1, label="TTS Speed 4", step=0.02, elem_id="tts_speed03", interactive=True)
                          vc_voice03 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='VC Speaker 4', visible=False, elem_id="vc_voice03")
                          tts_voice03.change(None, tts_voice03, None, js="(v) => setStorage('tts_voice03',v)")
                          tts_speed03.change(None, tts_speed03, None, js="(v) => setStorage('tts_speed03',v)")
                          vc_voice03.change(None, vc_voice03, None, js="(v) => setStorage('vc_voice03',v)")
                        with gr.Row(visible=False) as tts_voice04_row:
                          tts_voice04 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 5', visible=True, elem_id="tts_voice04")
                          tts_speed04 = gr.Slider(0.5, 1.5, value=1, label="TTS Speed 5", step=0.02, elem_id="tts_speed04", interactive=True)
                          vc_voice04 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='VC Speaker 5', visible=False, elem_id="vc_voice04")
                          tts_voice04.change(None, tts_voice04, None, js="(v) => setStorage('tts_voice04',v)")
                          tts_speed04.change(None, tts_speed04, None, js="(v) => setStorage('tts_speed04',v)")
                          vc_voice04.change(None, vc_voice04, None, js="(v) => setStorage('vc_voice04',v)")
                        with gr.Row(visible=False) as tts_voice05_row:
                          tts_voice05 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 6', visible=True, elem_id="tts_voice05")
                          tts_speed05 = gr.Slider(0.5, 1.5, value=1, label="TTS Speed 6", step=0.02, elem_id="tts_speed05", interactive=True)
                          vc_voice05 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='VC Speaker 6', visible=False, elem_id="vc_voice05")
                          tts_voice05.change(None, tts_voice05, None, js="(v) => setStorage('tts_voice05',v)")
                          tts_speed05.change(None, tts_speed05, None, js="(v) => setStorage('tts_speed05',v)")
                          vc_voice05.change(None, vc_voice05, None, js="(v) => setStorage('vc_voice05',v)")
                          
                        def update_speaker_visibility(value):
                            visibility_dict = {
                                f'tts_voice{i:02d}_row': gr.update(visible=i < value) for i in range(6)
                            }
                            return [value for value in visibility_dict.values()]
                        max_speakers.change(update_speaker_visibility, max_speakers, [tts_voice00_row, tts_voice01_row, tts_voice02_row, tts_voice03_row, tts_voice04_row, tts_voice05_row])
                        
                        with gr.Column():
                              with gr.Accordion("Advanced Settings", open=False):
                                  AUDIO_MIX = gr.Dropdown(['Mixing audio with sidechain compression', 'Adjusting volumes and mixing audio'], value='Adjusting volumes and mixing audio', label = 'Audio Mixing Method', info="Mix original and translated audio files to create a customized, balanced output with two available mixing modes.")
                                  # gr.HTML("<hr>")
                                  gr.Markdown("Default configuration of Whisper.")
                                  with gr.Row():
                                    WHISPER_MODEL_SIZE = gr.Dropdown(['tiny', 'base', 'base.en', 'small','small.en', 'medium', 'medium.en', 'large-v1', 'large-v2', 'large-v3'], value=whisper_model_default, label="Whisper model",  scale=1)
                                    WHISPER_MODEL_SIZE.change(None, WHISPER_MODEL_SIZE, None, js="(v) => setStorage('WHISPER_MODEL_SIZE',v)")
                                    compute_type = gr.Dropdown(list_compute_type, value=compute_type_default, label="Compute type",  scale=1)
                                    compute_type.change(None, compute_type, None, js="(v) => setStorage('compute_type',v)")
                                  with gr.Row():
                                    batch_size = gr.Slider(1, 32, value=16, label="Batch size", step=1)
                                    batch_size.change(None, batch_size, None, js="(v) => setStorage('batch_size',v)")
                                    chunk_size = gr.Slider(2, 30, value=5, label="Chunk size", step=1)
                                    chunk_size.change(None, chunk_size, None, js="(v) => setStorage('chunk_size',v)")
                                  # gr.HTML("<hr>")
                                  # MEDIA_OUTPUT_NAME = gr.Textbox(label="Translated file name" ,value="media_output.mp4", info="The name of the output file")
                                  preview = gr.Checkbox(label="Preview", visible=False, info="Preview cuts the video to only 10 seconds for testing purposes. Please deactivate it to retrieve the full video duration.")
                        
                        ## update_output_filename if media_input or TRANSLATE_AUDIO_TO change
                        # def update_output_filename(file,lang):
                        #     file_name, file_extension = os.path.splitext(os.path.basename(file.name.strip().replace(' ','_')))
                        #     output_name = f"{file_name}-{LANGUAGES[lang]}{file_extension}"
                        #     return gr.update(value=output_name)
                        # media_input.change(update_output_filename, [media_input,TRANSLATE_AUDIO_TO], [MEDIA_OUTPUT_NAME])
                        # TRANSLATE_AUDIO_TO.change(update_output_filename, [media_input,TRANSLATE_AUDIO_TO], [MEDIA_OUTPUT_NAME])
                        
                    with gr.Column(variant='compact'):
                        def update_output_list():
                          global total_input
                          global total_output
                          return total_output if len(total_output) < len(total_input) else []
                        with gr.Row():
                            media_btn = gr.Button("TRANSLATE", )
                        with gr.Row():    
                            media_output = gr.Files(label="PROGRESS BAR") #gr.Video()
                        with gr.Row():
                            tmp_output = gr.Files(label="TRANSLATED VIDEO", every=10, value=update_output_list) #gr.Video()
                      
                        ## Clear Button
                        def reset_param():
                          global total_input
                          global total_output
                          global list_ovc
                          total_input = []
                          total_output = []
                          self.input_dirs = []
                          os.system(f'rm -rf {os.path.join(tempfile.gettempdir(), "gradio-vgm")}/*')
                          os.system(f'rm -rf {os.path.join(tempfile.gettempdir(), "vgm-translate")}/*')
                          list_ovc = [voice for voice in os.listdir(os.path.join("model","openvoice","target_voice")) if os.path.isdir(os.path.join("model","openvoice","target_voice", voice))]
                          return gr.update(label="PROGRESS BAR", visible=True), gr.update(label="TRANSLATED VIDEO", visible=True)
                        with gr.Row():
                          clear_btn = gr.ClearButton(components=[media_input,link_input,srt_input,media_output,tmp_output], size='sm')
                          clear_btn.click(reset_param,[],[media_output,tmp_output, ])
                        line_ = gr.HTML("<hr>")
                        if os.getenv("YOUR_HF_TOKEN") == None or os.getenv("YOUR_HF_TOKEN") == "":
                          HFKEY = gr.Textbox(visible= True, label="HF Token", info="One important step is to accept the license agreement for using Pyannote. You need to have an account on Hugging Face and accept the license to use the models: https://huggingface.co/pyannote/speaker-diarization and https://huggingface.co/pyannote/segmentation. Get your KEY TOKEN here: https://hf.co/settings/tokens", placeholder="Token goes here...")
                        else:
                          HFKEY = gr.Textbox(visible= False, label="HF Token", info="One important step is to accept the license agreement for using Pyannote. You need to have an account on Hugging Face and accept the license to use the models: https://huggingface.co/pyannote/speaker-diarization and https://huggingface.co/pyannote/segmentation. Get your KEY TOKEN here: https://hf.co/settings/tokens", placeholder="Token goes here...")

                        # gr.Examples(
                        #     examples=[
                        #         [
                        #             "./assets/Video_main.mp4",
                        #             "",
                        #             False,
                        #             "large-v3",
                        #             16,
                        #             "float16",
                        #             "Spanish (es)",
                        #             "English (en)",
                        #             1,
                        #             2,
                        #             'en-AU-WilliamNeural-Male',
                        #             'en-CA-ClaraNeural-Female',
                        #             'en-GB-ThomasNeural-Male',
                        #             'en-GB-SoniaNeural-Female',
                        #             'en-NZ-MitchellNeural-Male',
                        #             'en-GB-MaisieNeural-Female',
                        #             "media_output.mp4",
                        #             'Adjusting volumes and mixing audio',
                        #         ],
                        #     ],
                        #     fn=translate_from_media,
                        #     inputs=[
                        #     media_input,
                        #     HFKEY,
                        #     PREVIEW,
                        #     WHISPER_MODEL_SIZE,
                        #     batch_size,
                        #     compute_type,
                        #     SOURCE_LANGUAGE,
                        #     TRANSLATE_AUDIO_TO,
                        #     min_speakers,
                        #     max_speakers,
                        #     tts_voice00,
                        #     tts_voice01,
                        #     tts_voice02,
                        #     tts_voice03,
                        #     tts_voice04,
                        #     tts_voice05,
                        #     MEDIA_OUTPUT_NAME,
                        #     AUDIO_MIX,
                        #     ],
                        #     outputs=[media_output],
                        #     cache_examples=False,
                        # )

            with gr.Tab("Settings"):
                with gr.Column():
                  with gr.Accordion("S2T - T2T - T2S", open=False):
                    with gr.Row():
                      s2t_method = gr.Dropdown(["Whisper"], label='S2T', value=user_settings['s2t'], visible=True, elem_id="s2t_method", interactive=True)
                      t2t_method = gr.Dropdown(["Google", "VB", "T5", "LLM"], label='T2T', value=user_settings['t2t'], visible=True, elem_id="t2t_method",interactive=True)
                      t2s_method = gr.Dropdown(["GTTS", "EdgeTTS", "PiperTTS","VietTTS"], label='T2S', value=user_settings['t2s'], visible=True, elem_id="t2s_method",interactive=True)
                      vc_method = gr.Dropdown(["None", "SVC", "RVC", "OpenVoice"], label='Voice Conversion', value=user_settings['vc'], visible=True, elem_id="vc_method",interactive=True)
                      s2t_method.change(None, s2t_method, None, js="(v) => setStorage('s2t_method',v)")
                      t2t_method.change(None, t2t_method, None, js="(v) => setStorage('t2t_method',v)")
                      t2s_method.change(None, t2s_method, None, js="(v) => setStorage('t2s_method',v)")
                      vc_method.change(None, vc_method, None, js="(v) => setStorage('vc_method',v)")
                    ## update t2s method
                    def update_vc_list(method):
                      visible = True if method != "None" else False
                      # print("method::", method, language, media_input)
                      match method:
                        case 'SVC':
                          list_vc = list_svc
                        case 'RVC':
                          list_vc = list_rvc
                        case 'OpenVoice':
                          list_vc = list_ovc
                        case _:
                          list_vc = [""]
                      visibility_dict = {
                          f'vc_voice{i:02d}': gr.update(choices=list_vc, value=list_vc[0], visible=visible) for i in range(6)
                      }
                      return [value for value in visibility_dict.values()]               
                    vc_method.change(update_vc_list, [vc_method], [vc_voice00, vc_voice01, vc_voice02, vc_voice03, vc_voice04, vc_voice05])
                    
                    ## update t2s method
                    def update_t2s_list(method, language):
                      print("method::", method, language)
                      match method:
                        case 'VietTTS':
                          list_tts = list_vtts
                        case 'EdgeTTS':
                          list_tts = [ x for x in list_etts if x.startswith(LANGUAGES[language])]
                        case 'PiperTTS':
                          list_tts = [ x for x in list_ptts if x.startswith(LANGUAGES[language])]
                        case _:
                          list_tts = list_gtts
                      visibility_dict = {
                          f'tts_voice{i:02d}': gr.update(choices=list_tts, value=list_tts[0]) for i in range(6)
                      }
                      return [value for value in visibility_dict.values()]
                    t2s_method.change(update_t2s_list, [t2s_method, TRANSLATE_AUDIO_TO], [tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05])
                    TRANSLATE_AUDIO_TO.change(update_t2s_list, [t2s_method, TRANSLATE_AUDIO_TO], [tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05])
                    
                  with gr.Accordion("LLM Settings", open=True) as LLM_Setting:
                    with gr.Row():
                      llm_url = gr.Textbox(label="LLM Endpoint", placeholder="LLM Endpoint goes here...", value=user_settings['llm_url'], elem_id="llm_url", scale=5)
                      llm_model = gr.Dropdown(label="LLM Model", choices=user_settings['llm_models'], value=user_settings['llm_model'], elem_id="llm_model",scale=5)        
                      llm_temp = gr.Slider(0.1, 1, value=0.3, step=0.1, label="Temparature",scale=5, interactive=True)
                      llm_k = gr.Slider(10, 3000, value=3000, step=10, label="K",scale=5, interactive=True)
                      llm_refresh = gr.Button("Refresh", scale=2)
                      ## Config LLM Settings
                      def update_llm_model(llm_url, s2t_method, t2t_method, t2s_method, vc_method):
                        models = get_llm_models(llm_url)
                        user_settings['s2t'] = s2t_method
                        user_settings['t2t'] = t2t_method
                        user_settings['t2s'] = t2s_method
                        user_settings['vc'] = vc_method
                        user_settings['llm_url'] = llm_url
                        user_settings['llm_models'] = models
                        user_settings['llm_model'] = models[0]
                        save_settings(settings=user_settings)
                        return gr.update(choices=models)
                      llm_url.blur(update_llm_model, [llm_url], [llm_model])
                      llm_url.change(None, llm_url, None, js="(v) => setStorage('llm_url',v)")
                      llm_model.change(None, llm_model, None, js="(v) => setStorage('llm_model',v)")
                      llm_temp.change(None, llm_temp, None, js="(v) => setStorage('llm_temp',v)")
                      llm_k.change(None, llm_k, None, js="(v) => setStorage('llm_k',v)")
                      llm_refresh.click(update_llm_model, [llm_url,s2t_method,t2t_method,t2s_method,vc_method], [llm_model])
                      
                  def update_llm_setting(method):
                    return gr.update(visible=method=='LLM')
                  t2t_method.change(update_llm_setting, t2t_method, LLM_Setting)

                  with gr.Accordion("Open Voice", visible=False) as open_voice_accordion:
                    ov_file = gr.File(label="Upload audio file", file_types=["audio"])
                    ov_name = gr.Textbox(label="Model Name")
                    ov_btn = gr.Button(value="Create Voice", variant="primary")
                  def update_voice_conversion(method):
                    return gr.update(visible=method=='OpenVoice')
                  vc_method.change(update_voice_conversion, [vc_method], [open_voice_accordion])                     

            with gr.Tab("Help"):
                gr.Markdown(tutorial)
                # gr.Markdown(news)

            # with gr.Accordion("Logs", open = False):
            #     logs = gr.Textbox()
            #     self.app.load(read_logs, None, logs, every=1)
            def update_output_visibility(self):
              return gr.update(label="TRANSLATED VIDEO"),gr.update(visible=False)
            
            # run
            ov_btn.click(self.create_open_voice, inputs=[ov_file, ov_name], outputs=[ov_file, ov_name])
            link_btn.click(self.handle_link_input, inputs=[link_input], outputs=[media_input, link_input])
            media_btn.click(self.batch_preprocess, inputs=[
                media_input,
                srt_input,
                s2t_method,
                t2t_method,
                t2s_method,
                vc_method,
                llm_url,
                llm_model,
                llm_temp,
                llm_k,
                match_length,
                match_start,
                HFKEY,
                preview,
                WHISPER_MODEL_SIZE,
                batch_size,
                chunk_size,
                compute_type,
                SOURCE_LANGUAGE,
                TRANSLATE_AUDIO_TO,
                min_speakers,
                max_speakers,
                tts_voice00,
                tts_voice01,
                tts_voice02,
                tts_voice03,
                tts_voice04,
                tts_voice05,
                tts_speed00,
                tts_speed01,
                tts_speed02,
                tts_speed03,
                tts_speed04,
                tts_speed05,
                vc_voice00,
                vc_voice01,
                vc_voice02,
                vc_voice03,
                vc_voice04,
                vc_voice05,
                AUDIO_MIX,
                ], outputs=media_output, concurrency_limit=1).then(
                #time.sleep(2),
                fn=update_output_visibility,
                inputs=[],
                outputs=[media_output,tmp_output]
                )
                
          self.app.load(
              None,
              inputs=None,
              outputs=[
              s2t_method,
              t2t_method,
              t2s_method,
              vc_method,
              llm_url,
              llm_model,
              llm_temp,
              llm_k,
              max_speakers,
              tts_voice00,
              tts_speed00,
              vc_voice00,
              tts_voice01,
              tts_speed01,
              vc_voice01,
              tts_voice02,
              tts_speed02,
              vc_voice02,
              tts_voice03,
              tts_speed03,
              vc_voice03,
              tts_voice04,
              tts_speed04,
              vc_voice04,
              tts_voice05,
              tts_speed05,
              vc_voice05,
              match_length,
              match_start,
              SOURCE_LANGUAGE,
              TRANSLATE_AUDIO_TO,
              WHISPER_MODEL_SIZE,
              compute_type,
              batch_size,
              chunk_size
                ],
              js=get_local_storage,
          )
          self.app.queue()

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
  mp.set_start_method('spawn', force=True)
  
  # os.system('rm -rf *.wav *.mp3 *.wav *.mp4')
  os.system('mkdir -p downloads')
  os.system(f'rm -rf {gradio_temp_dir}/*')
  os.system(f'rm -rf {os.path.join(tempfile.gettempdir(), "vgm-translate")}/*')
  port=6864
  os.system(f'rm -rf audio2/SPEAKER_* audio2/audio/* audio.out audio/*')
  print('Working in:: ', device)
  mainApp = Main()
  
  ## Enable Auth
  if os.getenv('ENABLE_AUTH', '') == "true":
    root = gr.mount_gradio_app(root, mainApp.app, path="/app", auth_dependency=is_authenticated)
    asyncio.run(init_database())
    uvicorn.run(root, host="0.0.0.0", port=port)
  else:
    auth_user = os.getenv('AUTH_USER', '')
    auth_pass = os.getenv('AUTH_PASS', '')
    
    mainApp.app.launch(
      auth=(auth_user, auth_pass) if auth_user != '' and auth_pass != '' else None,
      show_api=False,
      debug=True,
      inbrowser=True,
      show_error=True,
      server_name="0.0.0.0",
      server_port=port,
      # quiet=True,
      share=False)