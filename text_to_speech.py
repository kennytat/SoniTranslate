from gtts import gTTS
import re
import os
import edge_tts
import asyncio
# import nest_asyncio
from vietTTS.vietTTS import normalize, VietTTS
from utils.tts_utils import piper_tts
from utils.xtts import XTTS
from utils.utils import split_and_join_by_comma, num_to_str
from pydub import AudioSegment
import shutil
import torch
import gc
from ovc_voice_main import OpenVoice
from pathlib import Path
from pydub import AudioSegment
import numpy as np
class TTSClient():
  def __init__(self):
    self.tts_client = None

  def init_tts_client(self, client):
      match client:
        case "VietTTS":
          self.tts_client = VietTTS()
        case "XTTS":
          self.tts_client = XTTS()
        case _:
          self.tts_client = client
        
  def split_long_speech(self, tts_text, tts_voice, tts_speed, language, t2s_method, max_length=200):
      print("split_long_speech::", len(tts_text))
      split_texts = split_and_join_by_comma(tts_text, max_length)
      results = []
      for index, text in enumerate(split_texts):
        result = self.make_voice_gradio(text, tts_voice, tts_speed, "", language, t2s_method)
        results.append(result)

      audio = results[0]
      # Concatenate the remaining audio files
      for wav in results[1:]:
          audio += wav
      return audio
   
  def make_voice_gradio(self, tts_text, tts_voice, tts_speed, filename, language, t2s_method):
      print("make_voice_gradio::", tts_text, tts_voice, filename, language, t2s_method)
      try:
        if language == 'vi':
          tts_text = tts_text.lower()
          tts_text = normalize(tts_text)
          tts_text = num_to_str(tts_text)

        if t2s_method == "GTTS" and self.tts_client and self.tts_client == t2s_method: 
          audio = gTTS(tts_text, lang=language)
          if filename:
            audio.save(filename)
            sound = AudioSegment.from_mp3(filename)
            sound.export(filename, format="wav")
            return filename
          else:
            audio_bytes = b''.join(audio.stream())
            return (24000, np.frombuffer(audio_bytes, dtype=np.int16))
        if t2s_method == "PiperTTS" and self.tts_client and self.tts_client == t2s_method:
          audio = piper_tts(tts_text, tts_voice, tts_speed)
          if filename:
            audio.export(filename)
            return filename
          else:
            return (audio.frame_rate, np.array(audio.get_array_of_samples()))
        if t2s_method == "VietTTS" and language == "vi" and self.tts_client and self.tts_client.name == t2s_method:
          print("vietTTS::")
          audio = self.tts_client.text_to_speech(tts_text, tts_voice, tts_speed if tts_speed else 1)
          if filename:
            audio.export(filename, format="wav")
            return filename
          else:
            return (audio.frame_rate, np.array(audio.get_array_of_samples()))
        if t2s_method == "XTTS" and self.tts_client and self.tts_client.name == t2s_method:
          print("xTTS::")
          if len(tts_text) > 250 and "," in tts_text:
            audio = self.split_long_speech(tts_text, tts_voice, tts_speed, language, t2s_method, 200)
          else:
            audio = self.tts_client.text_to_speech(tts_text, tts_voice, tts_speed, language)
          if filename:
            audio.export(filename, format="wav")
            return filename
          else:
            return (audio.frame_rate, np.array(audio.get_array_of_samples()))
      except Exception as error:
        print("tts error:", error, tts_text)
      return None
  
def start_svc_voice(input_path, vc_voice):
  print("start svc_voice::", input_path, vc_voice)
  model_path = os.path.join(vc_voice, "G.pth")
  config_path = os.path.join(vc_voice, "config.json")
  if os.path.isfile(input_path):
    basename, ext = os.path.splitext(input_path)
    output_file = f"{basename}.out{ext}"
    os.system(f'svc infer -m {model_path} -c {config_path} {input_path}')
    shutil.move(output_file, input_path)
  if os.path.isdir(input_path):
    output_dir = f'{input_path}.out'
    os.system(f'svc infer -re -m {model_path} -c {config_path} {input_path}')
    if os.path.exists(input_path): shutil.rmtree(input_path, ignore_errors=True)
    shutil.move(output_dir, input_path)
  gc.collect(); torch.cuda.empty_cache()

def start_ovc_voice(input_path, tts_voice, vc_voice):
  print("start vc_voice::", input_path, tts_voice, vc_voice)
  ov = OpenVoice()
  if os.path.isfile(input_path):
    basename, ext = os.path.splitext(input_path)
    output_file = f"{basename}.out{ext}"
    ov.convert_voice(file, tts_voice, output_file, vc_voice)
    shutil.move(output_file, input_path)
  if os.path.isdir(input_path):
    output_dir = f"{input_path}-out"
    os.system(f"mkdir -p {output_dir}")
    for file in sorted(Path(input_path).glob("*.wav")):
      output_file = str(file).replace(input_path, output_dir)
      print("openvoice::", output_file)
      ov.convert_voice(file, tts_voice, output_file, vc_voice)
    if os.path.exists(input_path): shutil.rmtree(input_path, ignore_errors=True)
    shutil.move(output_dir, input_path)
  del ov
  gc.collect(); torch.cuda.empty_cache()

def voice_conversion(input_path, tts_voice, vc_method, vc_voice):
  print("Start Voice Convertion::")
  if vc_method == "SVC":
    svc_voice_ckpt_dir = os.path.join(os.getcwd(), "model", "svc", vc_voice)
    start_svc_voice(input_path, svc_voice_ckpt_dir)
  if vc_method == "OpenVoice":
    start_ovc_voice(input_path, tts_voice, vc_voice)    
  