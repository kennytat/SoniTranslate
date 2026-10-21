from gtts import gTTS
import re
import edge_tts
import asyncio
# import nest_asyncio
from vietTTS.vietTTS import normalize
from vieneuTTS.vieneuTTS import VieneuTTS
from utils.gpu_tts_pool import GpuJobSlot, is_cuda_oom, release_cuda
from utils.tts_utils import piper_tts
from utils.xtts import XTTS
from utils.utils import split_and_join_by_comma, num_to_str
from pydub import AudioSegment
import torch
import gc
import numpy as np

class TTSClient():
  def __init__(self):
    self.tts_client = None
    self.current_tts_client = None

  def release(self):
    if self.tts_client is not None and hasattr(self.tts_client, "release"):
      self.tts_client.release()
    self.tts_client = None
    self.current_tts_client = None
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  def init_tts_client(self, client):
    if self.current_tts_client != client:
      self.release()
      self.current_tts_client = client
      match client:
        case "VietTTS":
          self.tts_client = VieneuTTS()
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
        # Static markers must be checked on the RAW text BEFORE normalization
        # (normalize turns "sil 3" -> "sil ba." breaking the engine's check).
        if re.match(r'^\s*(sil\s+\d+|♪)\s*\.?\s*$', tts_text or "", re.IGNORECASE):
          audio = self.tts_client.text_to_speech(tts_text.lower(), tts_voice, tts_speed if tts_speed else 1) \
              if (t2s_method == "VietTTS" and self.tts_client and self.tts_client.name == t2s_method) else None
          if audio is not None and filename:
            audio.export(filename, format="wav")
            return filename
        if language == 'vi':
          tts_text = tts_text.lower()
          tts_text = normalize(tts_text)
          tts_text = num_to_str(tts_text)

        with GpuJobSlot(t2s_method):
          if t2s_method == "GTTS" and self.tts_client and self.tts_client == t2s_method:
            audio = gTTS(tts_text, lang=language)
            if filename:
              audio.save(filename)
              sound = AudioSegment.from_mp3(filename)
              sound.export(filename, format="wav")
              return filename
            audio_bytes = b''.join(audio.stream())
            return (24000, np.frombuffer(audio_bytes, dtype=np.int16))
          if t2s_method == "PiperTTS" and self.tts_client and self.tts_client == t2s_method:
            audio = piper_tts(tts_text, tts_voice, tts_speed)
            if filename:
              audio.export(filename)
              return filename
            return (audio.frame_rate, np.array(audio.get_array_of_samples()))
          if t2s_method == "VietTTS" and language == "vi" and self.tts_client and self.tts_client.name == t2s_method:
            print("vietTTS::")
            audio = self.tts_client.text_to_speech(tts_text, tts_voice, tts_speed if tts_speed else 1)
            if filename:
              audio.export(filename, format="wav")
              return filename
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
            return (audio.frame_rate, np.array(audio.get_array_of_samples()))
      except Exception as error:
        print("tts error:", error, tts_text)
        release_cuda()
        if is_cuda_oom(error):
          raise
      return None
