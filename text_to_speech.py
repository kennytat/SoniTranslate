from gtts import gTTS
import re
import os
import edge_tts
import asyncio
# import nest_asyncio
from vietTTS.vietTTS import normalize, VietTTS
from utils.tts_utils import piper_tts
from utils.xtts import XTTS
from utils.utils import split_and_join_by_comma
from pydub import AudioSegment
# import torch
# import gc



      
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
    return self.tts_client
        
  def split_long_speech(self, tts_text, tts_voice, tts_speed, filename, language, t2s_method, max_length=250):
      print("split_long_speech::")
      split_texts = split_and_join_by_comma(tts_text, max_length)
      results = []
      for index, text in enumerate(split_texts):
        name, ext = os.path.splitext(filename)
        filepath = f"{name}{index}{ext}"
        self.make_voice_gradio(text, tts_voice, tts_speed, filepath, language, t2s_method)
        results.append(filepath)

      audio = AudioSegment.from_file(results[0], format="wav")
      # Concatenate the remaining audio files
      for filepath in results[1:]:
          wav = AudioSegment.from_file(filepath, format="wav")
          audio += wav
      # create the output file
      audio.export(filename, format="wav")
   
  def make_voice_gradio(self, tts_text, tts_voice, tts_speed, filename, language, t2s_method):
      print("make_voice_gradio::",tts_text, tts_voice, filename, language, t2s_method)
      try:
        if language == 'vi':
          tts_text = tts_text.lower()
          tts_text = normalize(tts_text)
        if t2s_method == "GTTS" and self.tts_client and self.tts_client == t2s_method: 
          tts = gTTS(tts_text, lang=language)
          tts.save(filename)
          return
        if t2s_method == "EdgeTTS and self.tts_client and self.tts_client == t2s_method":
          asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
          return
        if t2s_method == "PiperTTS" and self.tts_client and self.tts_client == t2s_method:
          piper_tts(tts_text, tts_voice, tts_speed, filename)
          return
        if t2s_method == "VietTTS" and language == "vi" and self.tts_client and self.tts_client.name == t2s_method:
          print("vietTTS::")
          self.tts_client.text_to_speech(tts_text, filename, tts_voice, tts_speed if tts_speed else 1)
          return
        if t2s_method == "XTTS" and self.tts_client and self.tts_client.name == t2s_method:
          print("xTTS::")
          if len(tts_text) > 250:
            self.split_long_speech(tts_text, tts_voice, tts_speed, filename, language, t2s_method, 250)
          else:
            self.tts_client.text_to_speech(tts_text, filename, tts_voice, tts_speed, language)
          return   
      except Exception as error:
        print("tts error:", error, tts_text)
      return
  

  