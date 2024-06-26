from gtts import gTTS
import re
import os
import edge_tts
import asyncio
# import nest_asyncio
from vietTTS.vietTTS import text_to_speech, normalize
from utils.tts_utils import piper_tts
from utils.xtts import xtts
from pydub import AudioSegment
# import torch
# import gc


def split_long_speech(tts_text, tts_voice, tts_speed, filename, language, t2s_method):
    split_texts = []
    words = tts_text.split()
    mid_index = len(words) // 2
    # Find the split point around the middle index
    left_part = " ".join(words[:mid_index])
    right_part = " ".join(words[mid_index:])
    split_texts.append(left_part)
    split_texts.append(right_part)
    
    results = []
    for index, text in enumerate(split_texts):
      name, ext = os.path.splitext(filename)
      filepath = f"{name}{index}{ext}"
      make_voice_gradio(text, tts_voice, tts_speed, filepath, language, t2s_method)
      results.append(filepath)

    audio = AudioSegment.from_file(results[0], format="wav")
    # Concatenate the remaining audio files
    for filepath in results[1:]:
        wav = AudioSegment.from_file(filepath, format="wav")
        audio += wav
    # create the output file
    audio.export(filename, format="wav")
      

def make_voice_gradio(tts_text, tts_voice, tts_speed, filename, language, t2s_method):
    print("make_voice_gradio::",tts_text, tts_voice, filename, language, t2s_method)
    try:
      if language == 'vi':
        tts_text = tts_text.lower()
        tts_text = normalize(tts_text)
      if t2s_method == "GTTS": 
        tts = gTTS(tts_text, lang=language)
        tts.save(filename)
        return
      if t2s_method == "EdgeTTS":
        asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
        return
      if t2s_method == "PiperTTS":
        piper_tts(tts_text, tts_voice, tts_speed, filename)
        return
      if t2s_method == "VietTTS" and language == "vi":
        print("vietTTS::")
        text_to_speech(tts_text, filename, tts_voice, tts_speed if tts_speed else 1)
        return
      if t2s_method == "XTTS":
        print("xTTS::")
        if len(tts_text) > 350:
          split_long_speech(tts_text, tts_voice, tts_speed, filename, language, t2s_method)
        else:
          xtts(tts_text, filename, tts_voice, tts_speed, language)
        return   
    except Exception as error:
      print("tts error:", error, tts_text)
    return
 
