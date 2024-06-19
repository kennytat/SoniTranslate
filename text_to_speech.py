from gtts import gTTS
import re
import edge_tts
import asyncio
# import nest_asyncio
from vietTTS.vietTTS import text_to_speech, normalize
from utils.tts_utils import piper_tts
from utils.xtts import xtts
# import torch
# import gc


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
        xtts(tts_text, filename, tts_voice, tts_speed, language)
        return   
    except Exception as error:
      print("tts error:", error, tts_text)
    return
 
