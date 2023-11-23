from gtts import gTTS
import re
import edge_tts
import asyncio
import nest_asyncio
from vietTTS.vietTTS import text_to_speech
# def make_voice(tts_text, tts_voice, filename,language):
#     #print(tts_text, filename)
#     try:
#       nest_asyncio.apply()
#       asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
#     except:
#       try:
#           tts = gTTS(tts_text, lang=language)
#           tts.save(filename)
#           print(f'No audio was received. Please change the tts voice for {tts_voice}. TTS auxiliary will be used in the segment')
#       except:
#         tts = gTTS('a', lang=language)
#         tts.save(filename)
#         print('Error: Audio will be replaced.')

def make_voice_gradio(tts_text, tts_voice, filename, language, t2s_method):
    print("make_voice_gradio::",tts_text, tts_voice, filename, language, t2s_method)
    if t2s_method == "VietTTS" and language == "vi" :
      try:
        text_to_speech(tts_text, filename, tts_voice)
      except Exception as error:
        print("tts error:", error, tts_text)
        tts = gTTS('a', lang=language)
        tts.save(filename)
    else:
      try:
        asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
      except:
        try:
          tts = gTTS(tts_text, lang=language)
          tts.save(filename)
          print(f'No audio was received. Please change the tts voice for {tts_voice}. TTS auxiliary will be used in the segment')
        except:
          tts = gTTS('a', lang=language)
          tts.save(filename)
          print('Error: Audio will be replaced.')
 
