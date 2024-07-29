import re
import os
from tqdm import tqdm
from translate_text_processor import titlecase_with_dash
from vb_translate import vb_translate
from langdetect import detect
from llm_translate import LLM
from deep_translator import GoogleTranslator
import torch

from repl_dict import dictOfReplacementStrings
device = torch.device("cuda")

def post_process_text(text):
  text = titlecase_with_dash(text)
  for word, replacement in dictOfReplacementStrings.items():
    text = text.replace(word, replacement)
  return text

## Translate text using Google Translator
def translate_text(segments, TRANSLATE_AUDIO_TO="", t2t_method="", llm_endpoint="", llm_model="", llm_temp=0.5, llm_k=30):
    print("start translate_text::", segments)
    if t2t_method == "LLM" and TRANSLATE_AUDIO_TO == "vi":
      llm = LLM()
      llm_status = llm.initLLM(llm_endpoint, llm_model, llm_temp, llm_k)
      if llm_status:
        segments = llm.translate(segments)
        for index, segment in enumerate(segments):
          segments[index]['text'] = post_process_text(segments[index]['text'])
        del llm
      else:
        t2t_method = "VB"
      
    if t2t_method == "VB" and TRANSLATE_AUDIO_TO == "vi":
      print("vb_translator::", len(segments), "segments")
      source_text = "\n".join([ segment['text'] for segment in segments])
      translated_text = vb_translate(source_text.strip())
      print("vb_translator translated_text::", len(translated_text), "segments")
      for index, segment in enumerate(segments):
        segments[index]['text'] = post_process_text(translated_text[index])
    
    ## Last option to check if any non-translated sentences left then using Google translator
    google_translator = GoogleTranslator(source='auto', target=TRANSLATE_AUDIO_TO)
    for line in tqdm(range(len(segments))):
      # print("gg_translator::")
      try:
        text = segments[line]['text']
        if text and detect(text.strip()) != 'vi':
          translated_line = google_translator.translate(text.strip())
          # print("translate_text_in::", TRANSLATE_AUDIO_TO, t2t_method,f'{text}\n{translated_line}')
          segments[line]['text'] = post_process_text(translated_line)
      except Exception as e:
        pass
    return segments
