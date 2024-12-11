import re
import os
from tqdm import tqdm
from translate_text_processor import titlecase_with_dash
from vb_translate import vb_translate
from langdetect import detect
from llm_translate import LLM
from llama_translate import LLM as Llama
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
def translate_text(segments, SOURCE_LANGUAGE="", TRANSLATE_AUDIO_TO="", t2t_method="", llm_endpoint="", llm_model="", llm_temp=0.5, llm_k=30):
    print("start translate_text::", segments)
    if t2t_method == "LLM":
      systemPrompt = "Bạn là AI có khả năng dịch thuật nội dung từ tiếng Anh một cách chính xác và rất dễ hiểu cho người Việt Nam. Hãy cẩn thận dịch và chọn từ ngữ cho phù hợp." if TRANSLATE_AUDIO_TO == "vi" else ""
      llm_endpoint = llm_endpoint if TRANSLATE_AUDIO_TO == "vi" else ""
      llm_model = llm_model if TRANSLATE_AUDIO_TO == "vi" else ""
      api_key = "EMPTY" if TRANSLATE_AUDIO_TO == "vi" else ""
      llm = LLM(systemPrompt=systemPrompt)
      llm_status = llm.initLLM(endpoints=llm_endpoint, model=llm_model, api_key=api_key, temp=llm_temp, k=llm_k)
      print("llm_status::", llm_status)
      if llm_status:
        segments = llm.translate(segments=segments, source_lang=SOURCE_LANGUAGE, target_lang=TRANSLATE_AUDIO_TO)
        for index, segment in enumerate(segments):
          segments[index]['text'] = post_process_text(segments[index]['text'])
        del llm
      else:
        t2t_method = "VB" if TRANSLATE_AUDIO_TO == "vi" else "LLM"
      
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
        if text and TRANSLATE_AUDIO_TO not in detect(text.strip()):
          translated_line = google_translator.translate(text.strip())
          # print("translate_text_in::", TRANSLATE_AUDIO_TO, t2t_method,f'{text}\n{translated_line}')
          segments[line]['text'] = post_process_text(translated_line)
      except Exception as e:
        pass

    return segments

def grama_correction(segments, TRANSLATE_AUDIO_TO="", llm_temp=0.5, llm_k=30):
    ## Implement grama correction for vietnamese
    if TRANSLATE_AUDIO_TO == "vi":
      systemPrompt = "Sửa lỗi chính tả từ bản gốc sang bảng mới"
      llm = Llama(systemPrompt=systemPrompt)
      llm_status = llm.initLLM(temp=llm_temp, k=llm_k)
      if llm_status:
        segments = llm.translate(segments=segments)
        for index, segment in enumerate(segments):
          segments[index]['text'] = post_process_text(segments[index]['text'])
        del llm
      else:
        pass
    return segments