# import re
import os
from tqdm import tqdm
from translate_text_processor import titlecase_with_dash
# from vb_translate import vb_translate
from langdetect import detect
from llm_translate import LLM
# from llama_translate import LLM as Llama
from llm_correct_vi import LLM as LLMCorrect
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
def translate_text(segments, SOURCE_LANGUAGE="", TARGET_LANGUAGE="", t2t_method="", llm_endpoint="", llm_model="", llm_temp=0.6, llm_k=5):
    print("start translate_text::", t2t_method, llm_endpoint, llm_model, llm_temp, llm_k)
    if t2t_method == "LLM":
      systemPrompt = "Think and translate English accurately into clear, natural, appropriate Vietnamese." if TARGET_LANGUAGE == "vi" else ""
      llm_endpoint = llm_endpoint if TARGET_LANGUAGE == "vi" else ""
      llm_model = llm_model if TARGET_LANGUAGE == "vi" else ""
      api_key = "EMPTY" if TARGET_LANGUAGE == "vi" else ""
      llm = LLM(systemPrompt=systemPrompt)
      llm_status = llm.initLLM(llm_endpoint, llm_model, api_key, llm_temp, llm_k)
      if llm_status:
        segments = llm.translate(segments=segments, source_lang=SOURCE_LANGUAGE, target_lang=TARGET_LANGUAGE)
        for index, segment in enumerate(segments):
          segments[index]['text'] = post_process_text(segments[index]['text'])
        llm.stop()
        del llm
    
    # ## Last option to check if any non-translated sentences left then using Google translator
    # google_translator = GoogleTranslator(source='auto', target=TARGET_LANGUAGE)
    # for line in tqdm(range(len(segments))):
    #   # print("gg_translator::")
    #   try:
    #     text = segments[line]['text']
    #     if text and TARGET_LANGUAGE not in detect(text.strip()):
    #       translated_line = google_translator.translate(text.strip())
    #       # print("translate_text_in::", TARGET_LANGUAGE, t2t_method,f'{text}\n{translated_line}')
    #       segments[line]['text'] = post_process_text(translated_line)
    #   except Exception as e:
    #     pass

    return segments

def grammar_correction(source_segments, target_segments, SOURCE_LANGUAGE="", TARGET_LANGUAGE="", llm_endpoint="", llm_model="", llm_temp=0.6, llm_k=5):
    ## Implement grammar correction for vietnamese using llm
    if TARGET_LANGUAGE == "vi":
      systemPrompt="Review the English–Vietnamese translation pair, fix any errors, and produce a clear, natural Vietnamese version."
      llm_endpoint = llm_endpoint if TARGET_LANGUAGE == "vi" else ""
      llm_model = llm_model if TARGET_LANGUAGE == "vi" else ""
      api_key = "EMPTY" if TARGET_LANGUAGE == "vi" else ""
      llm = LLMCorrect(systemPrompt=systemPrompt)
      llm_status = llm.initLLM(
        endpoints=llm_endpoint, ## https://web.chattrust.ai/api
        model=llm_model, ## "deepseek-r1:70b"
        api_key=api_key, 
        temp=llm_temp,
        k=llm_k
      )
      if llm_status:
        segments = llm.translate(source_segments=source_segments, target_segments=target_segments, source_lang=SOURCE_LANGUAGE, target_lang=TARGET_LANGUAGE)
        for index, segment in enumerate(segments):
          segments[index]['text'] = post_process_text(segments[index]['text'])
        llm.stop()
        del llm
      else:
        pass
    
    # ## Implement grammar correction for vietnamese using Llamacpp
    # if TARGET_LANGUAGE == "vi":
    #   systemPrompt = "Sửa lỗi chính tả từ bản gốc sang bảng mới"
    #   llm = Llama(systemPrompt=systemPrompt)
    #   llm_status = llm.initLLM(temp=llm_temp, k=llm_k)
    #   if llm_status:
    #     segments = llm.translate(segments=segments)
    #     for index, segment in enumerate(segments):
    #       segments[index]['text'] = post_process_text(segments[index]['text'])
    #     del llm
    #   else:
    #     pass
    return segments