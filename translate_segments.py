import re
import os
from tqdm import tqdm
from translate_text_processor import titlecase_with_dash
from vb_translate import vb_translate
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
def translate_text(segments, SOURCE_LANGUAGE="", TRANSLATE_AUDIO_TO="", t2t_method="", llm_endpoint="", llm_model="", llm_temp=0.5, llm_k=30):
    print("start translate_text::", segments)
    if t2t_method == "LLM":
      systemPrompt = "Bạn là AI có khả năng dịch thuật nội dung từ tiếng Anh một cách chính xác và rất dễ hiểu cho người Việt Nam. Hãy cẩn thận dịch và chọn từ ngữ cho phù hợp." if TRANSLATE_AUDIO_TO == "vi" else ""
      llm_endpoint = llm_endpoint if TRANSLATE_AUDIO_TO == "vi" else ""
      llm_model = llm_model if TRANSLATE_AUDIO_TO == "vi" else ""
      api_key = "EMPTY" if TRANSLATE_AUDIO_TO == "vi" else ""
      llm = LLM(systemPrompt=systemPrompt)
      llm_status = llm.initLLM(llm_endpoint, llm_model, api_key, llm_temp, llm_k)
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

def grama_correction(source_segments, target_segments, SOURCE_LANGUAGE="", TRANSLATE_AUDIO_TO="", llm_temp=0.3, llm_k=200):
    ## Implement grama correction for vietnamese using llm
    if TRANSLATE_AUDIO_TO == "vi":
      glossary_table="""God: Thiên Chúa
      Jehovah God, God Jehovah, Jehovah, LORD God: Thiên Chúa Hằng Hữu
      Christ: Đấng Cứu Thế, Chúa Cứu Thế
      Jesus: Chúa Giê-xu
      Jesus Christ: Chúa Cứu Thế Giê-xu
      Holy Spirit: Đức Thánh Linh, Chúa Thánh Linh, Thánh Linh
      Do not use any "Ki-tô" word in the corrected version but use "Cơ Đốc" instead."""  
      systemPrompt="""
      Given a source paragraph in English and its Vietnamese translation, please correct any errors in the translation to ensure it is accurate, faithful to the original meaning, clear, and natural-sounding for Vietnamese readers.
      Only revise parts that are unnatural or difficult to understand. Prioritize accuracy and fidelity to the English text while making the translation easy to comprehend.
      Use the glossary table to choose the most appropriate and consistent terms.""" + "\n\nGLOSSARY / DICTIONARY GUIDANCE:\n" + glossary_table
      
      llm = LLMCorrect(systemPrompt=systemPrompt)
      llm_status = llm.initLLM(
        endpoints="https://web.chattrust.ai/api", ## https://web.chattrust.ai/api
        model="deepseek-r1:70b", ## "deepseek-r1:70b"
        api_key=os.getenv("OW_API_KEY", ""), 
        temp=llm_temp,
        k=llm_k
      )
      if llm_status:
        segments = llm.translate(source_segments=source_segments, target_segments=target_segments, source_lang=SOURCE_LANGUAGE, target_lang=TRANSLATE_AUDIO_TO)
        for index, segment in enumerate(segments):
          segments[index]['text'] = post_process_text(segments[index]['text'])
        del llm
      else:
        pass
    
    # ## Implement grama correction for vietnamese using Llamacpp
    # if TRANSLATE_AUDIO_TO == "vi":
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