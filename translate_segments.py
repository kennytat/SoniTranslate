import re
import os
from tqdm import tqdm
from translate_text_processor import post_process_text_vi, titlecase_with_dash
from vb_translate import vb_translate
from langdetect import detect
from llm_translate import LLM
from deep_translator import GoogleTranslator
import torch
import gc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from repl_dict import dictOfReplacementStrings
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
def post_process_text(text):
  text = titlecase_with_dash(text)
  for word, replacement in dictOfReplacementStrings.items():
    text = text.replace(word, replacement)
  return text

def t5_translator(input_text: str, tokenizer, model):
    def process_batch(sentences):
        sentences = [ text if text.endswith(".") else text + "." for text in sentences]
        input_ids = tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True, truncation=True).to(torch.device(device))
        output_ids = model.generate(input_ids.input_ids, max_length=20000)  # Set max_length to a larger value
        output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # print("output_texts::::", sentences, output_texts)
        output_texts = [re.sub(r"^(vi|vn|en)\:? ", "", text) for text in output_texts]
        output_texts = [post_process_text_vi(text) for text in output_texts]
        return output_texts
    result = process_batch(input_text.split("\n"))
    return "\n".join(result)

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
    
    if t2t_method == "T5" and TRANSLATE_AUDIO_TO == "vi":
      ## T5 translator - instantiate the pre-trained English-to-Vietnamese Transformer model
      BASE_MODEL = "./model/envit5-translation"
      LORA_WEIGHT = "./model/envit5-translation-lora-38500"
      t5_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
      t5_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
      t5_model = PeftModel.from_pretrained(t5_model, LORA_WEIGHT).cuda()
      t5_model.eval()
      if isinstance(segments, str):
        segments = t5_translator(segments.strip(), t5_tokenizer, t5_model)
        segments = post_process_text(segments)
      else:
        for line in tqdm(range(len(segments))):
          text = segments[line]['text']
          print("t5_translator::")
          translated_line = t5_translator(text.strip(), t5_tokenizer, t5_model)
          print("translate_text_in::", TRANSLATE_AUDIO_TO, t2t_method,f'{text}\n{translated_line}')
          segments[line]['text'] = post_process_text(translated_line)
      gc.collect(); torch.mps.empty_cache(); torch.cuda.empty_cache(); del t5_tokenizer; del t5_model
    elif t2t_method == "LLM" and TRANSLATE_AUDIO_TO == "vi":
      llm = LLM()
      llm.initLLM(llm_endpoint, llm_model)
      segments = llm.translate(segments)
      for index, segment in enumerate(segments):
        segments[index]['text'] = post_process_text(segments[index]['text'])
      del llm
    else:
      pass
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
