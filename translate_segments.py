import re
import os
from tqdm import tqdm
from translate_text_processor import post_process_text_vi
from vb_translate import vb_translate
from deep_translator import GoogleTranslator

import torch
import gc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

device = torch.device("cuda")
BASE_MODEL = "./model/envit5-translation"
LORA_WEIGHT = "./model/envit5-translation-lora-38500"

def t5_translator(input_text: str, tokenizer, model):
    def process_batch(sentences):
        sentences = [ text if text.endswith(".") else text + "." for text in sentences]
        input_ids = tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        output_ids = model.generate(input_ids.input_ids, max_length=20000)  # Set max_length to a larger value
        output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # print("output_texts::::", sentences, output_texts)
        output_texts = [re.sub(r"^(vi|vn|en)\:? ", "", text) for text in output_texts]
        return post_process_text_vi(output_texts)
    result = process_batch(input_text.split("\n"))
    return "\n".join(result)

## Translate text using Google Translator
def translate_text(segments, TRANSLATE_AUDIO_TO, t2t_method):
    if t2t_method == "VB" and TRANSLATE_AUDIO_TO == "vi":
      print("vb_translator::", len(segments), "segments")
      source_text = "\n".join([ segment['text'] for segment in segments])
      translated_text = vb_translate(source_text.strip())
      print("vb_translator translated_text::", len(translated_text), "segments")
      for index, segment in enumerate(segments):
        segments[index]['text'] = translated_text[index]
    elif t2t_method == "T5" and TRANSLATE_AUDIO_TO == "vi":
      ## T5 translator - instantiate the pre-trained English-to-Vietnamese Transformer model
      t5_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
      t5_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
      t5_model = PeftModel.from_pretrained(t5_model, LORA_WEIGHT).cuda()
      t5_model.eval()
      for line in tqdm(range(len(segments))):
        text = segments[line]['text']
        print("t5_translator::")
        translated_line = t5_translator(text.strip(), t5_tokenizer, t5_model)
        print("translate_text_in::", TRANSLATE_AUDIO_TO, t2t_method,f'{text}\n{translated_line}')
        segments[line]['text'] = translated_line
        gc.collect(); torch.cuda.empty_cache(); del t5_model
    else:
      ## Google translator
      google_translator = GoogleTranslator(source='auto', target=TRANSLATE_AUDIO_TO)
      for line in tqdm(range(len(segments))):
        text = segments[line]['text']
        print("gg_translator::")
        translated_line = google_translator.translate(text.strip())
        print("translate_text_in::", TRANSLATE_AUDIO_TO, t2t_method,f'{text}\n{translated_line}')
        segments[line]['text'] = translated_line
    return segments
