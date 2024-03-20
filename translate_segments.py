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
from gradio_client import Client


LANGUAGES_ABBR = {
    'Automatic detection': None,
    'Arabic': 'ar',
    'Cantonese': 'yue',
    'Chinese': 'zh',
    'Czech': 'cs',
    'Danish': 'da',
    'Dutch': 'nl',
    'English': 'en',
    'Finnish': 'fi',
    'French': 'fr',
    'German': 'de',
    'Greek': 'el',
    'Hebrew': 'he',
    'Hungarian': 'hu',
    'Indonesian': 'id',
    'Italian': 'it',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Persian': 'fa',
    'Polish': 'pl',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Spanish': 'es',
    'Tagalog': 'tl',
    'Thai': 'th',
    'Turkish': 'tr',
    'Ukrainian': 'uk',
    'Urdu': 'ur',
    'Vietnamese': 'vi',
    'Hindi': 'hi',
}

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
def translate_text(segments, SOURCE_LANGUAGE="", TRANSLATE_AUDIO_TO="", t2t_method="", llm_endpoint="", llm_model=""):
    print("start translate_text::", segments)
    if t2t_method == "VB" and TRANSLATE_AUDIO_TO == "vi":
      print("vb_translator::", len(segments), "segments")
      source_text = "\n".join([ segment['text'] for segment in segments])
      translated_text = vb_translate(source_text.strip())
      print("vb_translator translated_text::", len(translated_text), "segments")
      for index, segment in enumerate(segments):
        segments[index]['text'] = post_process_text(translated_text[index])
    elif t2t_method == "T5" and TRANSLATE_AUDIO_TO == "vi":
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
    elif t2t_method == "M4T":
      client = Client("https://facebook-seamless-m4t.hf.space/--replicas/0jsdd/")
      for line in tqdm(range(len(segments))):
        text = segments[line]['text']
        try:
          text = client.predict(
              'T2TT (Text to Text translation)',	# str (Option from: [('S2ST (Speech to Speech translation)', 'S2ST (Speech to Speech translation)'), ('S2TT (Speech to Text translation)', 'S2TT (Speech to Text translation)'), ('T2ST (Text to Speech translation)', 'T2ST (Text to Speech translation)'), ('T2TT (Text to Text translation)', 'T2TT (Text to Text translation)'), ('ASR (Automatic Speech Recognition)', 'ASR (Automatic Speech Recognition)')]) in 'Task' Dropdown component	
              "file",	# str  in 'Audio source' Radio component
              "",	# str (filepath on your computer (or URL) of file) in 'Input speech' Audio component
              "",	# str (filepath on your computer (or URL) of file) in 'Input speech' Audio component
              text,	# str  in 'Input text' Textbox component
              SOURCE_LANGUAGE,	# str (Option from: [('Afrikaans', 'Afrikaans'), ('Amharic', 'Amharic'), ('Armenian', 'Armenian'), ('Assamese', 'Assamese'), ('Basque', 'Basque'), ('Belarusian', 'Belarusian'), ('Bengali', 'Bengali'), ('Bosnian', 'Bosnian'), ('Bulgarian', 'Bulgarian'), ('Burmese', 'Burmese'), ('Cantonese', 'Cantonese'), ('Catalan', 'Catalan'), ('Cebuano', 'Cebuano'), ('Central Kurdish', 'Central Kurdish'), ('Croatian', 'Croatian'), ('Czech', 'Czech'), ('Danish', 'Danish'), ('Dutch', 'Dutch'), ('Egyptian Arabic', 'Egyptian Arabic'), ('English', 'English'), ('Estonian', 'Estonian'), ('Finnish', 'Finnish'), ('French', 'French'), ('Galician', 'Galician'), ('Ganda', 'Ganda'), ('Georgian', 'Georgian'), ('German', 'German'), ('Greek', 'Greek'), ('Gujarati', 'Gujarati'), ('Halh Mongolian', 'Halh Mongolian'), ('Hebrew', 'Hebrew'), ('Hindi', 'Hindi'), ('Hungarian', 'Hungarian'), ('Icelandic', 'Icelandic'), ('Igbo', 'Igbo'), ('Indonesian', 'Indonesian'), ('Irish', 'Irish'), ('Italian', 'Italian'), ('Japanese', 'Japanese'), ('Javanese', 'Javanese'), ('Kannada', 'Kannada'), ('Kazakh', 'Kazakh'), ('Khmer', 'Khmer'), ('Korean', 'Korean'), ('Kyrgyz', 'Kyrgyz'), ('Lao', 'Lao'), ('Lithuanian', 'Lithuanian'), ('Luo', 'Luo'), ('Macedonian', 'Macedonian'), ('Maithili', 'Maithili'), ('Malayalam', 'Malayalam'), ('Maltese', 'Maltese'), ('Mandarin Chinese', 'Mandarin Chinese'), ('Marathi', 'Marathi'), ('Meitei', 'Meitei'), ('Modern Standard Arabic', 'Modern Standard Arabic'), ('Moroccan Arabic', 'Moroccan Arabic'), ('Nepali', 'Nepali'), ('North Azerbaijani', 'North Azerbaijani'), ('Northern Uzbek', 'Northern Uzbek'), ('Norwegian Bokmål', 'Norwegian Bokmål'), ('Norwegian Nynorsk', 'Norwegian Nynorsk'), ('Nyanja', 'Nyanja'), ('Odia', 'Odia'), ('Polish', 'Polish'), ('Portuguese', 'Portuguese'), ('Punjabi', 'Punjabi'), ('Romanian', 'Romanian'), ('Russian', 'Russian'), ('Serbian', 'Serbian'), ('Shona', 'Shona'), ('Sindhi', 'Sindhi'), ('Slovak', 'Slovak'), ('Slovenian', 'Slovenian'), ('Somali', 'Somali'), ('Southern Pashto', 'Southern Pashto'), ('Spanish', 'Spanish'), ('Standard Latvian', 'Standard Latvian'), ('Standard Malay', 'Standard Malay'), ('Swahili', 'Swahili'), ('Swedish', 'Swedish'), ('Tagalog', 'Tagalog'), ('Tajik', 'Tajik'), ('Tamil', 'Tamil'), ('Telugu', 'Telugu'), ('Thai', 'Thai'), ('Turkish', 'Turkish'), ('Ukrainian', 'Ukrainian'), ('Urdu', 'Urdu'), ('Vietnamese', 'Vietnamese'), ('Welsh', 'Welsh'), ('West Central Oromo', 'West Central Oromo'), ('Western Persian', 'Western Persian'), ('Yoruba', 'Yoruba'), ('Zulu', 'Zulu')]) in 'Source language' Dropdown component
              TRANSLATE_AUDIO_TO,	# str (Option from: [('Bengali', 'Bengali'), ('Catalan', 'Catalan'), ('Czech', 'Czech'), ('Danish', 'Danish'), ('Dutch', 'Dutch'), ('English', 'English'), ('Estonian', 'Estonian'), ('Finnish', 'Finnish'), ('French', 'French'), ('German', 'German'), ('Hindi', 'Hindi'), ('Indonesian', 'Indonesian'), ('Italian', 'Italian'), ('Japanese', 'Japanese'), ('Korean', 'Korean'), ('Maltese', 'Maltese'), ('Mandarin Chinese', 'Mandarin Chinese'), ('Modern Standard Arabic', 'Modern Standard Arabic'), ('Northern Uzbek', 'Northern Uzbek'), ('Polish', 'Polish'), ('Portuguese', 'Portuguese'), ('Romanian', 'Romanian'), ('Russian', 'Russian'), ('Slovak', 'Slovak'), ('Spanish', 'Spanish'), ('Swahili', 'Swahili'), ('Swedish', 'Swedish'), ('Tagalog', 'Tagalog'), ('Telugu', 'Telugu'), ('Thai', 'Thai'), ('Turkish', 'Turkish'), ('Ukrainian', 'Ukrainian'), ('Urdu', 'Urdu'), ('Vietnamese', 'Vietnamese'), ('Welsh', 'Welsh'), ('Western Persian', 'Western Persian')]) in 'Target language' Dropdown component
              api_name="/run"
          )
          if text and text[1]:
            print("m4t_translator::", SOURCE_LANGUAGE, TRANSLATE_AUDIO_TO, t2t_method, text[1])
            segments[line]['text'] = post_process_text(text[1])
        except:
          print('An exception occurred')
    else:
      pass
    ## Last option to check if any non-translated sentences left then using Google translator
    google_translator = GoogleTranslator(source='auto', target=LANGUAGES_ABBR[TRANSLATE_AUDIO_TO])
    for line in tqdm(range(len(segments))):
      # print("gg_translator::")
      try:
        text = segments[line]['text']
        # if text and detect(text.strip()) != 'vi':
        translated_line = google_translator.translate(text.strip())
        # print("translate_text_in::", TRANSLATE_AUDIO_TO, t2t_method,f'{text}\n{translated_line}')
        segments[line]['text'] = post_process_text(translated_line)
      except Exception as e:
        pass
    return segments
