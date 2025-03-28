from dotenv import load_dotenv
import requests
import time
import random
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
from langdetect import detect
from utils.language_configuration import LANGUAGES
import concurrent.futures
from requests.exceptions import RequestException
import threading
import json
import re
import os
# from vietTTS.utils import concise_srt
# from utils.utils import srt_to_segments, segments_to_srt
load_dotenv()

fault_words = [
  "im_start",
  "im_end"
  ]

default_endpoints = [
    # "http://192.168.2.12:8081/v1",
    # "http://192.168.2.13:8081/v1",
    # "http://192.168.2.14:8081/v1",
    # "http://192.168.2.14:8082/v1",
]
class LLM():
  def __init__(self, systemPrompt = "") -> None:
    self.llm_chain = {}
    self.endpoints = default_endpoints
    self.interval = 60
    self.timeout = 2
    self.model = ""
    self.api_key = ""
    self.temp = 0.3
    self.k = 60
    self.available_endpoints = set(default_endpoints) 
    self.systemPrompt = systemPrompt if systemPrompt != "" else "This GPT functions as a translation tool that processes text from {source_language}, translating it into {target_language}. The output is a plain text content with a full translation in {target_language}. It accepts input in the form of {source_language} text, ensuring the texts are accurately digitized and represent the original manuscripts. The translation engine interprets and translates words into modern {target_language}, incorporating linguistic analysis to handle idiomatic expressions and cultural nuances. Response only translated text."

  def check_endpoint(self, endpoint: str):
      url = f"{endpoint}/models"
      try:
          headers = {
              "Authorization": f"Bearer {self.api_key}",
          } if "web.chattrust.ai" in endpoint else {}
          response = requests.get(url, headers=headers, timeout=self.timeout)
          print(f"check_endpoint -- {endpoint} -- status::", response.status_code)
          
          if response.status_code == 200:
              self.available_endpoints.add(endpoint)  # Add to available endpoints
          else:
              self.available_endpoints.discard(endpoint)  # Remove from available endpoints
      except RequestException as e:
          self.available_endpoints.discard(endpoint)  # Remove from available endpoints
      return url
      
  def monitor(self):
      while True:
          with concurrent.futures.ThreadPoolExecutor() as executor:
              # Check all endpoints concurrently
              future_to_endpoint = {
                  executor.submit(self.check_endpoint, endpoint): endpoint 
                  for endpoint in self.endpoints
              }
              
              for future in concurrent.futures.as_completed(future_to_endpoint):
                  result = future.result()
                  print("Available llm correct endpoints::\n", result)
          time.sleep(self.interval)
                
  def initLLM(self, endpoints="", model="", api_key="", temp=0.3, k=30):
    print("Initializing LLM::")
    endpoints = endpoints.split(',')
    self.endpoints = list(set(default_endpoints + endpoints))
    self.endpoints = self.endpoints if len(self.endpoints)>0 else ["https://openrouter.ai/api/v1"]
    self.temp = temp
    self.k = k
    self.model = model if model != "" else "deepseek/deepseek-chat-v3-0324"
    self.api_key = api_key if api_key != "" else os.getenv("OR_API_KEY", "")
    for endpoint in self.endpoints:
      self.check_endpoint(endpoint)
    self._monitor_thread = threading.Thread(target=self.monitor, daemon=True)
    self._monitor_thread.start()
    return True
              
  def process(self, source_text, target_text, source_lang="en", target_lang="vn"):
    max_attempts = 10
    attempts = 0
    source_language = next((key for key, value in LANGUAGES.items() if value == source_lang), None)
    target_language = next((key for key, value in LANGUAGES.items() if value == target_lang), None)
    user_prompt = f"""
SOURCE PARAGRAPH:
```
{source_text}
```
CURRENT TRANSLATED PARAGRAPH (to be corrected):
```
{target_text}
```
Please respond with the corrected Vietnamese translation in plain text, without extra explanation.
"""
    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json; charset=utf-8"
    }
    data = {
        "model": self.model,
        "messages": [
            {"role": "system", "content": self.systemPrompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    while attempts < max_attempts:
      try:
        endpoint = random.choice(list(self.available_endpoints))
        print(f"\n--- selected endpoint:: {endpoint} ---")
        print("\n--- chat_with_model REQUEST ---")
        print(json.dumps(data, ensure_ascii=False))
        print("------------------------------")
        
        # Add a timeout to avoid hanging forever
        response = requests.post(
            url=f"{endpoint}/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=3600
        )

        # In case of non-200 status, it might raise or we can do extra checks
        response.raise_for_status()

        # Attempt to parse JSON
        res = response.json()
        print("\n--- chat_with_model RESPONSE ---")
        print(json.dumps(res, ensure_ascii=False))
        print("--------------------------------")
        result = res['choices'][0]["message"]["content"]
        result = re.sub(r"[\`]+", " ", result)
        if "</think>" in result:
            result = result.split("</think>")[-1].strip()
        if result and not any(word in result.strip().lower() for word in fault_words) and target_lang in detect(result):
            return result.strip()
          
      except Exception as e:
        print("error::", e)
        result = {"content": ""}
        
      print(f"re-run {attempts}:")
      attempts += 1
    return target_text

  def translate(self, source_segments, target_segments, source_lang="en", target_lang="vi"):
      print("start llm_translate::")
      N_JOBS = len(self.available_endpoints) * 5 if len(self.available_endpoints) else 5
      print("Start LLM Translate:: concurrency =", N_JOBS)
      with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(N_JOBS)):
        t2t_results = Parallel(verbose=100)(delayed(self.process)(source_segments[line]['text'], target_segments[line]['text'], source_lang, target_lang) for (line) in tqdm(range(len(target_segments))))
      for index in tqdm(range(len(target_segments))):
        target_segments[index]['text'] = t2t_results[index]
      return target_segments
    
  def predict(self, source_text, target_text, source_lang="en", target_lang="vi"):
      print("start llm_translate::")
      result = self.process(source_text, target_text, source_lang, target_lang)
      return result  
    
# if __name__ == '__main__':
#   glossary_table="""God: Thiên Chúa
#   Jehovah God, God Jehovah, Jehovah, LORD God: Thiên Chúa Hằng Hữu
#   Christ: Đấng Cứu Thế, Chúa Cứu Thế
#   Jesus: Chúa Giê-xu
#   Jesus Christ: Chúa Cứu Thế Giê-xu
#   Holy Spirit: Đức Thánh Linh, Chúa Thánh Linh, Thánh Linh
#   Do not use any "Ki-tô" word in the corrected version but use "Cơ Đốc" instead."""  
#   systemPrompt="""
#   Given a source paragraph in English and its Vietnamese translation, please correct any errors in the translation to ensure it is accurate, faithful to the original meaning, clear, and natural-sounding for Vietnamese readers.
#   Only revise parts that are unnatural or difficult to understand. Prioritize accuracy and fidelity to the English text while making the translation easy to comprehend.
#   Use the glossary table to choose the most appropriate and consistent terms.""" + "\n\nGLOSSARY / DICTIONARY GUIDANCE:\n" + glossary_table
  
#   llm = LLM(systemPrompt=systemPrompt)
#   llm.initLLM(
#     endpoints="https://web.chattrust.ai/api", ## https://openrouter.ai/api/v1
#     model="deepseek-r1:70b", ## deepseek/deepseek-chat-v3-0324
#     api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImNjOGM4NzdkLWJkNWMtNGE5MS05Y2EyLTUzNjkwNjE3M2VmYiJ9.feqjSWieK7eKtdl0Qkw77BO8HswLHeXwA5l3xK8aw-M",
#     temp=0.3,
#     k=200
#   )

#   ## Translate segments
#   source_file = '/home/vgm/Downloads/logos-vi/logos-en.srt'
#   target_file = '/home/vgm/Downloads/logos-vi/logos-vi.srt'
#   source_segments = srt_to_segments(source_file)
#   target_segments = srt_to_segments(target_file)
#   segments = llm.translate(source_segments, target_segments, source_lang="en", target_lang="vi")
#   print("results::",  segments, len(segments))
