from dotenv import load_dotenv
import requests
import time
import os
import re
# import shutil
# import json
import random
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
from langdetect import detect
# from vietTTS.utils import concise_srt
# from utils.utils import srt_to_segments, segments_to_srt
from utils.language_configuration import LANGUAGES
from langchain_openai import ChatOpenAI
import concurrent.futures
from requests.exceptions import RequestException
import threading
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()

fault_words = [
  "im_start",
  "im_end",
  "<skip_think>"
  "<think>",
  "</think>"
  ]

default_endpoints = [
    #"http://172.27.188.32:8082/v1",
    # "http://192.168.2.12:8081/v1",
    # "http://192.168.2.13:8081/v1",
    # "http://192.168.2.14:8081/v1",
    # "http://192.168.2.14:8082/v1",
]

def cleanup_text(text):
    if '</think>' in text:
      text = text.split('</think>')[1]
    if '<skip_think>' in text:
      text = text.split('<skip_think>')[1]
    return text
  
def is_valid_response(source_text, target_text):
    source_text = str(source_text).replace("-", " ").strip()
    target_text = str(target_text).replace("-", " ").strip()
    source_len = len(source_text)
    target_len = len(target_text)

    print(f"----- length count ----- source: {source_len} - target {target_len} | {target_len/source_len}")

    if target_text == '':
        print("-----invalid response ---- : empty target_text")
        return False
    elif any(word in target_text.lower() for word in fault_words):
        print("-----invalid response ---- : fault_words")
        return False
    elif target_len/source_len > 2 or source_len/target_len > 2:
        print("-----invalid response ---- : length not match::" ,source_len, target_len)
        return False

    return True
    
class LLM():
  def __init__(self, systemPrompt = "") -> None:
    self.llm_chain = {}
    self.endpoints = default_endpoints
    self.interval = 5
    self.timeout = 2
    self.model = ""
    self.api_key = ""
    self.temp = 0.3
    self.k = 10
    self.available_endpoints = set(default_endpoints) 
    self.systemPrompt = systemPrompt if systemPrompt != "" else "This GPT functions as a translation tool that processes text from {source_language}, translating it into {target_language}. The output is a plain text content with a full translation in {target_language}. It accepts input in the form of {source_language} text, ensuring the texts are accurately digitized and represent the original manuscripts. The translation engine interprets and translates words into modern {target_language}, incorporating linguistic analysis to handle idiomatic expressions and cultural nuances. Response only translated text."
    self.prompt = ChatPromptTemplate(
          messages=[
              SystemMessagePromptTemplate.from_template(self.systemPrompt),
              # The `variable_name` here is what must align with memory
              MessagesPlaceholder(variable_name="history"),
              HumanMessagePromptTemplate.from_template("{input}"),
          ]
      )

  def check_endpoint(self, endpoint: str):
      url = f"{endpoint}/models"
      try:
          response = requests.get(url, timeout=self.timeout)
          
          if response.status_code == 200:
              self.available_endpoints.add(endpoint)  # Add to available endpoints
              self.llm_chain[endpoint] = ChatOpenAI(
                          model=self.model,
                          openai_api_key=self.api_key,
                          openai_api_base=endpoint,
                          max_tokens=2048,
                          temperature=self.temp,
                          top_p= 0.95,
                          frequency_penalty=1.3,
                          stop=["<|im_end|>"],
                      )
          else:
              self.available_endpoints.discard(endpoint)  # Remove from available endpoints
              if endpoint in self.llm_chain:
                del self.llm_chain[endpoint]
      except RequestException as e:
          self.available_endpoints.discard(endpoint)  # Remove from available endpoints
          if endpoint in self.llm_chain:
            del self.llm_chain[endpoint]
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
                  print("Available llm endpoints::\n", result)
          self.interval = 60
          time.sleep(self.interval)
                
  def initLLM(self, endpoints="", model="", api_key="", temp=0.5, k=5):
    print("Initializing LLM::")
    endpoints = endpoints.split(',')
    self.endpoints = list(set(default_endpoints + endpoints))
    self.endpoints = self.endpoints if len(self.endpoints) > 0 else ["https://openrouter.ai/api/v1"]
    self.temp = temp
    self.k = k
    self.model = model if model != "" else "openai/gpt-4o"
    self.api_key = api_key if api_key != "" else os.getenv("OR_API_KEY", "")
    for endpoint in self.endpoints:
      self.check_endpoint(endpoint)
    time.sleep(self.interval)
    self._monitor_thread = threading.Thread(target=self.monitor, daemon=True)
    self._monitor_thread.start()
    return True
        
  def process(self, text, source_lang="en", target_lang="vn"):
    max_attempts = 5
    attempts = 0
    source_language = next((key for key, value in LANGUAGES.items() if value == source_lang), None)
    target_language = next((key for key, value in LANGUAGES.items() if value == target_lang), None)
    llms = [v for k, v in self.llm_chain.items()]

    while attempts < max_attempts:
      try:
        llm = random.choice(llms)
        print('translate inferencing::', source_language, target_language)
        chain = self.prompt | llm
        llm_chain = RunnableWithMessageHistory(
            chain,
            lambda session_id: ChatMessageHistory(),  # Factory for creating history storage
            input_messages_key="input",               # Key for input messages
            history_messages_key="history",      # Key for history in the chain
            window_size=self.k                            # This is equivalent to the 'k' parameter - keep last 2 exchanges
        )
        result = llm_chain.invoke({
                  "input": text,
                  "source_language": source_language,
                  "target_language": target_language,
              }, config={"configurable": {"session_id": "default_session"}
        })
        if result.content and is_valid_response(text, cleanup_text(result.content)) and target_lang in detect(result.content):
            return cleanup_text(result.content)
      except Exception as e:
        print("error::", e)
        result = {"content": ""}
      print(f"re-run {attempts}:")
      attempts += 1
    return text

  def translate(self, segments, source_lang="en", target_lang="vi"):
      print("start llm_translate::")
      N_JOBS = len(self.available_endpoints) * 7 if len(self.available_endpoints) else 20
      print("Start LLM Translate:: concurrency =", N_JOBS)
      with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(N_JOBS)):
        t2t_results = Parallel(verbose=100)(delayed(self.process)(segments[line]['text'], source_lang, target_lang) for (line) in tqdm(range(len(segments))))
      for index in tqdm(range(len(segments))):
        segments[index]['text'] = t2t_results[index]
      return segments
    
  def predict(self, text, source_lang="en", target_lang="vi"):
      print("start llm_translate::")
      result = self.process(text, source_lang, target_lang)
      return result  
    
# if __name__ == '__main__':
  
#   # systemPrompt="""Sửa lỗi chính tả từ bản gốc sang bảng mới"""
#   # llm = LLM(systemPrompt=systemPrompt)
#   # llm.initLLM(
#   #   endpoints="https://openrouter.ai/api/v1", ## http://172.27.188.32:8081/v1
#   #   model="openai/gpt-4o", ## "trast-ai/trust-translator-llama3-5b4e" "nampdn-ai/vietmistral-bible-translation"
#   #   api_key=os.getenv("OR_API_KEY", ""),
#   #   temp=0.3,
#   #   k=10
#   # )
  
#   # systemPrompt="""Think and translate English accurately into clear, natural, appropriate Vietnamese."""
#   # llm = LLM(systemPrompt=systemPrompt)
#   llm = LLM()
#   llm.initLLM(
#     endpoints="http://172.27.188.32:8082/v1", ## http://172.27.188.31:8081/v1
#     model="trast-ai/trust-translator-0525", ## "trast-ai/trust-translator-llama3-5b4e" "nampdn-ai/vietmistral-bible-translation"
#     api_key="EMPTY",
#     temp=0.1,
#     k=10
#   )
#   text = "Reason and science are gifts from god that help us discern these patterns, and for this reason evangelicals write, value rational and scientific research into the pentateuch"
#   result = llm.predict(text, "en", "vi")
#   print("source::", text)
#   print("target::", result)
    
#   # ## Translate segments
#   # input_file = '/home/vgm/Desktop/en.srt'
#   # segments = srt_to_segments(input_file)
#   # # segments = concise_srt(segments)
#   # # segments_to_srt(segments, '/home/vgm/Desktop/en.srt')
#   # # print(segments, len(segments))
#   # segments = llm.translate(segments=segments, source_lang="en", target_lang="vi")
#   # # print("results::",  segments, len(segments))
#   # segments_to_srt(segments, '/home/vgm/Desktop/vi.srt')


#   # Translate texts
#   # input_texts = [
#   # "Reason and science are gifts from god that help us discern these patterns, and for this reason evangelicals write, value rational and scientific research into the pentateuch"
#   # ]
#   # for text in input_texts:
#   #   result = llm.process(text)
#   #   print("result::", result)
    
  

