from dotenv import load_dotenv
import os
# import shutil
# import json
import random
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import requests
from langdetect import detect
# from vietTTS.utils import concise_srt
# from utils.utils import srt_to_segments, segments_to_srt
from utils.language_configuration import LANGUAGES
from langchain_openai import ChatOpenAI
# from langchain import ConversationChain, LLMChain, PromptTemplate
# from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

load_dotenv()

fault_words = [
  "im_start",
  "im_end"
  ]

class LLM():
  def __init__(self, systemPrompt = "") -> None:
    self.llm_chain = []
    self.systemPrompt = systemPrompt if systemPrompt != "" else "This GPT functions as a translation tool that processes text from {source_language}, translating it into {target_language}. The output is a plain text content with a full translation in {target_language}. It accepts input in the form of {source_language} text, ensuring the texts are accurately digitized and represent the original manuscripts. The translation engine interprets and translates words into modern {target_language}, incorporating linguistic analysis to handle idiomatic expressions and cultural nuances. Response only translated text."
    self.prompt = ChatPromptTemplate(
          messages=[
              SystemMessagePromptTemplate.from_template(self.systemPrompt),
              # The `variable_name` here is what must align with memory
              # MessagesPlaceholder(variable_name="history"),
              HumanMessagePromptTemplate.from_template("{input}"),
          ]
      )
    
  def initLLM(self, endpoints="", model="", api_key="", temp=0.3, k=30):
    # self.memory = ConversationBufferWindowMemory(memory_key="history", return_messages=True, k=k)
    endpoints = endpoints.split(',')
    endpoints = endpoints if len(endpoints)>0 else ["https://openrouter.ai/api/v1"]
    model = model if model != "" else "openai/gpt-4o"
    api_key = api_key if api_key != "" else "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"
    for endpoint in endpoints:
      try:
        if endpoint:
          response = requests.get(f"{endpoint}/models")
          print("llm_status::", endpoint, response,temp, k)
          models = [item['id'] for item in response.json()["data"]]
          if model in models:
            llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                openai_api_base=endpoint,
                max_tokens=4096,
                temperature=temp,
                # max_retries=2,
                # model_kwargs={
                #   "stop":["<|im_end|>"],
                #   "frequency_penalty": 1.1
                # },
                top_p= 0.95,
                frequency_penalty=1.3,
                stop=["<|im_end|>"],
                
            )
            llm_chain = self.prompt | llm
            self.llm_chain.append(llm_chain)
      except Exception as e:
        print('initLLM error:',  endpoint, e)
    return True if len(self.llm_chain) > 0 else False
        
  def process(self, text, source_lang="en", target_lang="vn"):
    max_attempts = 3
    attempts = 0
    source_language = next((key for key, value in LANGUAGES.items() if value == source_lang), None)
    target_language = next((key for key, value in LANGUAGES.items() if value == target_lang), None)
    print('language::', source_language, target_language)

    while attempts < max_attempts:
      try:
        result = random.choice(self.llm_chain).invoke({
                  "input": text,
                  "source_language": source_language,
                  "target_language": target_language,
              })
        if result.content and not any(word in result.content.strip().lower() for word in fault_words) and target_lang in detect(result.content):
            return result.content
      except Exception as e:
        print("error::", e)
        result = {"content": ""}
      print(f"re-run {attempts}:")
      attempts += 1
    return text

  def translate(self, segments, source_lang="en", target_lang="vi"):
      print("start llm_translate::")
      # N_JOBS = os.cpu_count()
      # print("Start LLM Translate:: concurrency =", N_JOBS)
      with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(1)):
        t2t_results = Parallel(verbose=100)(delayed(self.process)(segments[line]['text'], source_lang, target_lang) for (line) in tqdm(range(len(segments))))
      for index in tqdm(range(len(segments))):
        segments[index]['text'] = t2t_results[index]
      return segments  
    
  def predict(self, text, source_lang="en", target_lang="vi"):
      print("start llm_translate::")
      result = self.process(text, source_lang, target_lang)
      return result  
    
# if __name__ == '__main__':
  
  # systemPrompt="""Sửa lỗi chính tả từ bản gốc sang bảng mới"""
  # llm = LLM(systemPrompt=systemPrompt)
#   llm.initLLM(
#     endpoints="https://openrouter.ai/api/v1", ## http://172.27.188.32:8081/v1
#     model="openai/gpt-4o", ## "trast-ai/trust-translator-llama3-5b4e" "nampdn-ai/vietmistral-bible-translation"
#     api_key="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",
#     temp=0.3,
#     k=10
#   )


  # llm = LLM()
  # llm.initLLM(
  #   endpoints="http://172.27.188.41:8081/v1", ## http://172.27.188.31:8081/v1
  #   model="trust-translator", ## "trast-ai/trust-translator-llama3-5b4e" "nampdn-ai/vietmistral-bible-translation"
  #   api_key="EMPTY",
  #   temp=0.3,
  #   k=10
  # )
  # text = "English is a West Germanic language in the Indo-European language family, whose speakers, called Anglophones, originated in early medieval England on the island"
  # result = llm.predict(text, "vi", "vi")
  # print(result)
    
  # ## Translate segments
  # input_file = '/home/vgm/Desktop/en.srt'
  # segments = srt_to_segments(input_file)
  # # segments = concise_srt(segments)
  # # segments_to_srt(segments, '/home/vgm/Desktop/en.srt')
  # print(segments, len(segments))
  # segments = llm.translate(segments=segments, source_lang="en", target_lang="vi")
  # print("results::",  segments, len(segments))
  # segments_to_srt(segments, '/home/vgm/Desktop/vi.srt')


  ## Translate texts
  # input_texts = [
  # "Reason and science are gifts from god that help us discern these patterns, and for this reason evangelicals write, value rational and scientific research into the pentateuch"
  # ]
  # for text in input_texts:
  #   result = llm.process(text)
  #   print("result::", result)
    
  

