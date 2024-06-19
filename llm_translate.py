from dotenv import load_dotenv
import os
import shutil
import json
import random
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import requests
from langdetect import detect
# from vietTTS.utils import concise_srt
from utils.utils import srt_to_segments, segments_to_srt

load_dotenv()

from langchain_community.chat_models import ChatOpenAI
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

class LLM():
  def __init__(self) -> None:
    self.llm_chain = []
    self.prompt = ChatPromptTemplate(
          messages=[
              SystemMessagePromptTemplate.from_template(
                  "Bạn là AI có khả năng dịch thuật nội dung từ tiếng Anh một cách chính xác và rất dễ hiểu cho người Việt Nam. Hãy cẩn thận dịch và chọn từ ngữ cho phù hợp."
              ),
              # The `variable_name` here is what must align with memory
              # MessagesPlaceholder(variable_name="history"),
              HumanMessagePromptTemplate.from_template("{input}"),
          ]
      )
    
  def initLLM(self, endpoints, model, temp=0.3, k=30):
    # self.memory = ConversationBufferWindowMemory(memory_key="history", return_messages=True, k=k)
    endpoints = endpoints.split(',')
    for endpoint in endpoints:
      try:
        if endpoint:
          response = requests.get(f"{endpoint}/models")
          print("llm_status::", endpoint, response,temp, k)
          models = [item['id'] for item in response.json()["data"]]
          if model in models:
            llm = ChatOpenAI(
                model=model,
                openai_api_key="EMPTY",
                openai_api_base=endpoint,
                max_tokens=4096,
                temperature=temp,
                # max_retries=2,
                model_kwargs={
                  "stop":["<|im_end|>"],
                  "top_p": 0.95,
                  # "top_k": 30
                },
            )
            llm_chain = LLMChain(llm=llm, 
                                prompt=self.prompt,
                                # memory=self.memory, 
                                verbose=True)
            self.llm_chain.append(llm_chain)
      except Exception as e:
        print('initLLM error:',  endpoint, e)
    return True if len(self.llm_chain) > 0 else False
        
  def process(self, text):
    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
      try:
        result = random.choice(self.llm_chain).predict(input=text)
        if result and "im_start" not in result and "im_end" not in result and (abs(len(text) - len(result))) <= 100 and detect(result) == 'vi':
            return result
      except Exception as e:
        result = ""
      print(f"re-run {attempts}: {len(text)}/{len(result)}\nen: {text}\nvi: {result}")
      attempts += 1
    return text

  def translate(self, segments):
      print("start llm_translate::")
      # N_JOBS = os.cpu_count()
      # print("Start LLM Translate:: concurrency =", N_JOBS)
      with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(10)):
        t2t_results = Parallel(verbose=100)(delayed(self.process)(segments[line]['text']) for (line) in tqdm(range(len(segments))))
      for index in tqdm(range(len(segments))):
        segments[index]['text'] = t2t_results[index]
      return segments  

# if __name__ == '__main__':
#   llm = LLM()
#   llm.initLLM( 
#     endpoints="https://infer-2.vgm.chat/v1,https://infer-3.vgm.chat/v1", 
#     model="bible-translator-llama3-5b4e", ## "nampdn-ai/vietmistral-chatvgm-3072" "nampdn-ai/vietmistral-bible-translation"
#     temp=0.3,
#     k=10
#   )
  
#   ## Translate segments
#   input_file = '/home/vgm/Desktop/en.srt'
#   segments = srt_to_segments(input_file)
#   # segments = concise_srt(segments)
#   # segments_to_srt(segments, '/home/vgm/Desktop/en.srt')
#   print(segments, len(segments))
#   segments = llm.translate(segments)
#   print("results::",  segments, len(segments))
#   segments_to_srt(segments, '/home/vgm/Desktop/vi.srt')


#   ## Translate texts
#   # input_texts = [
#   # "Reason and science are gifts from god that help us discern these patterns, and for this reason evangelicals write, value rational and scientific research into the pentateuch"
#   # ]
#   # for text in input_texts:
#   #   result = llm.process(text)
#   #   print("result::", result)
    
  


