from dotenv import load_dotenv
import os
import shutil
import json
import random
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import requests
load_dotenv()

from langchain.chat_models import ChatOpenAI
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
                  "Bạn là AI có khả năng dịch thuật nội dung Kinh Thánh từ tiếng Anh một cách chính xác và rất dễ hiểu cho người Việt Nam. Hãy cẩn thận dịch và chọn từ ngữ cho phù hợp."
              ),
              # The `variable_name` here is what must align with memory
              MessagesPlaceholder(variable_name="history"),
              HumanMessagePromptTemplate.from_template("{input}"),
          ]
      )
    self.memory = ConversationBufferWindowMemory(memory_key="history", return_messages=True, k=10)
    
  def initLLM(self, endpoints, model):
    endpoints = endpoints.split(',')
    for endpoint in endpoints:
      response = requests.get(f"{endpoint}/models")
      models = [item['id'] for item in response.json()["data"]]
      if model in models:
        llm = ChatOpenAI(
            model=model,
            openai_api_key="EMPTY",
            openai_api_base=endpoint,
            max_tokens=2048,
            temperature=0.5,
            model_kwargs={"stop":["<|im_end|>"]},
            # top_p=0.5
        )
        llm_chain = LLMChain(llm=llm, prompt=self.prompt, memory=self.memory, verbose=True)
        self.llm_chain.append(llm_chain)

  def translate(self, segments):
      print("start llm_translate::")
      # N_JOBS = os.cpu_count()
      # print("Start LLM Translate:: concurrency =", N_JOBS)
      with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(10)):
        t2t_results = Parallel(verbose=100)(delayed(random.choice(self.llm_chain).predict)(input=segments[line]['text']) for (line) in tqdm(range(len(segments))))
      for index in tqdm(range(len(segments))):
        segments[index]['text'] = t2t_results[index]
      return segments  

if __name__ == '__main__':
  input_texts = [
'I want to read actually from, three passages of scripture, i\'m going to read just briefly here from john chapter 3, and then from john chapter 7',
'for god so loved the world that he gave his one and only son, that whoever believes in him shall not perish but have eternal life',
'for god did not send his son into the world to condemn the world, but to save the world through him',
'whoever believes in him is not condemned, but whoever does not believe stands condemned already, because he has not believed in the name of god\'s one and only son',
'this is the verdict light has come into the world, but men loved darkness instead of light because their deeds were evil, everyone who does evil hates the light, and will not come into the light for fear that his deeds will be exposed, but whoever lives by the truth comes into the light so that it may be seen plainly, that what he has done has been done through god',
'And then in John chapter seven',
'In verse 37, on the last and greatest day of the feast, jesus stood and said in a loud voice, "if anyone is thirsty, let him come to me and drink, whoever believes in me, as the scripture has said, streams of living water, will flow from within him, by this he meant the spirit whom those who believed in him, were later to receive, up to that time, the spirit had not',
'been given since Jesus had not yet been glorified.',
'On hearing his words, some of the people said, surely this man is the prophet.',
'others said, "he is the christ," still others asked, "how can the christ come from galilee?',
'does not the scripture say that the christ will come from david\'s family and from bethlehem, the town where david lived, thus the people, were divided because of jesus, some wanted to seize him, but no one laid a hand on him, and then finally from acts and from chapter 4',
'And from verse 8, then peter, filled with the holy spirit, said to them,"rulers and elders of the people, if we\'re being called to account today for an act of kindness, shown to a cripple, and are asked how he was healed, then know this, you and all the people of israel, it is by the name of jesus christ of nazareth, whom you crucified, but whom god raised from the dead',
  ]
  
  # input_file = '/home/vgm/Desktop/test.txt'
  # output_file = f'{input_file}.txt'
  # input_texts = [line.strip() for line in open(input_file, 'r')]
  input_texts = [ {"text": text} for text in input_texts]
  llm = LLM()
  llm.initLLM("https://infer-2.vgm.chat/v1,https://infer-3.vgm.chat/v1", "nampdn-ai/vietmistral-chatvgm-3072")
  results = llm.translate(input_texts)
  print("input_texts::", len(input_texts))
  print("results::", len(results), results)
  
  # with open(output_file, 'a') as file:
  #   for item in results:
  #     file.write(item["text"] + '\n')

