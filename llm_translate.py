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
from utils import srt_to_segments, segments_to_srt

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    # MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

class LLM():
  def __init__(self) -> None:
    self.llm_chain = []
    self.prompt = ChatPromptTemplate(
          messages=[
              SystemMessagePromptTemplate.from_template(
                  "Bạn là AI có khả năng dịch thuật nội dung Kinh Thánh từ tiếng Anh một cách chính xác và rất dễ hiểu cho người Việt Nam. Hãy cẩn thận dịch và chọn từ ngữ cho phù hợp, chỉ dịch và không trả lời câu hỏi."
              ),
              # The `variable_name` here is what must align with memory
              # MessagesPlaceholder(variable_name="history"),
              HumanMessagePromptTemplate.from_template("{input}"),
          ]
      )
    # self.memory = ConversationBufferWindowMemory(memory_key="history", return_messages=True, k=10)
    
  def initLLM(self, endpoints, model):
    try:
      endpoints = endpoints.split(',')
      for endpoint in endpoints:
        response = requests.get(f"{endpoint}/models")
        models = [item['id'] for item in response.json()["data"]]
        if model in models:
          llm = ChatOpenAI(
              model=model,
              openai_api_key="EMPTY",
              openai_api_base=endpoint,
              max_tokens=8192,
              temperature=0.5,
              model_kwargs={"stop":["<|im_end|>"]},
              # top_p=0.5
          )
          llm_chain = LLMChain(llm=llm, prompt=self.prompt, verbose=True)
          self.llm_chain.append(llm_chain)
      return True
    except Exception as e:
      print('initLLM error:', e)
      return False
        
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
      with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(1)):
        t2t_results = Parallel(verbose=100)(delayed(self.process)(segments[line]['text']) for (line) in tqdm(range(len(segments))))
      for index in tqdm(range(len(segments))):
        segments[index]['text'] = t2t_results[index]
      return segments  

if __name__ == '__main__':
  input_file = '/home/vgm/Desktop/en.srt'
  segments = srt_to_segments(input_file)
  # segments = concise_srt(segments)
  # segments_to_srt(segments, '/home/vgm/Desktop/en.srt')
  print(segments, len(segments))

  # input_texts = [
  # "Perhaps the most common type of bible reading aid is a calendar-based, daily devotional and in this short tutorial i want to show you how, how easy it is to get to your favorite daily devotional, utilizing the dashboard card we set in a previous tutorial, take a look at my screen, you'll see i've already opened the homepage, if your homepage is not open, click the home icon on the toolbar and, and there's our dashboard card, for me, i set my utmost for his highest",
  # "You set your favorite, all we have to do to get to today's reading, is click the title on the card, so i'll click my utmost for his highest, and notice that devotional opens right to today's reading, easy breezy, and please notice for me, there's a cross-reference to 2 samuel 23:16, and this is not just true for your daily devotionals, but for all of your logos resources, when you come to a bible reference in a resource notice it's blue, it's hyperlinked, hover over it, rest your cursor on it",
  # "And you will get a pop-up preview from your preferred bible, if you want your bible to jump there, then click the reference, so i'll click 2 samuel 23:16 and my preferred bible, jumps right there, in logos, there's no page turning, all you have to do is click a reference and logos will, automatically look it up for you, isn't that cool? now let's say, you have your favorite daily devotional open, but you want to read another devotional",
  # "Notice on the devotionals toolbar, there is the slanted parallel lines, click on it and here are all of your daily devotionals all you have to do, is click one in the list, and that devotional will open again to today's day, so if you want to read your bible along with a daily devotional, make sure you have that dashboard card set"
  # ]
  
  # # input_file = '/home/vgm/Desktop/test.srt'
  # # output_file = f'{input_file}.txt'
  # # input_texts = [line.strip() for line in open(input_file, 'r')]
  # input_texts = [ {"text": text} for text in input_texts]
  # print("input_texts::", len(input_texts))
  llm = LLM()
  llm.initLLM("https://infer-2.vgm.chat/v1,https://infer-3.vgm.chat/v1", "nampdn-ai/vietmistral-chatvgm-3072") ## "nampdn-ai/vietmistral-chatvgm-3072" "nampdn-ai/vietmistral-bible-translation"
  segments = llm.translate(segments)
  print("results::",  segments, len(segments))
  segments_to_srt(segments, '/home/vgm/Desktop/vi.srt')
  
  # with open(output_file, 'a') as file:
  #   for item in results:
  #     file.write(item["text"] + '\n')

