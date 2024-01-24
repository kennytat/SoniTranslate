from dotenv import load_dotenv
import os
import shutil
import json
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed

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
def llm_translate(segments, endpoint, model):
    print("start llm_translate::")
    server = endpoint
    llm = ChatOpenAI(
        model=model,
        openai_api_key="EMPTY",
        openai_api_base=server,
        max_tokens=2048,
        temperature=0.5,
        model_kwargs={"stop":["<|im_end|>"]}
    )
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "Bạn là AI có khả năng dịch thuật nội dung Kinh Thánh từ tiếng Anh một cách chính xác và rất dễ hiểu cho người Việt Nam. Hãy cẩn thận dịch và chọn từ ngữ cho phù hợp."
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    memory = ConversationBufferWindowMemory(memory_key="history", return_messages=True, k=10)
    conversation = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    # N_JOBS = os.cpu_count()
    # print("Start LLM Translate:: concurrency =", N_JOBS)
    with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(10)):
      t2t_results = Parallel(verbose=100)(delayed(conversation.predict)(input=segments[line]['text']) for (line) in tqdm(range(len(segments))))
    for index in tqdm(range(len(segments))):
      segments[index]['text'] = t2t_results[index]
    return segments
  

if __name__ == '__main__':
  input_texts = [
    "And what a joy it is for me today to have.  Barbara Rainey back with us on Revive Our Heart.  She's been with us a number of times.  before our listeners love hearing from Barbara.  I love hearing from Barbara, I love the new resources she comes up with and.  her writing and her ministry.  She spent a long time friend along with her husband,",
		"Dennis, together they're the co-founders of Family Life Today.  and family life today had a lot to do with birthing Revive Our Hearts.  In fact, the early years of our recording were done in the studios in Little Rock, Arkansas.  of Family Life Ministries.  And so Barbara and Dennis have been friends.  And I'm just delighted that you get to hear from her today as we talk about a new book.",
		"that I think, Barbara, is my favorite so far.  You've written a lot of great ones.  But we're going to have a great conversation about this book.  And I want to just tell you, the book is about marriage.  And I am all ears having now been married for just over six months now.  So I'm reading this book in a way I probably wouldn't have read it before,",
		"And just so grateful for it.  But thank you for joining us.  here in Michigan.  Welcome to the allergy center of the Midwest.  I don't think it is, but anyway, I'm delighted to be here.  Yeah, we like Michigan.  And tell us a little bit about your family.  family for those who may not be familiar with you.  I want them to just get a feel of who you are and why you are deceased.",
		"in a life that you would be able to write this book on marriage.  Well, Dennis and I have been married 43 years.  and we have six children, five of them are married and.  from those five children we have twenty two grandkids which.  Every time I say it, it just kind of makes me just.  be still surprised.  It kind of takes my breath away because I think, how did that happen?",
		"I mean, it happens so fast.  I just want to know if you had to, could you give us all their middle names? I know.  No, absolutely not.  It would be a challenge.  I would have to think very carefully to make sure I got all the first names.  Right.  And in order.  22.  That is amazing.  And maybe some more at some point.  - Yeah, oh, I think so.  I don't think we're finished with 22.",
		"Wow, what a heritage.  What a blessing.  Yeah, it really is.  And it's a lot of fun.  It's hard to keep up with.  I forget a lot of birthdays.  I don't know what they like and what they don't like because there are too many.  I can't keep up with them, but it's fun."
  ]
  
  # input_file = '/home/vgm/Desktop/test.txt'
  # output_file = f'{input_file}.txt'
  # input_texts = [line.strip() for line in open(input_file, 'r')]
  
  input_texts = [ {"text": text} for text in input_texts]
  results = llm_translate(input_texts, "https://infer-2.vgm.chat/v1", "nampdn-ai/vietmistral-chatvgm-3072")
  print("input_texts::", len(input_texts))
  print("results::", len(results), results)
  
  # with open(output_file, 'a') as file:
  #   for item in results:
  #     file.write(item["text"] + '\n')

