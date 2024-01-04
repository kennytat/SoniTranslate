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
def llm_translate(segments, model):
    server = "https://serve-903.app.btngiadinh.com/v1"
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
    N_JOBS = os.cpu_count()
    print("Start LLM Translate:: concurrency =", N_JOBS)
    with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=N_JOBS):
      t2t_results = Parallel(verbose=100)(delayed(conversation.predict)(input=segments[line]['text']) for (line) in tqdm(range(len(segments))))
    for index in tqdm(range(len(segments))):
      segments[index]['text'] = t2t_results[index]
    return segments
  

if __name__ == '__main__':
  input_texts = [
    "Finally, be strong in the Lord and in the strength of His might.",
    "Put on the full armor of God, so that you will be able to stand firm against the schemes of the devil.",
    "For our struggle is not against flesh and blood, but against the rulers, against the powers, against the world forces of this darkness, against the spiritual forces of wickedness in the heavenly places.",
    "Therefore, take up the full armor of God, so that you will be able to resist on the evil day, and having done everything, to stand firm.",
    "Stand firm therefore, having belted your waist with truth, and having put on the breastplate of righteousness,",
    "and having strapped on your feet the preparation of the gospel of peace;"
]
  results = llm_translate(input_texts, "nampdn-ai/vietmistral-bible-translation-v1")
  print("input_texts::", len(input_texts))
  print("results::", len(results), results)

