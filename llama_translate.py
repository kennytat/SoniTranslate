from llama_cpp import Llama
from dotenv import load_dotenv
load_dotenv()
import os
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
from utils.utils import srt_to_segments, segments_to_srt

class LLM():
  def __init__(self, systemPrompt = "") -> None:
    self.llama_chain = []
    self.systemPrompt = systemPrompt if systemPrompt != "" else "This GPT functions as a translation tool that processes text from {source_language}, translating it into {target_language}. The output is a plain text content with a full translation in {target_language}. It accepts input in the form of {source_language} text, ensuring the texts are accurately digitized and represent the original manuscripts. The translation engine interprets and translates words into modern {target_language}, incorporating linguistic analysis to handle idiomatic expressions and cultural nuances. Response only translated text."
    
  def initLLM(self, model="./model/grama_correction/grama_correction_llama3_5b4e_f16.gguf", temp=0.3, k=30):
    # LOAD THE MODEL
    try:
      if os.path.isfile(model):        
        self.llm_chain = Llama(model_path=model, 
                    chat_format="chatml",
                    n_ctx=4096,
                    repeat_penalty=1.1,
                    temperature=temp,
                    top_p=0.95,
                    top_k=k,
                    n_gpu_layers=-1,
                    n_batch=512,
                    max_tokens=4096,
                    echo=False,
                    stop=["<|im_end|>"],
                    verbose=True,
                    )
        return True
      else:
        return False
    except Exception as e:
      print("initLlama error::", e)
      return False
        
  def process(self, text):
    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
      try:
        result = self.llm_chain.create_chat_completion(
              messages = [
                  {
                      "role": "user",
                      "content": text
                  }
              ],
        )
        print("result::", result["choices"][0]["message"]["content"].strip())
        if result["choices"][0]["message"]["content"] and "im_start" not in result["choices"][0]["message"]["content"].strip() and "im_end" not in result["choices"][0]["message"]["content"].strip():
            return result["choices"][0]["message"]["content"].strip()
      except Exception as e:
        print("error::", e)
      print(f"re-run {attempts}:")
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
  
  systemPrompt="""Sửa lỗi chính tả từ bản gốc sang bảng mới"""
  llm = LLM(systemPrompt=systemPrompt)
  llm.initLLM(
    model="./model/grama_correction/grama_correction_llama3_5b4e_f16.gguf",
    temp=0.3,
    k=10
  )
  text = "Sách giô-suê không nêu tên tác giả hay người biên soạn bản cuối cùng, chính sách này cũng không cho chúng ta biết về danh tính của tác giả hay người biên soạn cuối cùng, tựa đề sách giô-suê xuất hiện trong hầu hết các bản kinh thánh hiện đại của chúng ta, đã được thêm vào sách rất lâu sau khi sách được viết xong, nhưng các xu hướng về những quan điểm truyền thống của người do thái và cơ đốc nhân thời xưa về những vấn đề này được tóm gọn một cách rất tốt"
  result = llm.process(text)
  print(result)
  
  
  # ## Translate segments
  # input_file = '/home/vgm/Desktop/vi.srt'
  # segments = srt_to_segments(input_file)
  # print(segments, len(segments))
  # segments = llm.translate(segments=segments)
  # print("results::",  segments, len(segments))
  # segments_to_srt(segments, '/home/vgm/Desktop/vi-corrected.srt')