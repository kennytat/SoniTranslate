import os
import re
import time
import torch
import torchaudio
import gc
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import pinyin

# import joblib
# from tqdm import tqdm
# from joblib import Parallel, delayed

# download for mecab
# os.system("python -m unidic download")

voice_dir = os.path.join(os.getcwd(),"model", "viXTTS", "voices")
checkpoint_dir = os.path.join(os.getcwd(),"model", "viXTTS", "base_model")
os.makedirs(checkpoint_dir, exist_ok=True)

class XTTS():
  def __init__(self):
    self.name = "XTTS"
    self.tts_client = None
    self.xtts_config = os.path.join(checkpoint_dir, "config.json")
    self.config = XttsConfig()
    self.config.load_json(self.xtts_config)
    self.MODEL = Xtts.init_from_config(self.config)
    self.MODEL.load_checkpoint(self.config, checkpoint_dir=checkpoint_dir, use_deepspeed=True)
    if torch.cuda.is_available():
        self.MODEL.cuda()
    self.supported_languages = self.config.languages
    if not "vi" in self.supported_languages:
        self.supported_languages.append("vi")

  def calculate_keep_len(self, text, lang):
      """Simple hack for short sentences"""
      if lang in ["ja", "zh-cn"]:
          return -1

      word_count = len(text.split())
      num_punct = text.count(".") + text.count("!") + text.count("?") + text.count(",")

      if word_count < 5:
          return 15000 * word_count + 2000 * num_punct
      elif word_count < 10:
          return 13000 * word_count + 2000 * num_punct
      return -1

  def predict(
      self,
      text,
      outpath,
      tts_voice,
      tts_speed,
      language,
  ):
      speaker_wav = os.path.join(voice_dir, tts_voice)

      # if len(text) < 2:
      #     metrics_text = gr.Warning("Please give a longer text text")
      #     return (None, metrics_text)

      # if len(text) > 250:
      #     metrics_text = gr.Warning(
      #         str(len(text))
      #         + " characters.\n"
      #         + "Your text is too long, please keep it under 250 characters\n"
      #         + "Văn bản quá dài, vui lòng giữ dưới 250 ký tự."
      #     )
      #     return (None, metrics_text)
    
      try:
          metrics_text = ""
          t_latent = time.time()

          try:
            (
                gpt_cond_latent,
                speaker_embedding,
            ) = self.MODEL.get_conditioning_latents(
                audio_path=speaker_wav,
                gpt_cond_chunk_len=4,
                gpt_cond_len=self.MODEL.config.gpt_cond_len, 
                max_ref_length=self.MODEL.config.max_ref_len, 
                sound_norm_refs=self.MODEL.config.sound_norm_refs
            )

          except Exception as e:
              print("Speaker encoding error", str(e))

          text = re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)", r"\1 \2\2", text)

          print("I: Generating new audio...")
          t0 = time.time()
          out = self.MODEL.inference(
              text=text,
              language=language,
              gpt_cond_latent=gpt_cond_latent,
              speaker_embedding=speaker_embedding,
              speed=tts_speed,
              temperature=self.MODEL.config.temperature,
              length_penalty=self.MODEL.config.length_penalty,
              repetition_penalty=self.MODEL.config.repetition_penalty,
              top_k=self.MODEL.config.top_k,
              top_p=self.MODEL.config.top_p,
              enable_text_splitting=True,
          )
          inference_time = time.time() - t0
          print(f"I: Time to generate audio: {round(inference_time*1000)} milliseconds")
          metrics_text += (
              f"Time to generate audio: {round(inference_time*1000)} milliseconds\n"
          )
          real_time_factor = (time.time() - t0) / out["wav"].shape[-1] * 24000
          print(f"Real-time factor (RTF): {real_time_factor}")
          metrics_text += f"Real-time factor (RTF): {real_time_factor:.2f}\n"

          # Temporary hack for short sentences
          keep_len = self.calculate_keep_len(text, language)
          out["wav"] = out["wav"][:keep_len]

          torchaudio.save(outpath, torch.tensor(out["wav"]).unsqueeze(0), 24000)
          
      except RuntimeError as e:
          print("RuntimeError::", e)
      return "Done"

  def text_to_speech(self, text, output_file, tts_voice, tts_speed, language):
      text = pinyin.get(text, format="numerical")
      print("tts text::", text)
      if re.sub(r'^sil\s+','',text).isnumeric():
          silence_duration = int(re.sub(r'^sil\s+','',text)) * 1000
          print("Got integer::", text, silence_duration) 
          print("\n\n\n ==> Generating {} seconds of silence at {}".format(silence_duration, output_file))
          second_of_silence = AudioSegment.silent(duration=silence_duration) # or be explicit
          second_of_silence = second_of_silence.set_frame_rate(24000)
          second_of_silence.export(output_file, format="wav")
      elif text == "♪":
          second_of_silence = AudioSegment.silent(duration=2000) # or be explicit
          second_of_silence = second_of_silence.set_frame_rate(24000)
          second_of_silence.export(output_file, format="wav")
      else:
          self.predict(text, output_file, tts_voice, tts_speed, language)
          print("Wav segment written at: {}".format(output_file))
      gc.collect(); torch.cuda.empty_cache()
      return "Done"
      

  
# if __name__ == "__main__":
#   input_text="""
#   Trẻ con thường không nhận ra những gì người thầy của chúng làm cho chúng, các thầy cô phải làm việc vất vả để cho chúng những khám phá mới
#   nhưng nhiều lúc, học sinh trẻ chỉ làm được ít hơn là càm ràm và than phiền suốt chặng đường
#   chúng ta sẽ bàn kỹ hơn về vấn đề này. Nhưng như những người trưởng thành, khi nhìn lại, 
#   thì chúng ta nhận thấy các thầy cô giáo của mình thật tuyệt vời khi không để chúng ta tự học mà không có sự giúp đỡ, 
#   và chúng ta biết ơn những gì họ đã làm cho mình, nhưng khi nghĩ kĩ càng, thì chúng ta nên biết ơn nhiều hơn nữa vì những cơ hội mà các bài học thời thơ ấu đã cho chúng ta. Để học thêm nhiều điều mỗi ngày trong suốt cuộc đời chúng ta.
#   Trẻ con thường không nhận ra những gì người thầy của chúng làm cho chúng, các thầy cô phải làm việc vất vả để cho chúng những khám phá mới
#   nhưng nhiều lúc, học sinh trẻ chỉ làm được ít hơn là càm ràm và than phiền suốt chặng đường
#   chúng ta sẽ bàn kỹ hơn về vấn đề này. Nhưng như những người trưởng thành, khi nhìn lại, 
#   thì chúng ta nhận thấy các thầy cô giáo của mình thật tuyệt vời khi không để chúng ta tự học mà không có sự giúp đỡ, 
#   và chúng ta biết ơn những gì họ đã làm cho mình, nhưng khi nghĩ kĩ càng, thì chúng ta nên biết ơn nhiều hơn nữa vì những cơ hội mà các bài học thời thơ ấu đã cho chúng ta. Để học thêm nhiều điều mỗi ngày trong suốt cuộc đời chúng ta.
#   """
#   language="vi"
#   input_text=input_text.strip().split("\n")
#   print("lens::", len(input_text))
#   tts_voice="audio.wav"
#   with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(1)):
#     tts_results = Parallel(verbose=100)(delayed(xtts)(text, f"output{index}.wav", tts_voice, 1, language) for (index, text) in tqdm(enumerate(input_text)))
  
#   # root.xtts(input_text, output, tts_voice, 1, language)