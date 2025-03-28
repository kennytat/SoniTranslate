import re
import time
from pydub import AudioSegment
from .model import get_pretrained_model
import soundfile as sf

class STTS():
  def __init__(self):
    self.name = "STTS"
 
  def predict(self, text: str = "", outpath: str = "", repo_id: str = "",  sid: str = "", speed: float = 1.0):
      print(f"Input text: {text}. repo_id: {repo_id}. sid: {sid}, speed: {speed}")
      sid = int(sid)
      tts = get_pretrained_model(repo_id, speed)

      start = time.time()
      audio = tts.generate(text, sid=sid)
      end = time.time()

      if len(audio.samples) == 0:
          raise ValueError(
              "Error in generating audios. Please read previous error messages."
          )

      duration = len(audio.samples) / audio.sample_rate

      elapsed_seconds = end - start
      rtf = elapsed_seconds / duration

      info = f"""
      Wave duration  : {duration:.3f} s <br/>
      Processing time: {elapsed_seconds:.3f} s <br/>
      RTF: {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f} <br/>
      """

      print(info)
      print(f"\nrepo_id: {repo_id}\ntext: {text}\nsid: {sid}\nspeed: {speed}")
      if outpath:
        sf.write(
            outpath,
            audio.samples,
            samplerate=audio.sample_rate,
            subtype="PCM_16",
        )
      return audio

  def text_to_speech(self, text, output_file, tts_voice, tts_speed):
      print("stts text::", text, output_file, tts_voice, tts_speed)
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
          self.predict(text=text, outpath=output_file, sid=tts_voice, speed=tts_speed)
          print("Wav segment written at: {}".format(output_file))
      # gc.collect(); torch.cuda.empty_cache()
      # time.sleep(2)
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