import os
import re
import soundfile as sf
import unicodedata
import librosa
from vietTTS.txt_parser import normalize, num_to_str
from vietTTS.text2mel import text2mel
from vietTTS.mel2wave import mel2wave
from pydub import AudioSegment
from langdetect import detect_langs, detect
import json

TTS_MODEL_DIR = os.path.join(os.getcwd(),"model","vietTTS")
LEXICON_PATH = os.path.join(TTS_MODEL_DIR, "lexicon.txt")

def nat_normalize_text(text):
    def replace_sil_num(match):
      num = int(match.group(1))
      return "sil " * num
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    sil = 'sil'
    text = re.sub(r"[\n]+", f" {sil} ", text)
    text = re.sub(r",{3,}", " nghithemmotchut ", text)
    text = text.replace('"', " ")
    text = text.replace("-", " ")
    text = text.replace('–', '')
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,:;?!()♪]+", f" {sil} ", text)
    text = re.sub(f"( {sil}+)+ ", f" {sil} ", text)
    text = re.sub(r'sil(\d+)', replace_sil_num, text)
    text = re.sub("nghithemmotchut", f" {sil} {sil} ", text)
    text = re.sub("[ ]+", " ", text)
    if not re.search(r'\ssil\s*$', text):
      text = text + ' sil'
    text = num_to_str(text)
    text = text + " đây là đoạn văn kết nối không cần đọc" # This text is added to paragraph end for deductible later, equal to 2 second
    print("Nat text: ", text)
    return text.strip()

def text_to_speech(text, output_file, model_name, speed = 1):
    tts_voice_ckpt_dir = os.path.join(TTS_MODEL_DIR, model_name)
    print("Starting TTS {}".format(output_file))
    ## Init setting
    playback_rate = speed
    default_tail_time = 1.9
    default_sil_time = 0.3
    ### Get hifigan path
    config_path = os.path.join(tts_voice_ckpt_dir, "config.json")
    print("hifigan config_path::",config_path)
    data = json.load(open(config_path))
    sample_rate = data['sampling_rate']
    speed_threshold = data['speed_threshold']
    # print("hifigan config data::", sample_rate)  
    print("test text::", text)
    if re.sub(r'^sil\s+','',text).isnumeric():
        silence_duration = int(re.sub(r'^sil\s+','',text)) * 1000
        print("Got integer::", text, silence_duration) 
        print("\n\n\n ==> Generating {} seconds of silence at {}".format(silence_duration, output_file))
        second_of_silence = AudioSegment.silent(duration=silence_duration) # or be explicit
        second_of_silence = second_of_silence.set_frame_rate(sample_rate)
        second_of_silence.export(output_file, format="wav")
    else:
        text = text if detect(text) == 'vi' else ' sil '
        text = normalize(text)
        text = nat_normalize_text(text)
        
        ## Text2Mel with edited default_sil_time, text with sil_num and playback_rate
        mel = text2mel(
            text,
            LEXICON_PATH,
            default_sil_time,
            os.path.join(tts_voice_ckpt_dir, "acoustic_latest_ckpt.pickle"),
            os.path.join(tts_voice_ckpt_dir, "duration_latest_ckpt.pickle"),
            playback_rate,
            sample_rate
        )
        print("text2mel {}: {}s - sil_time".format(output_file, default_sil_time))
        ## Mel2Wave
        
        hifigan_cp = os.path.join(TTS_MODEL_DIR, f'hk_hifi_{sample_rate}.pickle')
        ### Start mel2wave
        wave = mel2wave(mel, config_path, hifigan_cp)
        
        ## Trim paragraph-end duration
        print("Trim paragraph end duration::")
        duration = librosa.get_duration(y=wave, sr=sample_rate) - ( default_tail_time / (speed_threshold * playback_rate))
        print("wav duration::", duration)
        wave = librosa.util.fix_length(wave, size=int(sample_rate * duration))
        print("wav trimmed::", wave, len(wave))
        ## Write to output file
        
        sf.write(output_file, wave, samplerate=sample_rate)

        print("mel2wav:: {}".format(output_file))
        print("Wav segment written at: {}".format(output_file))
    return "Done"


if __name__ == '__main__':
  model_name = "vn_han_male"
  raw_str = """Tiếng Việt, cũng gọi là tiếng Việt Nam hay Việt ngữ là ngôn ngữ của người Việt và là ngôn ngữ chính thức tại Việt Nam. Đây là tiếng mẹ đẻ của khoảng 85% dân cư Việt Nam cùng với hơn 4 triệu người Việt kiều. Tiếng Việt còn là ngôn ngữ thứ hai của các dân tộc thiểu số tại Việt Nam và là ngôn ngữ dân tộc thiểu số được công nhận tại Cộng hòa Séc. Dựa trên từ vựng cơ bản, tiếng Việt được phân loại là một ngôn ngữ thuộc ngữ hệ Nam Á. Tiếng Việt là ngôn ngữ có nhiều người nói nhất trong ngữ hệ này (nhiều hơn tổng số người nói của tất cả các ngôn ngữ còn lại trong ngữ hệ). Vì Việt Nam thuộc Vùng văn hoá Đông Á, tiếng Việt cũng chịu nhiều ảnh hưởng về từ tiếng Hán, do vậy là ngôn ngữ có ít điểm tương đồng nhất với các ngôn ngữ khác trong ngữ hệ Nam Á."""
  result = text_to_speech(raw_str, "/mnt/ssd256/Projects/SoniTranslate/test/viettts.ogg", model_name)
  # print("result::", result)

