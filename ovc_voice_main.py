import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import shutil
import tempfile
app_temp_dir = os.getenv("APP_TEMP_DIR", os.path.join(tempfile.gettempdir(), "vgm-translate"))
if torch.cuda.is_available():
    device = "cuda"
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    # torch.cuda.set_device(int(CUDA_VISIBLE_DEVICES.split(',')[0]))
else:
    device = "cpu"
base_converter_dir = os.path.join(os.getcwd(),"model","openvoice", "converter")
source_voice_dir = os.path.join(os.getcwd(),"model","openvoice", "source_voice")
target_voice_dir = os.path.join(os.getcwd(),"model","openvoice", "target_voice")
encode_message = "@MyShell"

class OpenVoice():
  def __init__(self):
    self.tone_color_converter = ToneColorConverter(f'{base_converter_dir}/config.json', device=device)
    self.tone_color_converter.load_ckpt(f'{base_converter_dir}/checkpoint.pth')
    
  def create_voice(self, audio_path, model_name=""):
    print("Creating voice::", audio_path, model_name)
    model_path = os.path.join(target_voice_dir, model_name)
    os.system(f"rm -rf {model_path}")
    return se_extractor.get_se(audio_path=audio_path, vc_model=self.tone_color_converter, target_dir=target_voice_dir, vad=False, model_name=model_name)

  def convert_voice(self, src_audio_path, src_voice_name, target_audio_path, target_voice_name):
    print("converting voice:", src_audio_path, src_voice_name, target_audio_path, target_voice_name)
    try:
      source_se, _ = se_extractor.get_se(audio_path=src_audio_path, vc_model=self.tone_color_converter, target_dir=source_voice_dir, vad=False, model_name=src_voice_name)
      target_se  = torch.load(os.path.join(target_voice_dir,target_voice_name, "se.pth"), map_location=device)
      self.tone_color_converter.convert(
        audio_src_path=src_audio_path, 
        src_se=source_se,
        tgt_se=target_se, 
        output_path=target_audio_path, 
        message=encode_message)
    except Exception as e:
      print('openvoice conversion error:', e)

  def batch_convert(self, segments, speaker_to_voice, speaker_to_vc):
    os.system(f'rm -rf audio/*')
    for audio, speaker in segments:
      file_input = os.path.join(app_temp_dir, "audio2", f"{audio}")
      file_output = audio
      if speaker_to_vc[speaker] != "None":
        self.convert_voice(file_input, speaker_to_voice[speaker], file_output, speaker_to_vc[speaker])
    os.system(f'mv audio/* audio2/audio')

# if __name__ == "__main__":
#   root = OpenVoice()
#   root.convert_voice("/mnt/backup/AI-data/voice/vietnamese/Hoan/wavs-22k05/Hoan_5480.wav", "vn_han_male", "output_test.wav", "vi_hoan_male")
#   # root.create_voice("resources/han_demo.mp3", "vi_han_male")