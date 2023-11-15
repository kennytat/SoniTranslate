import os
import shutil
from pathlib import Path

class SVCClassVoices:
    def __init__(self):
        self.file_index = "" # root

    def apply_conf(self,
                   model_voice_path00,
                   model_voice_path01,
                   model_voice_path02,
                   model_voice_path03,
                   model_voice_path04,
                   model_voice_path05):
      self.model_voice_path00 = model_voice_path00
      self.model_voice_path01 = model_voice_path01
      self.model_voice_path02 = model_voice_path02
      self.model_voice_path03 = model_voice_path03
      self.model_voice_path04 = model_voice_path04
      self.model_voice_path05 = model_voice_path05
      
    def __call__(self, speaker_list, audio_files):
      try:
        speaker_to_model = {
            'SPEAKER_00': self.model_voice_path00,
            'SPEAKER_01': self.model_voice_path01,
            'SPEAKER_02': self.model_voice_path02,
            'SPEAKER_03': self.model_voice_path03,
            'SPEAKER_04': self.model_voice_path04,
            'SPEAKER_05': self.model_voice_path05,
        }
        speakers = list(dict.fromkeys(speaker_list))
        for speaker in speakers:
          speaker_dir = os.path.join("audio2", speaker)
          if os.path.exists(speaker_dir): shutil.rmtree(speaker_dir, ignore_errors=True)
          Path(speaker_dir).mkdir(parents=True, exist_ok=True)
          
        for index, audio_file in enumerate(audio_files):
          speaker_dir = os.path.join("audio2", speaker_list[index])
          os.system(f"ln -n {os.path.join('audio2', audio_file)} {os.path.join(speaker_dir, os.path.basename(audio_file))}")
          # os.system(f"rm {os.path.join(speaker_dir, os.path.basename(audio_file))}")
        
        for speaker in speakers:
          input_dir = os.path.join('audio2',speaker)
          model_name = speaker_to_model.get(speaker, self.model_voice_path00) #'vn_han_male'
          SVC_MODEL_DIR = os.path.join(os.getcwd(),"model","svc", model_name)
          model_path = os.path.join(SVC_MODEL_DIR, "G.pth")
          config_path = os.path.join(SVC_MODEL_DIR, "config.json")
          output_dir = f'{input_dir}.out'
          print('svc command:', f'svc infer -re -m {model_path} -c {config_path} {input_dir}')
          os.system(f'svc infer -re -m {model_path} -c {config_path} {input_dir}')
          if os.path.exists(input_dir): shutil.rmtree(input_dir, ignore_errors=True)
          os.system(f'mv {output_dir}/* audio2/audio')
          if os.path.exists(output_dir): shutil.rmtree(output_dir, ignore_errors=True)
      except KeyError:
        print('SVC Error:: Skip SVC')
    
# if __name__ == "__main__":
#   svc_voices = SVCClassVoices()
#   audio_files = ['audio/0.388.ogg', 'audio/16.412.ogg', 'audio/17.712.ogg', 'audio/26.094.ogg', 'audio/29.015.ogg', 'audio/46.485.ogg', 'audio/67.513.ogg','audio/76.356.ogg', 'audio/78.336.ogg', 'audio/84.198.ogg', 'audio/103.596.ogg', 'audio/108.143.ogg']
#   speakers_list = ['SPEAKER_00', 'SPEAKER_00', 'SPEAKER_01', 'SPEAKER_00', 'SPEAKER_00', 'SPEAKER_00', 'SPEAKER_01', 'SPEAKER_00', 'SPEAKER_01', 'SPEAKER_00', 'SPEAKER_00', 'SPEAKER_01']
#   svc_voices.apply_conf('vn_han_male','vn_han_male', None,None,None,None)
#   svc_voices(speakers_list,audio_files)