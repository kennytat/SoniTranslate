import os
import shutil
from pathlib import Path

class SVCClassVoices:
    def __init__(self):
        self.file_index = "" # root
      
    def __call__(self, speaker_list, audio_files, speaker_to_model):
      try:
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
          model_name = speaker_to_model[speaker]
          if model_name != "None":
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
#   audio_files = ['audio/0.388.wav', 'audio/16.412.wav', 'audio/17.712.wav', 'audio/26.094.wav', 'audio/29.015.wav', 'audio/46.485.wav', 'audio/67.513.wav','audio/76.356.wav', 'audio/78.336.wav', 'audio/84.198.wav', 'audio/103.596.wav', 'audio/108.143.wav']
#   speakers_list = ['SPEAKER_00', 'SPEAKER_00', 'SPEAKER_01', 'SPEAKER_00', 'SPEAKER_00', 'SPEAKER_00', 'SPEAKER_01', 'SPEAKER_00', 'SPEAKER_01', 'SPEAKER_00', 'SPEAKER_00', 'SPEAKER_01']
#   svc_voices.apply_conf('vn_han_male','vn_han_male', None,None,None,None)
#   svc_voices(speakers_list,audio_files)