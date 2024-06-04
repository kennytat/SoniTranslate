
# from utils.utils import download_manager
import json
  
from pathlib import Path
import os
# def piper_tts_voices_list():
#     file_path = download_manager(
#         url="https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json",
#         path="./model/piperTTS",
#     )

#     with open(file_path, "r", encoding="utf8") as file:
#         data = json.load(file)
#     piper_id_models = [key + " VITS-onnx" for key in data.keys()]

#     return piper_id_models
  
# print(piper_tts_voices_list())

print(os.path.dirname(str(Path.cwd())))