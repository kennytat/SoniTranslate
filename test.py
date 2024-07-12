
# from utils.utils import download_manager
import json
import re
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


def normalise(text):
  # verse = TTSnorm(verse)
  text = text.replace("etc.", ",")
  text = re.sub(r"[\“\”\’\‘\!\@\#\$\%\^\&\*\(\)\_\=\+\(\)\[\]\{\}\;\:\"\,\.\<\>\/\?\\\|\`\~]+", ",", text)
  text = re.sub(r"^\,", "", text)
  text = re.sub(r"[\—\-\–]+", " ", text)
  text = re.sub(r"[\s\.\,]+(?=\s)", ", ", text)
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  text = text[:-1] if text.endswith(',') else text
  text = text + "." if not text.endswith('.') else text
  return text

metadata = "/mnt/backup/AI-data/voice/english/vgmLJSpeech/metadata.txt"
output = "/mnt/backup/AI-data/voice/english/vgmLJSpeech/metadata-lite.txt"
with open(output, "a") as lite:
  with open(metadata, "r") as file:
    lines = file.read()
    lines = lines.split("\n")
    for line in lines:
      content = line.split("|")[-1]
      content = normalise(content)
      if len(content) > 100 and len(content) < 250:
        lite.write(f"{content}\n")
    
    