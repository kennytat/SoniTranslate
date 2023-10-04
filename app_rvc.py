#%cd SoniTranslate
from dotenv import load_dotenv
import srt
import json
import re
import yt_dlp
from datetime import timedelta, datetime
from pathlib import Path,PureWindowsPath, PurePosixPath
import joblib
from joblib import Parallel, delayed
import numpy as np
import gradio as gr
import whisperx
from whisperx.utils import LANGUAGES as LANG_TRANSCRIPT
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH as DAMT, DEFAULT_ALIGN_MODELS_HF as DAMHF
from IPython.utils import capture
import torch
from gtts import gTTS
import librosa
import ffmpeg
import edge_tts
import asyncio
import gc
from pydub import AudioSegment
from tqdm import tqdm
from deep_translator import GoogleTranslator
import os
from audio_segments import create_translated_audio
from text_to_speech import make_voice_gradio
from translate_segments import translate_text
import time
import shutil
from urllib.parse import unquote
import zipfile
import rarfile
import logging
import tempfile
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)

load_dotenv()

title = "<center><strong><font size='7'>VGM Translate</font></strong></center>"

news = """ ## 📖 News
        🔥 2023/07/26: New UI and add mix options.

        🔥 2023/07/27: Fix some bug processing the video and audio.

        🔥 2023/08/01: Add options for use RVC models.

        🔥 2023/08/02: Added support for Arabic, Czech, Danish, Finnish, Greek, Hebrew, Hungarian, Korean, Persian, Polish, Russian, Turkish, Urdu, Hindi, and Vietnamese languages. 🌐

        🔥 2023/08/03: Changed default options and added directory view of downloads..
        """

description = """
### 🎥 **Translate videos easily with VGM Translate!** 📽️

🎥 Upload a video or provide a video link. 📽️
🎥 Upload SRT File for skiping S2T & T2T 📽️
 - SRT Format: "<video|audio name>-<language>.srt" - Example: "video-vi.srt"
 - See the tab labeled 'Help' for instructions on how to use it. Let's start having fun with video translation! 🚀🎉
"""



tutorial = """

# 🔰 **Instructions for use:**

1. 📤 **Upload a video** on the first tab or 🌐 **use a video link** on the second tab.

2. 🌍 Choose the language in which you want to **translate the video**.

3. 🗣️ Specify the **number of people speaking** in the video and **assign each one a text-to-speech voice** suitable for the translation language.

4. 🚀 Press the '**Translate**' button to obtain the results.


# 🎤 How to Use RVC and RVC2 Voices 🎶

The goal is to apply a RVC (Retrieval-based Voice Conversion) to the generated TTS (Text-to-Speech) 🎙️

1. In the `Custom Voice RVC` tab, download the models you need 📥 You can use links from Hugging Face and Google Drive in formats like zip, pth, or index. You can also download complete HF space repositories, but this option is not very stable 😕

2. Now, go to `Replace voice: TTS to RVC` and check the `enable` box ✅ After this, you can choose the models you want to apply to each TTS speaker 👩‍🦰👨‍🦱👩‍🦳👨‍🦲

3. Adjust the F0 method that will be applied to all RVCs 🎛️

4. Press `APPLY CONFIGURATION` to apply the changes you made 🔄

5. Go back to the video translation tab and click on 'Translate' ▶️ Now, the translation will be done applying the RVCs 🗣️

Tip: You can use `Test RVC` to experiment and find the best TTS or configurations to apply to the RVC 🧪🔍

"""

LANGUAGES = {
    'Automatic detection': 'Automatic detection',
    'Arabic (ar)': 'ar',
    'Chinese (zh)': 'zh',
    'Czech (cs)': 'cs',
    'Danish (da)': 'da',
    'Dutch (nl)': 'nl',
    'English (en)': 'en',
    'Finnish (fi)': 'fi',
    'French (fr)': 'fr',
    'German (de)': 'de',
    'Greek (el)': 'el',
    'Hebrew (he)': 'he',
    'Hungarian (hu)': 'hu',
    'Italian (it)': 'it',
    'Japanese (ja)': 'ja',
    'Korean (ko)': 'ko',
    'Persian (fa)': 'fa',
    'Polish (pl)': 'pl',
    'Portuguese (pt)': 'pt',
    'Russian (ru)': 'ru',
    'Spanish (es)': 'es',
    'Turkish (tr)': 'tr',
    'Ukrainian (uk)': 'uk',
    'Urdu (ur)': 'ur',
    'Vietnamese (vi)': 'vi',
    'Hindi (hi)': 'hi',
}

ydl = yt_dlp.YoutubeDL()
# Check GPU
CUDA_MEM = int(torch.cuda.get_device_properties(0).total_memory)
print("CUDA_MEM::", CUDA_MEM)
if torch.cuda.is_available():
    device = "cuda"
    list_compute_type = ['float16', 'float32']
    compute_type_default = 'float16'
    whisper_model_default = 'large-v2' if CUDA_MEM > 13000000000 else 'medium'
else:
    device = "cpu"
    list_compute_type = ['float32']
    compute_type_default = 'float32'
    whisper_model_default = 'medium'
print('Working in: ', device)



list_tts = ['af-ZA-AdriNeural-Female', 'af-ZA-WillemNeural-Male', 'am-ET-AmehaNeural-Male', 'am-ET-MekdesNeural-Female', 'ar-AE-FatimaNeural-Female', 'ar-AE-HamdanNeural-Male', 'ar-BH-AliNeural-Male', 'ar-BH-LailaNeural-Female', 'ar-DZ-AminaNeural-Female', 'ar-DZ-IsmaelNeural-Male', 'ar-EG-SalmaNeural-Female', 'ar-EG-ShakirNeural-Male', 'ar-IQ-BasselNeural-Male', 'ar-IQ-RanaNeural-Female', 'ar-JO-SanaNeural-Female', 'ar-JO-TaimNeural-Male', 'ar-KW-FahedNeural-Male', 'ar-KW-NouraNeural-Female', 'ar-LB-LaylaNeural-Female', 'ar-LB-RamiNeural-Male', 'ar-LY-ImanNeural-Female', 'ar-LY-OmarNeural-Male', 'ar-MA-JamalNeural-Male', 'ar-MA-MounaNeural-Female', 'ar-OM-AbdullahNeural-Male', 'ar-OM-AyshaNeural-Female', 'ar-QA-AmalNeural-Female', 'ar-QA-MoazNeural-Male', 'ar-SA-HamedNeural-Male', 'ar-SA-ZariyahNeural-Female', 'ar-SY-AmanyNeural-Female', 'ar-SY-LaithNeural-Male', 'ar-TN-HediNeural-Male', 'ar-TN-ReemNeural-Female', 'ar-YE-MaryamNeural-Female', 'ar-YE-SalehNeural-Male', 'az-AZ-BabekNeural-Male', 'az-AZ-BanuNeural-Female', 'bg-BG-BorislavNeural-Male', 'bg-BG-KalinaNeural-Female', 'bn-BD-NabanitaNeural-Female', 'bn-BD-PradeepNeural-Male', 'bn-IN-BashkarNeural-Male', 'bn-IN-TanishaaNeural-Female', 'bs-BA-GoranNeural-Male', 'bs-BA-VesnaNeural-Female', 'ca-ES-EnricNeural-Male', 'ca-ES-JoanaNeural-Female', 'cs-CZ-AntoninNeural-Male', 'cs-CZ-VlastaNeural-Female', 'cy-GB-AledNeural-Male', 'cy-GB-NiaNeural-Female', 'da-DK-ChristelNeural-Female', 'da-DK-JeppeNeural-Male', 'de-AT-IngridNeural-Female', 'de-AT-JonasNeural-Male', 'de-CH-JanNeural-Male', 'de-CH-LeniNeural-Female', 'de-DE-AmalaNeural-Female', 'de-DE-ConradNeural-Male', 'de-DE-KatjaNeural-Female', 'de-DE-KillianNeural-Male', 'el-GR-AthinaNeural-Female', 'el-GR-NestorasNeural-Male', 'en-AU-NatashaNeural-Female', 'en-AU-WilliamNeural-Male', 'en-CA-ClaraNeural-Female', 'en-CA-LiamNeural-Male', 'en-GB-LibbyNeural-Female', 'en-GB-MaisieNeural-Female', 'en-GB-RyanNeural-Male', 'en-GB-SoniaNeural-Female', 'en-GB-ThomasNeural-Male', 'en-HK-SamNeural-Male', 'en-HK-YanNeural-Female', 'en-IE-ConnorNeural-Male', 'en-IE-EmilyNeural-Female', 'en-IN-NeerjaExpressiveNeural-Female', 'en-IN-NeerjaNeural-Female', 'en-IN-PrabhatNeural-Male', 'en-KE-AsiliaNeural-Female', 'en-KE-ChilembaNeural-Male', 'en-NG-AbeoNeural-Male', 'en-NG-EzinneNeural-Female', 'en-NZ-MitchellNeural-Male', 'en-NZ-MollyNeural-Female', 'en-PH-JamesNeural-Male', 'en-PH-RosaNeural-Female', 'en-SG-LunaNeural-Female', 'en-SG-WayneNeural-Male', 'en-TZ-ElimuNeural-Male', 'en-TZ-ImaniNeural-Female', 'en-US-AnaNeural-Female', 'en-US-AriaNeural-Female', 'en-US-ChristopherNeural-Male', 'en-US-EricNeural-Male', 'en-US-GuyNeural-Male', 'en-US-JennyNeural-Female', 'en-US-MichelleNeural-Female', 'en-US-RogerNeural-Male', 'en-US-SteffanNeural-Male', 'en-ZA-LeahNeural-Female', 'en-ZA-LukeNeural-Male', 'es-AR-ElenaNeural-Female', 'es-AR-TomasNeural-Male', 'es-BO-MarceloNeural-Male', 'es-BO-SofiaNeural-Female', 'es-CL-CatalinaNeural-Female', 'es-CL-LorenzoNeural-Male', 'es-CO-GonzaloNeural-Male', 'es-CO-SalomeNeural-Female', 'es-CR-JuanNeural-Male', 'es-CR-MariaNeural-Female', 'es-CU-BelkysNeural-Female', 'es-CU-ManuelNeural-Male', 'es-DO-EmilioNeural-Male', 'es-DO-RamonaNeural-Female', 'es-EC-AndreaNeural-Female', 'es-EC-LuisNeural-Male', 'es-ES-AlvaroNeural-Male', 'es-ES-ElviraNeural-Female', 'es-GQ-JavierNeural-Male', 'es-GQ-TeresaNeural-Female', 'es-GT-AndresNeural-Male', 'es-GT-MartaNeural-Female', 'es-HN-CarlosNeural-Male', 'es-HN-KarlaNeural-Female', 'es-MX-DaliaNeural-Female', 'es-MX-JorgeNeural-Male', 'es-NI-FedericoNeural-Male', 'es-NI-YolandaNeural-Female', 'es-PA-MargaritaNeural-Female', 'es-PA-RobertoNeural-Male', 'es-PE-AlexNeural-Male', 'es-PE-CamilaNeural-Female', 'es-PR-KarinaNeural-Female', 'es-PR-VictorNeural-Male', 'es-PY-MarioNeural-Male', 'es-PY-TaniaNeural-Female', 'es-SV-LorenaNeural-Female', 'es-SV-RodrigoNeural-Male', 'es-US-AlonsoNeural-Male', 'es-US-PalomaNeural-Female', 'es-UY-MateoNeural-Male', 'es-UY-ValentinaNeural-Female', 'es-VE-PaolaNeural-Female', 'es-VE-SebastianNeural-Male', 'et-EE-AnuNeural-Female', 'et-EE-KertNeural-Male', 'fa-IR-DilaraNeural-Female', 'fa-IR-FaridNeural-Male', 'fi-FI-HarriNeural-Male', 'fi-FI-NooraNeural-Female', 'fil-PH-AngeloNeural-Male', 'fil-PH-BlessicaNeural-Female', 'fr-BE-CharlineNeural-Female', 'fr-BE-GerardNeural-Male', 'fr-CA-AntoineNeural-Male', 'fr-CA-JeanNeural-Male', 'fr-CA-SylvieNeural-Female', 'fr-CH-ArianeNeural-Female', 'fr-CH-FabriceNeural-Male', 'fr-FR-DeniseNeural-Female', 'fr-FR-EloiseNeural-Female', 'fr-FR-HenriNeural-Male', 'ga-IE-ColmNeural-Male', 'ga-IE-OrlaNeural-Female', 'gl-ES-RoiNeural-Male', 'gl-ES-SabelaNeural-Female', 'gu-IN-DhwaniNeural-Female', 'gu-IN-NiranjanNeural-Male', 'he-IL-AvriNeural-Male', 'he-IL-HilaNeural-Female', 'hi-IN-MadhurNeural-Male', 'hi-IN-SwaraNeural-Female', 'hr-HR-GabrijelaNeural-Female', 'hr-HR-SreckoNeural-Male', 'hu-HU-NoemiNeural-Female', 'hu-HU-TamasNeural-Male', 'id-ID-ArdiNeural-Male', 'id-ID-GadisNeural-Female', 'is-IS-GudrunNeural-Female', 'is-IS-GunnarNeural-Male', 'it-IT-DiegoNeural-Male', 'it-IT-ElsaNeural-Female', 'it-IT-IsabellaNeural-Female', 'ja-JP-KeitaNeural-Male', 'ja-JP-NanamiNeural-Female', 'jv-ID-DimasNeural-Male', 'jv-ID-SitiNeural-Female', 'ka-GE-EkaNeural-Female', 'ka-GE-GiorgiNeural-Male', 'kk-KZ-AigulNeural-Female', 'kk-KZ-DauletNeural-Male', 'km-KH-PisethNeural-Male', 'km-KH-SreymomNeural-Female', 'kn-IN-GaganNeural-Male', 'kn-IN-SapnaNeural-Female', 'ko-KR-InJoonNeural-Male', 'ko-KR-SunHiNeural-Female', 'lo-LA-ChanthavongNeural-Male', 'lo-LA-KeomanyNeural-Female', 'lt-LT-LeonasNeural-Male', 'lt-LT-OnaNeural-Female', 'lv-LV-EveritaNeural-Female', 'lv-LV-NilsNeural-Male', 'mk-MK-AleksandarNeural-Male', 'mk-MK-MarijaNeural-Female', 'ml-IN-MidhunNeural-Male', 'ml-IN-SobhanaNeural-Female', 'mn-MN-BataaNeural-Male', 'mn-MN-YesuiNeural-Female', 'mr-IN-AarohiNeural-Female', 'mr-IN-ManoharNeural-Male', 'ms-MY-OsmanNeural-Male', 'ms-MY-YasminNeural-Female', 'mt-MT-GraceNeural-Female', 'mt-MT-JosephNeural-Male', 'my-MM-NilarNeural-Female', 'my-MM-ThihaNeural-Male', 'nb-NO-FinnNeural-Male', 'nb-NO-PernilleNeural-Female', 'ne-NP-HemkalaNeural-Female', 'ne-NP-SagarNeural-Male', 'nl-BE-ArnaudNeural-Male', 'nl-BE-DenaNeural-Female', 'nl-NL-ColetteNeural-Female', 'nl-NL-FennaNeural-Female', 'nl-NL-MaartenNeural-Male', 'pl-PL-MarekNeural-Male', 'pl-PL-ZofiaNeural-Female', 'ps-AF-GulNawazNeural-Male', 'ps-AF-LatifaNeural-Female', 'pt-BR-AntonioNeural-Male', 'pt-BR-FranciscaNeural-Female', 'pt-PT-DuarteNeural-Male', 'pt-PT-RaquelNeural-Female', 'ro-RO-AlinaNeural-Female', 'ro-RO-EmilNeural-Male', 'ru-RU-DmitryNeural-Male', 'ru-RU-SvetlanaNeural-Female', 'si-LK-SameeraNeural-Male', 'si-LK-ThiliniNeural-Female', 'sk-SK-LukasNeural-Male', 'sk-SK-ViktoriaNeural-Female', 'sl-SI-PetraNeural-Female', 'sl-SI-RokNeural-Male', 'so-SO-MuuseNeural-Male', 'so-SO-UbaxNeural-Female', 'sq-AL-AnilaNeural-Female', 'sq-AL-IlirNeural-Male', 'sr-RS-NicholasNeural-Male', 'sr-RS-SophieNeural-Female', 'su-ID-JajangNeural-Male', 'su-ID-TutiNeural-Female', 'sv-SE-MattiasNeural-Male', 'sv-SE-SofieNeural-Female', 'sw-KE-RafikiNeural-Male', 'sw-KE-ZuriNeural-Female', 'sw-TZ-DaudiNeural-Male', 'sw-TZ-RehemaNeural-Female', 'ta-IN-PallaviNeural-Female', 'ta-IN-ValluvarNeural-Male', 'ta-LK-KumarNeural-Male', 'ta-LK-SaranyaNeural-Female', 'ta-MY-KaniNeural-Female', 'ta-MY-SuryaNeural-Male', 'ta-SG-AnbuNeural-Male', 'ta-SG-VenbaNeural-Female', 'te-IN-MohanNeural-Male', 'te-IN-ShrutiNeural-Female', 'th-TH-NiwatNeural-Male', 'th-TH-PremwadeeNeural-Female', 'tr-TR-AhmetNeural-Male', 'tr-TR-EmelNeural-Female', 'uk-UA-OstapNeural-Male', 'uk-UA-PolinaNeural-Female', 'ur-IN-GulNeural-Female', 'ur-IN-SalmanNeural-Male', 'ur-PK-AsadNeural-Male', 'ur-PK-UzmaNeural-Female', 'uz-UZ-MadinaNeural-Female', 'uz-UZ-SardorNeural-Male', 'vi-VN-HoaiMyNeural-Female', 'vi-VN-NamMinhNeural-Male', 'zh-CN-XiaoxiaoNeural-Female', 'zh-CN-XiaoyiNeural-Female', 'zh-CN-YunjianNeural-Male', 'zh-CN-YunxiNeural-Male', 'zh-CN-YunxiaNeural-Male', 'zh-CN-YunyangNeural-Male', 'zh-CN-liaoning-XiaobeiNeural-Female', 'zh-CN-shaanxi-XiaoniNeural-Female']

### voices
with capture.capture_output() as cap:
    os.system('rm -rf *.wav *.mp3 *.ogg *.mp4')
    os.system('mkdir -p downloads')
    os.system('mkdir -p model/logs')
    os.system('mkdir -p model/weights')
    os.system(f'rm -rf {os.path.join(tempfile.gettempdir(), "vgm-translate")}/*')
    del cap


def print_tree_directory(root_dir, indent=''):
    if not os.path.exists(root_dir):
        print(f"{indent}Invalid directory or file: {root_dir}")
        return

    items = os.listdir(root_dir)

    for index, item in enumerate(sorted(items)):
        item_path = os.path.join(root_dir, item)
        is_last_item = index == len(items) - 1

        if os.path.isfile(item_path) and item_path.endswith('.zip'):
            with zipfile.ZipFile(item_path, 'r') as zip_file:
                print(f"{indent}{'└──' if is_last_item else '├──'} {item} (zip file)")
                zip_contents = zip_file.namelist()
                for zip_item in sorted(zip_contents):
                    print(f"{indent}{'    ' if is_last_item else '│   '}{zip_item}")
        else:
            print(f"{indent}{'└──' if is_last_item else '├──'} {item}")

            if os.path.isdir(item_path):
                new_indent = indent + ('    ' if is_last_item else '│   ')
                print_tree_directory(item_path, new_indent)


def upload_model_list():
    weight_root = os.path.join("model","weights")
    models = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            models.append(name)

    index_root = os.path.join("model","logs")
    index_paths = []
    for name in os.listdir(index_root):
        if name.endswith(".index"):
            index_paths.append(name)
            # index_paths.append(os.path.join(index_root, name))

    print("rvc models::", len(models))
    return models, index_paths

def manual_download(url, dst):
    token = os.getenv("YOUR_HF_TOKEN")
    user_header = f"\"Authorization: Bearer {token}\""

    if 'drive.google' in url:
        print("Drive link")
        if 'folders' in url:
            print("folder")
            os.system(f'gdown --folder "{url}" -O {dst} --fuzzy -c')
        else:
            print("single")
            os.system(f'gdown "{url}" -O {dst} --fuzzy -c')
    elif 'huggingface' in url:
        print("HuggingFace link")
        if '/blob/' in url or '/resolve/' in url:
          if '/blob/' in url:
              url = url.replace('/blob/', '/resolve/')
          #parsed_link = '\n{}\n\tout={}'.format(url, unquote(url.split('/')[-1]))
          #os.system(f'echo -e "{parsed_link}" | aria2c --header={user_header} --console-log-level=error --summary-interval=10 -i- -j5 -x16 -s16 -k1M -c -d "{dst}"')
          os.system(f"wget -P {dst} {url}")
        else:
          os.system(f"git clone {url} {dst+'repo/'}")
    elif 'http' in url or 'magnet' in url:
        parsed_link = '"{}"'.format(url)
        os.system(f'aria2c --optimize-concurrent-downloads --console-log-level=error --summary-interval=10 -j5 -x16 -s16 -k1M -c -d {dst} -Z {parsed_link}')


def download_list(text_downloads):
    try:
      urls = [elem.strip() for elem in text_downloads.split(',')]
    except:
      return 'No valid link'

    os.system('mkdir -p downloads')
    os.system('mkdir -p model/logs')
    os.system('mkdir -p model/weights')
    path_download = "downloads/"
    for url in urls:
      manual_download(url, path_download)
    
    # Tree
    print('####################################')
    print_tree_directory("downloads", indent='')
    print('####################################')

    # Place files
    select_zip_and_rar_files("downloads/")

    models, _ = upload_model_list()
    os.system("rm -rf downloads/repo")

    return f"Downloaded = {models}"


def select_zip_and_rar_files(directory_path="downloads/"):
    #filter
    zip_files = []
    rar_files = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".zip"):
            zip_files.append(file_name)
        elif file_name.endswith(".rar"):
            rar_files.append(file_name)

    # extract
    for file_name in zip_files:
        file_path = os.path.join(directory_path, file_name)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(directory_path)

    for file_name in rar_files:
        file_path = os.path.join(directory_path, file_name)
        with rarfile.RarFile(file_path, 'r') as rar_ref:
            rar_ref.extractall(directory_path)

    # set in path
    def move_files_with_extension(src_dir, extension, destination_dir):
        for root, _, files in os.walk(src_dir):
            for file_name in files:
                if file_name.endswith(extension):
                    source_file = os.path.join(root, file_name)
                    destination = os.path.join(destination_dir, file_name)
                    shutil.move(source_file, destination)

    move_files_with_extension(directory_path, ".index", os.path.join("model","logs"))
    move_files_with_extension(directory_path, ".pth", os.path.join("model","weights"))

    return 'Download complete'

def custom_rvc_model_voice_enable(enable_custom_voice):
    if enable_custom_voice:
        os.environ["RVC_VOICES_MODELS"] = 'ENABLE'
    else:
        os.environ["RVC_VOICES_MODELS"] = 'DISABLE'

def custom_svc_model_voice_enable(enable_custom_voice):
    if enable_custom_voice:
        os.environ["SVC_VOICES_MODELS"] = 'ENABLE'
    else:
        os.environ["SVC_VOICES_MODELS"] = 'DISABLE'
           
def new_dir_now():
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d%H%M")
    return date_time
  
def segments_to_srt(segments, output_path):
  # print("segments_to_srt::", type(segments[0]), segments)
  def srt_time(str):
    return re.sub(r"\.",",",re.sub(r"0{3}$","",str)) if re.search(r"\.\d{6}", str) else f'{str},000'
  for index, segment in enumerate(segments):
      startTime = srt_time(str(0)+str(timedelta(seconds=segment['start'])))
      endTime = srt_time(str(0)+str(timedelta(seconds=segment['end'])))
      text = segment['text']
      segmentId = index+1
      segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text and text[0] == ' ' else text}\n\n"
      with open(output_path, 'a', encoding='utf-8') as srtFile:
          srtFile.write(segment)

def srt_to_segments(segments, srt_input_path):
  srt_input = open(srt_input_path, 'r').read()
  srt_list = list(srt.parse(srt_input))
  srt_segments = list([vars(obj) for obj in srt_list])
  for i, segment in enumerate(segments):
    segments[i]['start'] = srt_segments[i]['start'].total_seconds()
    segments[i]['end'] = srt_segments[i]['end'].total_seconds()
    segments[i]['text'] = str(srt_segments[i]['content'])
    del segments[i]['words']
    del segments[i]['chars']
  # print("srt_to_segments::", type(segments), segments)
  return segments
          
def segments_to_txt(segments, output_path):
  for segment in segments:
      text = segment['text']
      segment = f"{text[1:] if text[0] == ' ' else text}\n"
      with open(output_path, 'a', encoding='utf-8') as txtFile:
          txtFile.write(segment)
                   
def is_video_or_audio(file_path):
    try:
        info = ffmpeg.probe(file_path, select_streams='v:0', show_entries='stream=codec_type')
        if len(info["streams"]) > 0 and info["streams"][0]["codec_type"] == "video":
            return "video"
    except ffmpeg.Error:
        print("ffmpeg error:")
        pass

    try:
        info = ffmpeg.probe(file_path, select_streams='a:0', show_entries='stream=codec_type')
        if len(info["streams"]) > 0 and info["streams"][0]["codec_type"] == "audio":
            return "audio"
    except ffmpeg.Error:
        print("ffmpeg error:")
        pass
    return "Unknown"

def is_windows_path(path):
    # Use a regular expression to check for a Windows drive letter and separator
    windows_path_pattern = re.compile(r"^[A-Za-z]:\\")
    return bool(windows_path_pattern.match(path))
  
def youtube_download(url, output_path):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'force_overwrites': True,
        'max_downloads': 5,
        'no_warnings': True,
        'ignore_no_formats_error': True,
        'restrictfilenames': True,
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
        ydl_download.download([url])

     
models, index_paths = upload_model_list()

f0_methods_voice = ["pm", "harvest", "crepe", "rmvpe"]

from voice_main import ClassVoices
voices = ClassVoices()

'''
def translate_from_media(video, WHISPER_MODEL_SIZE, batch_size, compute_type,
                         TRANSLATE_AUDIO_TO, min_speakers, max_speakers,
                         tts_voice00, tts_voice01,tts_voice02,tts_voice03,tts_voice04,tts_voice05):

    YOUR_HF_TOKEN = os.getenv("My_hf_token")

    create_translated_audio(result_diarize, audio_files, translated_output_file)

    os.system("rm -rf audio_dub_stereo.wav")
    os.system("ffmpeg -i audio_dub_solo.wav -ac 1 audio_dub_stereo.wav")

    os.system(f"rm -rf {mix_audio}")
    os.system(f'ffmpeg -y -i audio.wav -i audio_dub_stereo.wav -filter_complex "[0:0]volume=0.15[a];[1:0]volume=1.90[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio}')

    os.system(f"rm -rf {media_output}")
    os.system(f"ffmpeg -i {OutputFile} -i {mix_audio} -c:v copy -c:a copy -map 0:v -map 1:a -shortest {media_output}")

    return media_output
'''

def batch_preprocess(
  media_inputs,
  path_inputs,
  link_inputs,
  srt_inputs,
  s2t_method,
  t2t_method,
  t2s_method,
  disable_timeline,
  YOUR_HF_TOKEN,
  preview=False,
  WHISPER_MODEL_SIZE="large-v1",
  batch_size=16,
  compute_type="float16",
  SOURCE_LANGUAGE= "Automatic detection",
  TRANSLATE_AUDIO_TO="English (en)",
  min_speakers=1,
  max_speakers=2,
  tts_voice00="en-AU-WilliamNeural-Male",
  tts_voice01="en-CA-ClaraNeural-Female",
  tts_voice02="en-GB-ThomasNeural-Male",
  tts_voice03="en-GB-SoniaNeural-Female",
  tts_voice04="en-NZ-MitchellNeural-Male",
  tts_voice05="en-GB-MaisieNeural-Female",
  AUDIO_MIX_METHOD='Adjusting volumes and mixing audio',
  progress=gr.Progress(),
):
  ## Move all srt files to srt tempdir
  media_inputs = media_inputs if media_inputs is not None else []
  output = []
  srt_temp_dir = os.path.join(tempfile.gettempdir(), "vgm-translate", 'srt')
  Path(srt_temp_dir).mkdir(parents=True, exist_ok=True)
  os.system(f"rm -rf {srt_temp_dir}/*")
  youtube_temp_dir = os.path.join(tempfile.gettempdir(), "vgm-translate", 'youtube')
  Path(youtube_temp_dir).mkdir(parents=True, exist_ok=True)
  os.system(f"rm -rf {youtube_temp_dir}/*")
  
  path_inputs = [item.strip() for item in path_inputs.split(',')]
  print("path_inputs::", path_inputs)
  if path_inputs is not None and len(path_inputs) > 0 and path_inputs[0] != '':
    for media_path in path_inputs:
      media_path = media_path.strip()
      print("media_path::", media_path)
      if is_windows_path(media_path):
        window_path = PureWindowsPath(media_path)
        path_arr = [item for item in window_path.parts]
        path_arr[0] = re.sub(r'\:\\','',path_arr[0].lower())
        wsl_path = str(PurePosixPath('/mnt', *path_arr))
        print("wsl_path::", wsl_path)
        if os.path.exists(wsl_path):
          media_inputs.append(wsl_path)
        else:
          raise Exception(f"Path not exist:: {wsl_path}")
      else:
        if os.path.exists(media_path):
          media_inputs.append(media_path)
        else:
          raise Exception(f"Path not exist:: {media_path}")
          
  link_inputs = link_inputs.split(',')
  # print("link_inputs::", link_inputs)
  if link_inputs is not None and len(link_inputs) > 0 and link_inputs[0] != '':
    for url in link_inputs:
      url = url.strip()
      # print('testing url::', url.startswith( 'https://www.youtube.com' ))
      if url.startswith('https://www.youtube.com'):
        media_info =  ydl.extract_info(url, download=False)
        download_path = f"{os.path.join(youtube_temp_dir, media_info['title'])}.mp4"
        youtube_download(url, download_path)
        media_inputs.append(download_path) 
    
  if srt_inputs is not None and len(srt_inputs)> 0:
    for srt in srt_inputs:
      os.system(f"mv {srt.name} {srt_temp_dir}/")
  if media_inputs is not None and len(media_inputs)> 0:
    for media in media_inputs:
      result = translate_from_media(media, s2t_method, t2t_method, t2s_method, disable_timeline, YOUR_HF_TOKEN, preview, WHISPER_MODEL_SIZE, batch_size, compute_type, SOURCE_LANGUAGE, TRANSLATE_AUDIO_TO, min_speakers, max_speakers, tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05, AUDIO_MIX_METHOD, progress)
      output.append(result)
  return output

def tts(segment, speaker_to_voice, TRANSLATE_AUDIO_TO, t2s_method):
    text = segment['text']
    start = segment['start']
    end = segment['end']

    try:
        speaker = segment['speaker']
    except KeyError:
        segment['speaker'] = "SPEAKER_99"
        speaker = segment['speaker']
        print(f"NO SPEAKER DETECT IN SEGMENT: TTS auxiliary will be used in the segment time {segment['start'], segment['text']}")

    # make the tts audio
    filename = f"audio/{start}.ogg"

    if speaker in speaker_to_voice and speaker_to_voice[speaker] != 'None':
        make_voice_gradio(text, speaker_to_voice[speaker], filename, TRANSLATE_AUDIO_TO, t2s_method)
    elif speaker == "SPEAKER_99":
        try:
            tts = gTTS(text, lang=TRANSLATE_AUDIO_TO)
            tts.save(filename)
            print('Using GTTS')
        except:
            tts = gTTS('a', lang=TRANSLATE_AUDIO_TO)
            tts.save(filename)
            print('Error: Audio will be replaced.')

    # duration
    duration_true = end - start
    duration_tts = librosa.get_duration(filename=filename)

    # porcentaje
    porcentaje = duration_tts / duration_true

    if porcentaje > 2.1:
        porcentaje = 2.1
    elif porcentaje <= 1.2 and porcentaje >= 0.8:
        porcentaje = 1.0
    elif porcentaje <= 0.79:
        porcentaje = 0.8

    # Smooth and round
    porcentaje = round(porcentaje+0.0, 1)
    porcentaje = 1.0 if disable_timeline else porcentaje     
    
    # apply aceleration or opposite to the audio file in audio2 folder
    os.system(f"ffmpeg -y -loglevel panic -i {filename} -filter:a atempo={porcentaje} audio2/{filename}")

    duration_create = librosa.get_duration(filename=f"audio2/{filename}")
    return (filename, speaker) 
  
def translate_from_media(
    media_input,
    s2t_method,
    t2t_method,
    t2s_method,
    disable_timeline,
    YOUR_HF_TOKEN,
    preview=False,
    WHISPER_MODEL_SIZE="large-v2",
    batch_size=16,
    compute_type="float16",
    SOURCE_LANGUAGE= "Automatic detection",
    TRANSLATE_AUDIO_TO="English (en)",
    min_speakers=1,
    max_speakers=2,
    tts_voice00="en-AU-WilliamNeural-Male",
    tts_voice01="en-CA-ClaraNeural-Female",
    tts_voice02="en-GB-ThomasNeural-Male",
    tts_voice03="en-GB-SoniaNeural-Female",
    tts_voice04="en-NZ-MitchellNeural-Male",
    tts_voice05="en-GB-MaisieNeural-Female",
    AUDIO_MIX_METHOD='Adjusting volumes and mixing audio',
    progress=gr.Progress(),
    ):
    print("processing::", media_input)
    if YOUR_HF_TOKEN == "" or YOUR_HF_TOKEN == None:
      YOUR_HF_TOKEN = os.getenv("YOUR_HF_TOKEN")
      if YOUR_HF_TOKEN == None:
        print('No valid token')
        return "No valid token"
      else:
        os.environ["YOUR_HF_TOKEN"] = YOUR_HF_TOKEN

    media_input = media_input if isinstance(media_input, str) else media_input.name
    # media_input = '/home/vgm/Desktop/WE KNOW LOVE 09 17 23 - ANAHEIM CHURCH.mp4'
    # print(media_input)

    if "SET_LIMIT" == os.getenv("DEMO"):
      preview=True
      print("DEMO; set preview=True; The generation is **limited to 10 seconds** to prevent errors with the CPU. If you use a GPU, you won't have any of these limitations.")
      AUDIO_MIX_METHOD='Adjusting volumes and mixing audio'
      print("DEMO; set Adjusting volumes and mixing audio")
      WHISPER_MODEL_SIZE="medium"
      print("DEMO; set whisper model to medium")

    TRANSLATE_AUDIO_TO = LANGUAGES[TRANSLATE_AUDIO_TO]
    SOURCE_LANGUAGE = LANGUAGES[SOURCE_LANGUAGE]


    if not os.path.exists('audio'):
        os.makedirs('audio')

    if not os.path.exists('audio2/audio'):
        os.makedirs('audio2/audio')

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float32" if device == "cpu" else compute_type

    temp_dir = os.path.join(tempfile.gettempdir(), "vgm-translate", new_dir_now())
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    # is_video = True if is_video_or_audio(media_input) == 'video' else False
    is_video = True if os.path.splitext(os.path.basename(media_input.strip()))[1] == '.mp4' else False
    
    OutputFile = os.path.join(temp_dir, 'Video.mp4') if is_video else os.path.join(temp_dir, 'Audio.mp3')
    audio_wav = os.path.join(temp_dir, "audio_origin.wav")
    audio_webm = os.path.join(temp_dir, "audio_origin.webm")  
    translated_output_file = os.path.join(temp_dir, "audio_translated.ogg")
    mix_audio = os.path.join(temp_dir, "audio_mix.mp3") 
    file_name, file_extension = os.path.splitext(os.path.basename(media_input.strip().replace(' ','_')))
    media_output_name = f"{file_name}-{TRANSLATE_AUDIO_TO}{file_extension}"
    media_output = os.path.join(temp_dir, media_output_name)
    
    
    # os.system("rm -rf Video.mp4")
    # os.system("rm -rf audio_origin.webm")
    # os.system("rm -rf audio_origin.wav")

    progress(0.15, desc="Processing video...")
    if os.path.exists(media_input):
        if is_video:
          if preview:
              print('Creating a preview video of 10 seconds, to disable this option, go to advanced settings and turn off preview.')
              os.system(f'ffmpeg -y -i "{media_input}" -ss 00:00:20 -t 00:00:10 -c:v libx264 -c:a aac -strict experimental {OutputFile}')
          else:
              # Check if the file ends with ".mp4" extension
              if media_input.endswith(".mp4"):
                  destination_path = OutputFile
                  shutil.copy(media_input, destination_path)
              else:
                  print("File does not have the '.mp4' extension. Converting video.")
                  os.system(f'ffmpeg -y -i "{media_input}" -c:v libx264 -c:a aac -strict experimental {OutputFile}')
        else:
          if preview:
              print('Creating a preview video of 10 seconds, to disable this option, go to advanced settings and turn off preview.')
              os.system(f'ffmpeg -y -i "{media_input}" -ss 00:00:20 -t 00:00:10 -strict experimental {OutputFile}')
          else:
              # Check if the file ends with ".mp4" extension
              if media_input.endswith(".mp3"):
                  destination_path = OutputFile
                  shutil.copy(media_input, destination_path)
              else:
                  print("File does not have the '.mp3' extension. Converting audio.")
                  os.system(f"ffmpeg -y -i '{media_input}' -strict experimental '{OutputFile}'")   

        for i in range (120):
            time.sleep(1)
            print('process media...')
            if os.path.exists(OutputFile):
                time.sleep(1)
                os.system(f"ffmpeg -y -i '{OutputFile}' -vn -acodec pcm_s16le -ar 44100 -ac 2 '{audio_wav}'")
                time.sleep(1)
                break
            if i == 119:
              print('Error processing media')
              return

        for i in range (120):
            time.sleep(1)
            print('process audio...')
            if os.path.exists(audio_wav):
                break
            if i == 119:
              print("Error can't create the audio")
              return

    else:
        if preview:
            print('Creating a preview from the link, 10 seconds to disable this option, go to advanced settings and turn off preview.')
            #https://github.com/yt-dlp/yt-dlp/issues/2220
            mp4_ = f'yt-dlp -f "mp4" --downloader ffmpeg --downloader-args "ffmpeg_i: -ss 00:00:20 -t 00:00:10" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {media_input}'
            wav_ = "ffmpeg -y -i {OutputFile} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_wav}"
            os.system(mp4_)
            os.system(wav_)
        else:
            mp4_ = f'yt-dlp -f "mp4" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {media_input}'
            wav_ = f'python -m yt_dlp --output {audio_wav} --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --extract-audio --audio-format wav {media_input}'

            os.system(wav_)

            for i in range (120):
                time.sleep(1)
                print('process audio...')
                if os.path.exists(audio_wav) and not os.path.exists(audio_webm):
                    time.sleep(1)
                    os.system(mp4_)
                    break
                if i == 119:
                  print('Error donwloading the audio')
                  return

    print("Set file complete.")
    progress(0.30, desc="Transcribing...")

    SOURCE_LANGUAGE = None if SOURCE_LANGUAGE == 'Automatic detection' else SOURCE_LANGUAGE

    # 1. Transcribe with original whisper (batched)
    print("Start transcribing::")
    with capture.capture_output() as cap:

      model = whisperx.load_model(
          WHISPER_MODEL_SIZE,
          device,
          compute_type=compute_type,
          language= SOURCE_LANGUAGE,
          )
      del cap
    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(audio, batch_size=batch_size)
    gc.collect(); torch.cuda.empty_cache(); del model
    print("Transcript complete::")

    
    # 2. Align whisper output
    print("Start aligning::")
    progress(0.45, desc="Aligning...")
    DAMHF.update(DAMT) #lang align
    EXTRA_ALIGN = {
        "hi": "theainerd/Wav2Vec2-large-xlsr-hindi"
    } # add new align models here
    #print(result['language'], DAM.keys(), EXTRA_ALIGN.keys())
    SOURCE_LANGUAGE = result['language']
    if not result['language'] in DAMHF.keys() and not result['language'] in EXTRA_ALIGN.keys():
        audio = result = None
        print("Automatic detection: Source language not compatible")
        print(f"Detected language {result['language']}  incompatible, you can select the source language to avoid this error.")
        return

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
        model_name = None if result["language"] in DAMHF.keys() else EXTRA_ALIGN[result["language"]]
        )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=True,
        )
    gc.collect(); torch.cuda.empty_cache(); del model_a
    print("Align complete::")

    if result['segments'] == []:
        print('No active speech found in audio')
        return

    # 3. Assign speaker labels
    print("Start Diarizing::")
    progress(0.60, desc="Diarizing...")
    with capture.capture_output() as cap:
      diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
      del cap
    diarize_segments = diarize_model(
        audio_wav,
        min_speakers=min_speakers,
        max_speakers=max_speakers)
    result_diarize = whisperx.assign_word_speakers(diarize_segments, result)
    gc.collect(); torch.cuda.empty_cache(); del diarize_model
    print("Diarize complete")
    
    print("Start translating::")
    progress(0.75, desc="Translating...")
    if TRANSLATE_AUDIO_TO == "zh":
        TRANSLATE_AUDIO_TO = "zh-CN"
    if TRANSLATE_AUDIO_TO == "he":
        TRANSLATE_AUDIO_TO = "iw"
    print("os.path.splitext(media_input)[0]::", os.path.splitext(media_input)[0])
    ## Write source segment and srt,txt to file
    media_output_basename = os.path.join(temp_dir, file_name)
    segments_to_srt(result_diarize['segments'], f'{media_output_basename}-{SOURCE_LANGUAGE}.srt')
    segments_to_txt(result_diarize['segments'], f'{media_output_basename}-{SOURCE_LANGUAGE}.txt')
    with open(f'{media_output_basename}-{SOURCE_LANGUAGE}.json', 'a', encoding='utf-8') as srtFile:
      srtFile.write(json.dumps(result_diarize['segments']))
    target_srt_inputpath = os.path.join(tempfile.gettempdir(), "vgm-translate", 'srt', f'{os.path.splitext(media_output_name)[0]}.srt')
    if os.path.exists(target_srt_inputpath):
      # Start convert from srt if srt found
      print("srt file exist::", target_srt_inputpath)
      result_diarize['segments'] = srt_to_segments(result_diarize['segments'], target_srt_inputpath)
    else:
      # Start translate if srt not found
      result_diarize['segments'] = translate_text(result_diarize['segments'], TRANSLATE_AUDIO_TO, t2t_method)
    ## Write target segment and srt to file
    segments_to_srt(result_diarize['segments'], f'{media_output_basename}-{TRANSLATE_AUDIO_TO}.srt')
    with open(f'{media_output_basename}-{TRANSLATE_AUDIO_TO}.json', 'a', encoding='utf-8') as srtFile:
      srtFile.write(json.dumps(result_diarize['segments']))
    print("Translation complete")

    progress(0.85, desc="Text_to_speech...")
    audio_files = []
    speakers_list = []

    # Mapping speakers to voice variables
    speaker_to_voice = {
        'SPEAKER_00': tts_voice00,
        'SPEAKER_01': tts_voice01,
        'SPEAKER_02': tts_voice02,
        'SPEAKER_03': tts_voice03,
        'SPEAKER_04': tts_voice04,
        'SPEAKER_05': tts_voice05
    }
    print("Start TTS::")
    JOBS = os.cpu_count()/2 if t2s_method == "VietTTS" else 1
    with joblib.parallel_config(backend="multiprocessing", prefer="threads", n_jobs=int(JOBS)):
      tts_results = Parallel(verbose=100)(delayed(tts)(segment, speaker_to_voice, TRANSLATE_AUDIO_TO, t2s_method) for (segment) in tqdm(result_diarize['segments']))
    audio_files = [result[0] for result in tts_results]
    speakers_list = [result[1] for result in tts_results]

    # custom voice
    if os.getenv('SVC_VOICES_MODELS') == 'ENABLE':
        progress(0.90, desc="Applying SVC customized voices...")
        print("start SVC::")
        input_dir = os.path.join('audio2','audio')
        model_name = 'vn_han_male'
        SVC_MODEL_DIR = os.path.join(os.getcwd(),"model","svc", model_name)
        model_path = os.path.join(SVC_MODEL_DIR, "G.pth")
        config_path = os.path.join(SVC_MODEL_DIR, "config.json")
        output_dir = f'{input_dir}.out'
        os.system(f'svc infer -re -m {model_path} -c {config_path} {input_dir}')
        if os.path.exists(input_dir): shutil.rmtree(input_dir, ignore_errors=True)
        shutil.move(output_dir, input_dir)     
        
    # custom voice
    if os.getenv('RVC_VOICES_MODELS') == 'ENABLE':
        progress(0.90, desc="Applying RVC customized voices...")
        voices(speakers_list, audio_files)

    # replace files with the accelerates
    os.system("mv -f audio2/audio/*.ogg audio/")

    os.system(f"rm -rf {translated_output_file}")

    progress(0.95, desc="Creating final translated media...")

    create_translated_audio(result_diarize, audio_files, translated_output_file, disable_timeline)

    os.system(f"rm -rf {mix_audio}")

    # TYPE MIX AUDIO
    if AUDIO_MIX_METHOD == 'Adjusting volumes and mixing audio':
        # volume mix
        os.system(f'ffmpeg -y -i {audio_wav} -i {translated_output_file} -filter_complex "[0:0]volume=0.15[a];[1:0]volume=1.90[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio}')
    else:
        try:
            # background mix
            os.system(f'ffmpeg -i {audio_wav} -i {translated_output_file} -filter_complex "[1:a]asplit=2[sc][mix];[0:a][sc]sidechaincompress=threshold=0.003:ratio=20[bg]; [bg][mix]amerge[final]" -map [final] {mix_audio}')
        except:
            # volume mix except
            os.system(f'ffmpeg -y -i {audio_wav} -i {translated_output_file} -filter_complex "[0:0]volume=0.25[a];[1:0]volume=1.80[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio}')

    os.system(f"rm -rf {media_output}")
    if is_video:
      os.system(f"ffmpeg -i {OutputFile} -i {mix_audio} -c:v copy -c:a copy -map 0:v -map 1:a -shortest {media_output}")
    os.system(f"rm -rf {OutputFile}")
    if media_input.startswith('/tmp'):
      os.system(f"rm -rf {media_input}")
    ## Archve all files and return output
    archive_path = os.path.join(Path(temp_dir).parent.absolute(), os.path.splitext(os.path.basename(media_output))[0])
    shutil.make_archive(archive_path, 'zip', temp_dir)
    os.system(f"rm -rf {temp_dir}")
    final_output = f"{archive_path}.zip"
    return final_output

import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

sys.stdout = Logger("output.log")

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()

def submit_file_func(file):
    print(file.name)
    return file.name, file.name

# max tts
MAX_TTS = 6

#### video
# theme = gr.themes.Base.load(os.path.join('themes','taithrah-minimal@0.0.1.json')).set(
#             background_fill_primary ="#171717",
#             panel_background_fill = "#171717"
#         )
theme="Taithrah/Minimal"
demo = gr.Blocks(theme=theme)
with demo:
  gr.Markdown(title)
  gr.Markdown(description)
  with gr.Tabs():
    with gr.Tab("Audio Translation for a Video"):
        with gr.Row():
            with gr.Column():
                #media_input = gr.UploadButton("Click to Upload a video", file_types=["video"], file_count="single") #gr.Video() # height=300,width=300
                media_input = gr.File(label="VIDEO|AUDIO", interactive=True, file_count='directory', file_types=['audio','video'])
                path_input = gr.Textbox(label="Import Windows Path",info="Example: M:\\warehouse\\video.mp4", placeholder="Windows path goes here, seperate by comma...")        
                link_input = gr.Textbox(label="Youtube Link",info="Example: https://www.youtube.com/watch?v=M2LksyGYPoc,https://www.youtube.com/watch?v=DrG2c1vxGwU", placeholder="URL goes here, seperate by comma...")        
                srt_input = gr.File(label="SRT(Optional)", interactive=True, file_count='directory', file_types=['.srt'])
                gr.ClearButton(components=[media_input,link_input,srt_input], size='sm')
                disable_timeline = gr.Checkbox(label="Disable",container=False, interative=True, info='Disable timeline matching with origin language?')
                ## media_input change function
                # link = gr.HTML()
                # media_input.change(submit_file_func, media_input, [media_input, link], show_progress='full')

                SOURCE_LANGUAGE = gr.Dropdown(['Automatic detection', 'Arabic (ar)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='Automatic detection',label = 'Source language', info="This is the original language of the video")
                TRANSLATE_AUDIO_TO = gr.Dropdown(['Arabic (ar)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='Vietnamese (vi)',label = 'Translate audio to', info="Select the target language, and make sure to select the language corresponding to the speakers of the target language to avoid errors in the process.")

                line_ = gr.HTML("<hr></h2>")
                gr.Markdown("Select how many people are speaking in the video.")
                min_speakers = gr.Slider(1, MAX_TTS, default=1, label="min_speakers", step=1, visible=False)
                max_speakers = gr.Slider(1, MAX_TTS, value=1, step=1, label="Max speakers", interative=True)
                gr.Markdown("Select the voice you want for each speaker.")
                def submit(value):
                    visibility_dict = {
                        f'tts_voice{i:02d}': gr.update(visible=i < value) for i in range(6)
                    }
                    return [value for value in visibility_dict.values()]
                tts_voice00 = gr.Dropdown(list_tts, value='vi-VN-NamMinhNeural-Male', label = 'TTS Speaker 1', visible=True, interactive=True)
                tts_voice01 = gr.Dropdown(list_tts, value='vi-VN-HoaiMyNeural-Female', label = 'TTS Speaker 2', visible=False, interactive=True)
                tts_voice02 = gr.Dropdown(list_tts, value='en-GB-ThomasNeural-Male', label = 'TTS Speaker 3', visible=False, interactive=True)
                tts_voice03 = gr.Dropdown(list_tts, value='en-GB-SoniaNeural-Female', label = 'TTS Speaker 4', visible=False, interactive=True)
                tts_voice04 = gr.Dropdown(list_tts, value='en-NZ-MitchellNeural-Male', label = 'TTS Speaker 5', visible=False, interactive=True)
                tts_voice05 = gr.Dropdown(list_tts, value='en-GB-MaisieNeural-Female', label = 'TTS Speaker 6', visible=False, interactive=True)
                max_speakers.change(submit, max_speakers, [tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05])

                with gr.Column():
                      with gr.Accordion("Advanced Settings", open=False):

                          AUDIO_MIX = gr.Dropdown(['Mixing audio with sidechain compression', 'Adjusting volumes and mixing audio'], value='Adjusting volumes and mixing audio', label = 'Audio Mixing Method', info="Mix original and translated audio files to create a customized, balanced output with two available mixing modes.")

                          gr.HTML("<hr></h2>")
                          gr.Markdown("Default configuration of Whisper.")
                          WHISPER_MODEL_SIZE = gr.Dropdown(['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2'], value=whisper_model_default, label="Whisper model", interactive=True)
                          batch_size = gr.Slider(1, 32, value=16, label="Batch size", step=1, interactive=True)
                          compute_type = gr.Dropdown(list_compute_type, value=compute_type_default, label="Compute type", interactive=True)

                          gr.HTML("<hr></h2>")
                          # MEDIA_OUTPUT_NAME = gr.Textbox(label="Translated file name" ,value="media_output.mp4", info="The name of the output file")
                          PREVIEW = gr.Checkbox(label="Preview", info="Preview cuts the video to only 10 seconds for testing purposes. Please deactivate it to retrieve the full video duration.")
                
                ## update_output_filename if media_input or TRANSLATE_AUDIO_TO change
                # def update_output_filename(file,lang):
                #     file_name, file_extension = os.path.splitext(os.path.basename(file.name.strip().replace(' ','_')))
                #     output_name = f"{file_name}-{LANGUAGES[lang]}{file_extension}"
                #     return gr.update(value=output_name)
                # media_input.change(update_output_filename, [media_input,TRANSLATE_AUDIO_TO], [MEDIA_OUTPUT_NAME])
                # TRANSLATE_AUDIO_TO.change(update_output_filename, [media_input,TRANSLATE_AUDIO_TO], [MEDIA_OUTPUT_NAME])
                
            with gr.Column(variant='compact'):
                with gr.Row():
                    video_button = gr.Button("TRANSLATE", )
                with gr.Row():
                    media_output = gr.Files(label="DOWNLOAD TRANSLATED VIDEO") #gr.Video()

                line_ = gr.HTML("<hr></h2>")
                if os.getenv("YOUR_HF_TOKEN") == None or os.getenv("YOUR_HF_TOKEN") == "":
                  HFKEY = gr.Textbox(visible= True, label="HF Token", info="One important step is to accept the license agreement for using Pyannote. You need to have an account on Hugging Face and accept the license to use the models: https://huggingface.co/pyannote/speaker-diarization and https://huggingface.co/pyannote/segmentation. Get your KEY TOKEN here: https://hf.co/settings/tokens", placeholder="Token goes here...")
                else:
                  HFKEY = gr.Textbox(visible= False, label="HF Token", info="One important step is to accept the license agreement for using Pyannote. You need to have an account on Hugging Face and accept the license to use the models: https://huggingface.co/pyannote/speaker-diarization and https://huggingface.co/pyannote/segmentation. Get your KEY TOKEN here: https://hf.co/settings/tokens", placeholder="Token goes here...")

                # gr.Examples(
                #     examples=[
                #         [
                #             "./assets/Video_main.mp4",
                #             "",
                #             False,
                #             "large-v2",
                #             16,
                #             "float16",
                #             "Spanish (es)",
                #             "English (en)",
                #             1,
                #             2,
                #             'en-AU-WilliamNeural-Male',
                #             'en-CA-ClaraNeural-Female',
                #             'en-GB-ThomasNeural-Male',
                #             'en-GB-SoniaNeural-Female',
                #             'en-NZ-MitchellNeural-Male',
                #             'en-GB-MaisieNeural-Female',
                #             "media_output.mp4",
                #             'Adjusting volumes and mixing audio',
                #         ],
                #     ],
                #     fn=translate_from_media,
                #     inputs=[
                #     media_input,
                #     HFKEY,
                #     PREVIEW,
                #     WHISPER_MODEL_SIZE,
                #     batch_size,
                #     compute_type,
                #     SOURCE_LANGUAGE,
                #     TRANSLATE_AUDIO_TO,
                #     min_speakers,
                #     max_speakers,
                #     tts_voice00,
                #     tts_voice01,
                #     tts_voice02,
                #     tts_voice03,
                #     tts_voice04,
                #     tts_voice05,
                #     MEDIA_OUTPUT_NAME,
                #     AUDIO_MIX,
                #     ],
                #     outputs=[media_output],
                #     cache_examples=False,
                # )

    with gr.Tab("Settings"):
        with gr.Column():
          with gr.Accordion("S2T - T2T - T2S", open=False):
            with gr.Row():
              s2t_method = gr.Dropdown(["Whisper"], label='S2T', value='Whisper', visible=True, interactive= True)
              t2t_method = gr.Dropdown(["Google", "Meta", "VB", "T5"], label='T2T', value='VB', visible=True, interactive= True)
              t2s_method = gr.Dropdown(["Google", "Edge", "Meta", "VietTTS"], label='T2S', value='VietTTS', visible=True, interactive= True)
            # def update_models():
            #   models, index_paths = upload_model_list()
            #   for i in range(8):                      
            #     dict_models = {
            #         f'model_voice_path{i:02d}': gr.update(choices=models) for i in range(8)
            #     }
            #     dict_index = {
            #         f'file_index2_{i:02d}': gr.update(choices=index_paths) for i in range(8)
            #     }
            #     dict_changes = {**dict_models, **dict_index}
            #     return [value for value in dict_changes.values()]
              
        with gr.Column():
          with gr.Accordion("Download RVC Models", open=False):
            url_links = gr.Textbox(label="URLs", value="",info="Automatically download the RVC models from the URL. You can use links from HuggingFace or Drive, and you can include several links, each one separated by a comma. Example: https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.pth, https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.index", placeholder="urls here...", lines=1)
            download_finish = gr.HTML()
            download_button = gr.Button("DOWNLOAD MODELS")

            def update_models():
              models, index_paths = upload_model_list()
              for i in range(8):                      
                dict_models = {
                    f'model_voice_path{i:02d}': gr.update(choices=models) for i in range(8)
                }
                dict_index = {
                    f'file_index2_{i:02d}': gr.update(choices=index_paths) for i in range(8)
                }
                dict_changes = {**dict_models, **dict_index}
                return [value for value in dict_changes.values()]

        with gr.Column():
          with gr.Accordion("SVC Setting", open=False):
            with gr.Column(variant='compact'):
              with gr.Column():
                gr.Markdown("### 1. To enable its use, mark it as enable.")
                enable_svc_custom_voice = gr.Checkbox(label="ENABLE", value=True, info="Check this to enable the use of the models.", interactive=True)
                enable_svc_custom_voice.change(custom_rvc_model_voice_enable, [enable_svc_custom_voice], [])

            #     gr.Markdown("### 2. Select a voice that will be applied to each TTS of each corresponding speaker and apply the configurations.")
            #     gr.Markdown('Depending on how many "TTS Speaker" you will use, each one needs its respective model. Additionally, there is an auxiliary one if for some reason the speaker is not detected correctly.')
            #     gr.Markdown("Voice to apply to the first speaker.")
            #     with gr.Row():
            #       model_voice_path00 = gr.Dropdown(models, label = 'Model-1', visible=True, interactive=True)
            #       file_index2_00 = gr.Dropdown(index_paths, label = 'Index-1', visible=True, interactive=True)
            #       name_transpose00 = gr.Number(label = 'Transpose-1', value=0, visible=True, interactive=True)
            #     gr.HTML("<hr></h2>")
            #     gr.Markdown("Voice to apply to the second speaker.")
            #     with gr.Row():
            #       model_voice_path01 = gr.Dropdown(models, label='Model-2', visible=True, interactive=True)
            #       file_index2_01 = gr.Dropdown(index_paths, label='Index-2', visible=True, interactive=True)
            #       name_transpose01 = gr.Number(label='Transpose-2', value=0, visible=True, interactive=True)
            #     gr.HTML("<hr></h2>")
            #     gr.Markdown("Voice to apply to the third speaker.")
            #     with gr.Row():
            #       model_voice_path02 = gr.Dropdown(models, label='Model-3', visible=True, interactive=True)
            #       file_index2_02 = gr.Dropdown(index_paths, label='Index-3', visible=True, interactive=True)
            #       name_transpose02 = gr.Number(label='Transpose-3', value=0, visible=True, interactive=True)
            #     gr.HTML("<hr></h2>")
            #     gr.Markdown("Voice to apply to the fourth speaker.")
            #     with gr.Row():
            #       model_voice_path03 = gr.Dropdown(models, label='Model-4', visible=True, interactive=True)
            #       file_index2_03 = gr.Dropdown(index_paths, label='Index-4', visible=True, interactive=True)
            #       name_transpose03 = gr.Number(label='Transpose-4', value=0, visible=True, interactive=True)
            #     gr.HTML("<hr></h2>")
            #     gr.Markdown("Voice to apply to the fifth speaker.")
            #     with gr.Row():
            #       model_voice_path04 = gr.Dropdown(models, label='Model-5', visible=True, interactive=True)
            #       file_index2_04 = gr.Dropdown(index_paths, label='Index-5', visible=True, interactive=True)
            #       name_transpose04 = gr.Number(label='Transpose-5', value=0, visible=True, interactive=True)
            #     gr.HTML("<hr></h2>")
            #     gr.Markdown("Voice to apply to the sixth speaker.")
            #     with gr.Row():
            #       model_voice_path05 = gr.Dropdown(models, label='Model-6', visible=True, interactive=True)
            #       file_index2_05 = gr.Dropdown(index_paths, label='Index-6', visible=True, interactive=True)
            #       name_transpose05 = gr.Number(label='Transpose-6', value=0, visible=True, interactive=True)
            #     gr.HTML("<hr></h2>")
            #     gr.Markdown("- Voice to apply in case a speaker is not detected successfully.")
            #     with gr.Row():
            #       model_voice_path06 = gr.Dropdown(models, label='Model-Aux', visible=True, interactive=True)
            #       file_index2_06 = gr.Dropdown(index_paths, label='Index-Aux', visible=True, interactive=True)
            #       name_transpose06 = gr.Number(label='Transpose-Aux', value=0, visible=True, interactive=True)
            #     gr.HTML("<hr></h2>")
            #     with gr.Row():
            #       f0_method_global = gr.Dropdown(f0_methods_voice, value='harvest', label = 'Global F0 method', visible=True, interactive= True)

            # with gr.Row(variant='compact'):
            #   button_config = gr.Button("APPLY CONFIGURATION")

            #   confirm_conf = gr.HTML()

            # button_config.click(voices.apply_conf, inputs=[
            #     f0_method_global,
            #     s2t_method, t2t_method, t2s_method,
            #     model_voice_path00, name_transpose00, file_index2_00,
            #     model_voice_path01, name_transpose01, file_index2_01,
            #     model_voice_path02, name_transpose02, file_index2_02,
            #     model_voice_path03, name_transpose03, file_index2_03,
            #     model_voice_path04, name_transpose04, file_index2_04,
            #     model_voice_path05, name_transpose05, file_index2_05,
            #     model_voice_path06, name_transpose06, file_index2_06,
            #     ], outputs=[confirm_conf])


        with gr.Column():
          with gr.Accordion("RVC Setting", open=False):
            with gr.Column(variant='compact'):
              with gr.Column():
                gr.Markdown("### 1. To enable its use, mark it as enable.")
                enable_rvc_custom_voice = gr.Checkbox(label="ENABLE", info="Check this to enable the use of the models.")
                enable_rvc_custom_voice.change(custom_rvc_model_voice_enable, [enable_rvc_custom_voice], [])

                gr.Markdown("### 2. Select a voice that will be applied to each TTS of each corresponding speaker and apply the configurations.")
                gr.Markdown('Depending on how many "TTS Speaker" you will use, each one needs its respective model. Additionally, there is an auxiliary one if for some reason the speaker is not detected correctly.')
                gr.Markdown("Voice to apply to the first speaker.")
                with gr.Row():
                  model_voice_path00 = gr.Dropdown(models, label = 'Model-1', visible=True, interactive=True)
                  file_index2_00 = gr.Dropdown(index_paths, label = 'Index-1', visible=True, interactive=True)
                  name_transpose00 = gr.Number(label = 'Transpose-1', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                gr.Markdown("Voice to apply to the second speaker.")
                with gr.Row():
                  model_voice_path01 = gr.Dropdown(models, label='Model-2', visible=True, interactive=True)
                  file_index2_01 = gr.Dropdown(index_paths, label='Index-2', visible=True, interactive=True)
                  name_transpose01 = gr.Number(label='Transpose-2', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                gr.Markdown("Voice to apply to the third speaker.")
                with gr.Row():
                  model_voice_path02 = gr.Dropdown(models, label='Model-3', visible=True, interactive=True)
                  file_index2_02 = gr.Dropdown(index_paths, label='Index-3', visible=True, interactive=True)
                  name_transpose02 = gr.Number(label='Transpose-3', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                gr.Markdown("Voice to apply to the fourth speaker.")
                with gr.Row():
                  model_voice_path03 = gr.Dropdown(models, label='Model-4', visible=True, interactive=True)
                  file_index2_03 = gr.Dropdown(index_paths, label='Index-4', visible=True, interactive=True)
                  name_transpose03 = gr.Number(label='Transpose-4', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                gr.Markdown("Voice to apply to the fifth speaker.")
                with gr.Row():
                  model_voice_path04 = gr.Dropdown(models, label='Model-5', visible=True, interactive=True)
                  file_index2_04 = gr.Dropdown(index_paths, label='Index-5', visible=True, interactive=True)
                  name_transpose04 = gr.Number(label='Transpose-5', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                gr.Markdown("Voice to apply to the sixth speaker.")
                with gr.Row():
                  model_voice_path05 = gr.Dropdown(models, label='Model-6', visible=True, interactive=True)
                  file_index2_05 = gr.Dropdown(index_paths, label='Index-6', visible=True, interactive=True)
                  name_transpose05 = gr.Number(label='Transpose-6', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                gr.Markdown("- Voice to apply in case a speaker is not detected successfully.")
                with gr.Row():
                  model_voice_path06 = gr.Dropdown(models, label='Model-Aux', visible=True, interactive=True)
                  file_index2_06 = gr.Dropdown(index_paths, label='Index-Aux', visible=True, interactive=True)
                  name_transpose06 = gr.Number(label='Transpose-Aux', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                with gr.Row():
                  f0_method_global = gr.Dropdown(f0_methods_voice, value='harvest', label = 'Global F0 method', visible=True, interactive= True)

            with gr.Row(variant='compact'):
              button_config = gr.Button("APPLY CONFIGURATION")

              confirm_conf = gr.HTML()

            button_config.click(voices.apply_conf, inputs=[
                f0_method_global,
                s2t_method, t2t_method, t2s_method,
                model_voice_path00, name_transpose00, file_index2_00,
                model_voice_path01, name_transpose01, file_index2_01,
                model_voice_path02, name_transpose02, file_index2_02,
                model_voice_path03, name_transpose03, file_index2_03,
                model_voice_path04, name_transpose04, file_index2_04,
                model_voice_path05, name_transpose05, file_index2_05,
                model_voice_path06, name_transpose06, file_index2_06,
                ], outputs=[confirm_conf])


          with gr.Column():
                with gr.Accordion("Test RVC", open=False):

                  with gr.Row(variant='compact'):
                    text_test = gr.Textbox(label="Text", value="This is an example",info="write a text", placeholder="...", lines=5)
                    with gr.Column(): 
                      tts_test = gr.Dropdown(list_tts, value='en-GB-ThomasNeural-Male', label = 'TTS', visible=True, interactive= True)
                      model_voice_path07 = gr.Dropdown(models, label = 'Model', visible=True, interactive= True) #value=''
                      file_index2_07 = gr.Dropdown(index_paths, label = 'Index', visible=True, interactive= True) #value=''
                      transpose_test = gr.Number(label = 'Transpose', value=0, visible=True, interactive= True, info="integer, number of semitones, raise by an octave: 12, lower by an octave: -12")
                      f0method_test = gr.Dropdown(f0_methods_voice, value='harvest', label = 'F0 method', visible=True, interactive= True) 
                  with gr.Row(variant='compact'):
                    button_test = gr.Button("Test audio")

                  with gr.Column():
                    with gr.Row():
                      original_ttsvoice = gr.Audio()
                      ttsvoice = gr.Audio()

                    button_test.click(voices.make_test, inputs=[
                        text_test,
                        tts_test,
                        model_voice_path07,
                        file_index2_07,
                        transpose_test,
                        f0method_test,
                        ], outputs=[ttsvoice, original_ttsvoice])

                download_button.click(download_list, [url_links], [download_finish]).then(update_models, [], 
                                  [
                                    model_voice_path00, model_voice_path01, model_voice_path02, model_voice_path03, model_voice_path04, model_voice_path05, model_voice_path06, model_voice_path07,
                                    file_index2_00, file_index2_01, file_index2_02, file_index2_03, file_index2_04, file_index2_05, file_index2_06, file_index2_07
                                  ])


    with gr.Tab("Help"):
        gr.Markdown(tutorial)
        gr.Markdown(news)

    with gr.Accordion("Logs", open = False):
        logs = gr.Textbox()
        demo.load(read_logs, None, logs, every=1)

    # run
    video_button.click(batch_preprocess, inputs=[
        media_input,
        path_input,
        link_input,
        srt_input,
        s2t_method,
        t2t_method,
        t2s_method,
        disable_timeline,
        HFKEY,
        PREVIEW,
        WHISPER_MODEL_SIZE,
        batch_size,
        compute_type,
        SOURCE_LANGUAGE,
        TRANSLATE_AUDIO_TO,
        min_speakers,
        max_speakers,
        tts_voice00,
        tts_voice01,
        tts_voice02,
        tts_voice03,
        tts_voice04,
        tts_voice05,
        AUDIO_MIX,
        ], outputs=media_output)

if __name__ == "__main__":
  # os.system('rm -rf /tmp/gradio/*')
  auth_user = os.getenv('AUTH_USER', '')
  auth_pass = os.getenv('AUTH_PASS', '')
  demo.queue(concurrency_count=1).launch(
    # auth=(auth_user, auth_pass) if auth_user != '' and auth_pass != '' else None,
    show_api=False,
    debug=False,
    inbrowser=True,
    # show_error=True,
    server_name="0.0.0.0",
    server_port=6860,
    share=False)