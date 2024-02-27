from dotenv import load_dotenv
import json
import yt_dlp
from pathlib import Path,PureWindowsPath, PurePosixPath
import joblib
from joblib import Parallel, delayed
import gradio as gr
import whisperx
from whisperx.utils import LANGUAGES as LANG_TRANSCRIPT
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH as DAMT, DEFAULT_ALIGN_MODELS_HF as DAMHF
from IPython.utils import capture
import torch
import torch.multiprocessing as mp
from gtts import gTTS
import librosa
import math
import gc
from tqdm import tqdm
import os
from audio_segments import create_translated_audio
from text_to_speech import make_voice_gradio
from translate_segments import translate_text
import time
import shutil
import logging
import tempfile
from vietTTS.utils import concise_srt
from vietTTS.upsample import Predictor
import soundfile as sf
from utils import new_dir_now, segments_to_srt, srt_to_segments, segments_to_txt, is_video_or_audio, is_windows_path, youtube_download, get_llm_models
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)

load_dotenv()
total_input = []
total_output = []
upsampler = None
title = "<center><strong><font size='7'>VGM Translate</font></strong></center>"

description = """
### 🎥 **Translate videos easily with VGM Translate!** 📽️

🎥 Upload a video or provide a video link. 📽️
🎥 Upload SRT File for skiping S2T & T2T 📽️
 - SRT Format: "<media name>-<target-language>-SPEAKER.srt" - Example: "video-vi-SPEAKER.srt"
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
    'Cantonese (yue)': 'yue',
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

# Check GPU
CUDA_MEM = int(torch.cuda.get_device_properties(0).total_memory)
if torch.cuda.is_available():
    device = "cuda"
    list_compute_type = ['float16', 'float32']
    compute_type_default = 'float16'
    whisper_model_default = 'large-v3' if CUDA_MEM > 9000000000 else 'medium'
else:
    device = "cpu"
    list_compute_type = ['float32']
    compute_type_default = 'float32'
    whisper_model_default = 'medium'

list_gtts = ['af-ZA-AdriNeural-Female', 'af-ZA-WillemNeural-Male', 'am-ET-AmehaNeural-Male', 'am-ET-MekdesNeural-Female', 'ar-AE-FatimaNeural-Female', 'ar-AE-HamdanNeural-Male', 'ar-BH-AliNeural-Male', 'ar-BH-LailaNeural-Female', 'ar-DZ-AminaNeural-Female', 'ar-DZ-IsmaelNeural-Male', 'ar-EG-SalmaNeural-Female', 'ar-EG-ShakirNeural-Male', 'ar-IQ-BasselNeural-Male', 'ar-IQ-RanaNeural-Female', 'ar-JO-SanaNeural-Female', 'ar-JO-TaimNeural-Male', 'ar-KW-FahedNeural-Male', 'ar-KW-NouraNeural-Female', 'ar-LB-LaylaNeural-Female', 'ar-LB-RamiNeural-Male', 'ar-LY-ImanNeural-Female', 'ar-LY-OmarNeural-Male', 'ar-MA-JamalNeural-Male', 'ar-MA-MounaNeural-Female', 'ar-OM-AbdullahNeural-Male', 'ar-OM-AyshaNeural-Female', 'ar-QA-AmalNeural-Female', 'ar-QA-MoazNeural-Male', 'ar-SA-HamedNeural-Male', 'ar-SA-ZariyahNeural-Female', 'ar-SY-AmanyNeural-Female', 'ar-SY-LaithNeural-Male', 'ar-TN-HediNeural-Male', 'ar-TN-ReemNeural-Female', 'ar-YE-MaryamNeural-Female', 'ar-YE-SalehNeural-Male', 'az-AZ-BabekNeural-Male', 'az-AZ-BanuNeural-Female', 'bg-BG-BorislavNeural-Male', 'bg-BG-KalinaNeural-Female', 'bn-BD-NabanitaNeural-Female', 'bn-BD-PradeepNeural-Male', 'bn-IN-BashkarNeural-Male', 'bn-IN-TanishaaNeural-Female', 'bs-BA-GoranNeural-Male', 'bs-BA-VesnaNeural-Female', 'ca-ES-EnricNeural-Male', 'ca-ES-JoanaNeural-Female', 'cs-CZ-AntoninNeural-Male', 'cs-CZ-VlastaNeural-Female', 'cy-GB-AledNeural-Male', 'cy-GB-NiaNeural-Female', 'da-DK-ChristelNeural-Female', 'da-DK-JeppeNeural-Male', 'de-AT-IngridNeural-Female', 'de-AT-JonasNeural-Male', 'de-CH-JanNeural-Male', 'de-CH-LeniNeural-Female', 'de-DE-AmalaNeural-Female', 'de-DE-ConradNeural-Male', 'de-DE-KatjaNeural-Female', 'de-DE-KillianNeural-Male', 'el-GR-AthinaNeural-Female', 'el-GR-NestorasNeural-Male', 'en-AU-NatashaNeural-Female', 'en-AU-WilliamNeural-Male', 'en-CA-ClaraNeural-Female', 'en-CA-LiamNeural-Male', 'en-GB-LibbyNeural-Female', 'en-GB-MaisieNeural-Female', 'en-GB-RyanNeural-Male', 'en-GB-SoniaNeural-Female', 'en-GB-ThomasNeural-Male', 'en-HK-SamNeural-Male', 'en-HK-YanNeural-Female', 'en-IE-ConnorNeural-Male', 'en-IE-EmilyNeural-Female', 'en-IN-NeerjaExpressiveNeural-Female', 'en-IN-NeerjaNeural-Female', 'en-IN-PrabhatNeural-Male', 'en-KE-AsiliaNeural-Female', 'en-KE-ChilembaNeural-Male', 'en-NG-AbeoNeural-Male', 'en-NG-EzinneNeural-Female', 'en-NZ-MitchellNeural-Male', 'en-NZ-MollyNeural-Female', 'en-PH-JamesNeural-Male', 'en-PH-RosaNeural-Female', 'en-SG-LunaNeural-Female', 'en-SG-WayneNeural-Male', 'en-TZ-ElimuNeural-Male', 'en-TZ-ImaniNeural-Female', 'en-US-AnaNeural-Female', 'en-US-AriaNeural-Female', 'en-US-ChristopherNeural-Male', 'en-US-EricNeural-Male', 'en-US-GuyNeural-Male', 'en-US-JennyNeural-Female', 'en-US-MichelleNeural-Female', 'en-US-RogerNeural-Male', 'en-US-SteffanNeural-Male', 'en-ZA-LeahNeural-Female', 'en-ZA-LukeNeural-Male', 'es-AR-ElenaNeural-Female', 'es-AR-TomasNeural-Male', 'es-BO-MarceloNeural-Male', 'es-BO-SofiaNeural-Female', 'es-CL-CatalinaNeural-Female', 'es-CL-LorenzoNeural-Male', 'es-CO-GonzaloNeural-Male', 'es-CO-SalomeNeural-Female', 'es-CR-JuanNeural-Male', 'es-CR-MariaNeural-Female', 'es-CU-BelkysNeural-Female', 'es-CU-ManuelNeural-Male', 'es-DO-EmilioNeural-Male', 'es-DO-RamonaNeural-Female', 'es-EC-AndreaNeural-Female', 'es-EC-LuisNeural-Male', 'es-ES-AlvaroNeural-Male', 'es-ES-ElviraNeural-Female', 'es-GQ-JavierNeural-Male', 'es-GQ-TeresaNeural-Female', 'es-GT-AndresNeural-Male', 'es-GT-MartaNeural-Female', 'es-HN-CarlosNeural-Male', 'es-HN-KarlaNeural-Female', 'es-MX-DaliaNeural-Female', 'es-MX-JorgeNeural-Male', 'es-NI-FedericoNeural-Male', 'es-NI-YolandaNeural-Female', 'es-PA-MargaritaNeural-Female', 'es-PA-RobertoNeural-Male', 'es-PE-AlexNeural-Male', 'es-PE-CamilaNeural-Female', 'es-PR-KarinaNeural-Female', 'es-PR-VictorNeural-Male', 'es-PY-MarioNeural-Male', 'es-PY-TaniaNeural-Female', 'es-SV-LorenaNeural-Female', 'es-SV-RodrigoNeural-Male', 'es-US-AlonsoNeural-Male', 'es-US-PalomaNeural-Female', 'es-UY-MateoNeural-Male', 'es-UY-ValentinaNeural-Female', 'es-VE-PaolaNeural-Female', 'es-VE-SebastianNeural-Male', 'et-EE-AnuNeural-Female', 'et-EE-KertNeural-Male', 'fa-IR-DilaraNeural-Female', 'fa-IR-FaridNeural-Male', 'fi-FI-HarriNeural-Male', 'fi-FI-NooraNeural-Female', 'fil-PH-AngeloNeural-Male', 'fil-PH-BlessicaNeural-Female', 'fr-BE-CharlineNeural-Female', 'fr-BE-GerardNeural-Male', 'fr-CA-AntoineNeural-Male', 'fr-CA-JeanNeural-Male', 'fr-CA-SylvieNeural-Female', 'fr-CH-ArianeNeural-Female', 'fr-CH-FabriceNeural-Male', 'fr-FR-DeniseNeural-Female', 'fr-FR-EloiseNeural-Female', 'fr-FR-HenriNeural-Male', 'ga-IE-ColmNeural-Male', 'ga-IE-OrlaNeural-Female', 'gl-ES-RoiNeural-Male', 'gl-ES-SabelaNeural-Female', 'gu-IN-DhwaniNeural-Female', 'gu-IN-NiranjanNeural-Male', 'he-IL-AvriNeural-Male', 'he-IL-HilaNeural-Female', 'hi-IN-MadhurNeural-Male', 'hi-IN-SwaraNeural-Female', 'hr-HR-GabrijelaNeural-Female', 'hr-HR-SreckoNeural-Male', 'hu-HU-NoemiNeural-Female', 'hu-HU-TamasNeural-Male', 'id-ID-ArdiNeural-Male', 'id-ID-GadisNeural-Female', 'is-IS-GudrunNeural-Female', 'is-IS-GunnarNeural-Male', 'it-IT-DiegoNeural-Male', 'it-IT-ElsaNeural-Female', 'it-IT-IsabellaNeural-Female', 'ja-JP-KeitaNeural-Male', 'ja-JP-NanamiNeural-Female', 'jv-ID-DimasNeural-Male', 'jv-ID-SitiNeural-Female', 'ka-GE-EkaNeural-Female', 'ka-GE-GiorgiNeural-Male', 'kk-KZ-AigulNeural-Female', 'kk-KZ-DauletNeural-Male', 'km-KH-PisethNeural-Male', 'km-KH-SreymomNeural-Female', 'kn-IN-GaganNeural-Male', 'kn-IN-SapnaNeural-Female', 'ko-KR-InJoonNeural-Male', 'ko-KR-SunHiNeural-Female', 'lo-LA-ChanthavongNeural-Male', 'lo-LA-KeomanyNeural-Female', 'lt-LT-LeonasNeural-Male', 'lt-LT-OnaNeural-Female', 'lv-LV-EveritaNeural-Female', 'lv-LV-NilsNeural-Male', 'mk-MK-AleksandarNeural-Male', 'mk-MK-MarijaNeural-Female', 'ml-IN-MidhunNeural-Male', 'ml-IN-SobhanaNeural-Female', 'mn-MN-BataaNeural-Male', 'mn-MN-YesuiNeural-Female', 'mr-IN-AarohiNeural-Female', 'mr-IN-ManoharNeural-Male', 'ms-MY-OsmanNeural-Male', 'ms-MY-YasminNeural-Female', 'mt-MT-GraceNeural-Female', 'mt-MT-JosephNeural-Male', 'my-MM-NilarNeural-Female', 'my-MM-ThihaNeural-Male', 'nb-NO-FinnNeural-Male', 'nb-NO-PernilleNeural-Female', 'ne-NP-HemkalaNeural-Female', 'ne-NP-SagarNeural-Male', 'nl-BE-ArnaudNeural-Male', 'nl-BE-DenaNeural-Female', 'nl-NL-ColetteNeural-Female', 'nl-NL-FennaNeural-Female', 'nl-NL-MaartenNeural-Male', 'pl-PL-MarekNeural-Male', 'pl-PL-ZofiaNeural-Female', 'ps-AF-GulNawazNeural-Male', 'ps-AF-LatifaNeural-Female', 'pt-BR-AntonioNeural-Male', 'pt-BR-FranciscaNeural-Female', 'pt-PT-DuarteNeural-Male', 'pt-PT-RaquelNeural-Female', 'ro-RO-AlinaNeural-Female', 'ro-RO-EmilNeural-Male', 'ru-RU-DmitryNeural-Male', 'ru-RU-SvetlanaNeural-Female', 'si-LK-SameeraNeural-Male', 'si-LK-ThiliniNeural-Female', 'sk-SK-LukasNeural-Male', 'sk-SK-ViktoriaNeural-Female', 'sl-SI-PetraNeural-Female', 'sl-SI-RokNeural-Male', 'so-SO-MuuseNeural-Male', 'so-SO-UbaxNeural-Female', 'sq-AL-AnilaNeural-Female', 'sq-AL-IlirNeural-Male', 'sr-RS-NicholasNeural-Male', 'sr-RS-SophieNeural-Female', 'su-ID-JajangNeural-Male', 'su-ID-TutiNeural-Female', 'sv-SE-MattiasNeural-Male', 'sv-SE-SofieNeural-Female', 'sw-KE-RafikiNeural-Male', 'sw-KE-ZuriNeural-Female', 'sw-TZ-DaudiNeural-Male', 'sw-TZ-RehemaNeural-Female', 'ta-IN-PallaviNeural-Female', 'ta-IN-ValluvarNeural-Male', 'ta-LK-KumarNeural-Male', 'ta-LK-SaranyaNeural-Female', 'ta-MY-KaniNeural-Female', 'ta-MY-SuryaNeural-Male', 'ta-SG-AnbuNeural-Male', 'ta-SG-VenbaNeural-Female', 'te-IN-MohanNeural-Male', 'te-IN-ShrutiNeural-Female', 'th-TH-NiwatNeural-Male', 'th-TH-PremwadeeNeural-Female', 'tr-TR-AhmetNeural-Male', 'tr-TR-EmelNeural-Female', 'uk-UA-OstapNeural-Male', 'uk-UA-PolinaNeural-Female', 'ur-IN-GulNeural-Female', 'ur-IN-SalmanNeural-Male', 'ur-PK-AsadNeural-Male', 'ur-PK-UzmaNeural-Female', 'uz-UZ-MadinaNeural-Female', 'uz-UZ-SardorNeural-Male', 'vi-VN-HoaiMyNeural-Female', 'vi-VN-NamMinhNeural-Male', 'zh-CN-XiaoxiaoNeural-Female', 'zh-CN-XiaoyiNeural-Female', 'zh-CN-YunjianNeural-Male', 'zh-CN-YunxiNeural-Male', 'zh-CN-YunxiaNeural-Male', 'zh-CN-YunyangNeural-Male', 'zh-CN-liaoning-XiaobeiNeural-Female', 'zh-CN-shaanxi-XiaoniNeural-Female']
list_vtts = [voice for voice in os.listdir(os.path.join("model","vits")) if os.path.isdir(os.path.join("model","vits", voice))]
list_svc = [voice for voice in os.listdir(os.path.join("model","svc")) if os.path.isdir(os.path.join("model","svc", voice))]
list_rvc = [voice for voice in os.listdir(os.path.join("model","rvc")) if voice.endswith('.pth')]

     
# models, index_paths = upload_model_list()

f0_methods_voice = ["pm", "harvest", "crepe", "rmvpe"]

from rvc_voice_main import RVCClassVoices
rvc_voices = RVCClassVoices()

from svc_voice_main import SVCClassVoices
svc_voices = SVCClassVoices()

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

# Function to save settings to a JSON file
def save_settings(settings, filename='user_settings.json'):
    with open(filename, 'w') as f:
        json.dump(settings, f)

# Function to load settings from a JSON file
def load_settings(filename='user_settings.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

user_settings=load_settings()
 
def batch_preprocess(
  media_inputs,
  # path_inputs,
  link_inputs,
  srt_inputs,
  s2t_method,
  t2t_method,
  t2s_method,
  vc_method,
  disable_timeline,
  YOUR_HF_TOKEN,
  preview=False,
  WHISPER_MODEL_SIZE="large-v2",
  batch_size=16,
  chunk_size=5,
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
  rvc_voice00=None,
  rvc_voice01=None,
  rvc_voice02=None,
  rvc_voice03=None,
  rvc_voice04=None,
  rvc_voice05=None,
  svc_voice00=None,
  svc_voice01=None,
  svc_voice02=None,
  svc_voice03=None,
  svc_voice04=None,
  svc_voice05=None,
  AUDIO_MIX_METHOD='Adjusting volumes and mixing audio',
  progress=gr.Progress(),
):
  if vc_method == "RVC":
    rvc_voices.apply_conf(f0method='harvest',
      model_voice_path00=rvc_voice00, transpose00=0, file_index2_00=rvc_voice00.replace('.pth','.index'),
      model_voice_path01=rvc_voice01, transpose01=0, file_index2_01=rvc_voice01.replace('.pth','.index'),
      model_voice_path02=rvc_voice02, transpose02=0, file_index2_02=rvc_voice02.replace('.pth','.index'),
      model_voice_path03=rvc_voice03, transpose03=0, file_index2_03=rvc_voice03.replace('.pth','.index'),
      model_voice_path04=rvc_voice04, transpose04=0, file_index2_04=rvc_voice04.replace('.pth','.index'),
      model_voice_path05=rvc_voice05, transpose05=0, file_index2_05=rvc_voice05.replace('.pth','.index'),
      model_voice_path99=None, transpose99=0, file_index2_99=None)
  if vc_method == "SVC":
    svc_voices.apply_conf(
      model_voice_path00=svc_voice00,
      model_voice_path01=svc_voice01, 
      model_voice_path02=svc_voice02,
      model_voice_path03=svc_voice03,
      model_voice_path04=svc_voice04, 
      model_voice_path05=svc_voice05,
    )
  ## Move all srt files to srt tempdir
  media_inputs = media_inputs if media_inputs is not None else []
  media_inputs = media_inputs if isinstance(media_inputs, list) else [media_inputs]
  output = []
  srt_temp_dir = os.path.join(tempfile.gettempdir(), "vgm-translate", 'srt')
  Path(srt_temp_dir).mkdir(parents=True, exist_ok=True)
  os.system(f"rm -rf {srt_temp_dir}/*")
  youtube_temp_dir = os.path.join(tempfile.gettempdir(), "vgm-translate", 'youtube')
  Path(youtube_temp_dir).mkdir(parents=True, exist_ok=True)
  os.system(f"rm -rf {youtube_temp_dir}/*")
  
  # path_inputs = [item.strip() for item in path_inputs.split(',')]
  # print("path_inputs::", path_inputs)
  # if path_inputs is not None and len(path_inputs) > 0 and path_inputs[0] != '':
  #   for media_path in path_inputs:
  #     media_path = media_path.strip()
  #     print("media_path::", media_path)
  #     if is_windows_path(media_path):
  #       window_path = PureWindowsPath(media_path)
  #       path_arr = [item for item in window_path.parts]
  #       path_arr[0] = re.sub(r'\:\\','',path_arr[0].lower())
  #       wsl_path = str(PurePosixPath('/mnt', *path_arr))
  #       print("wsl_path::", wsl_path)
  #       if os.path.exists(wsl_path):
  #         media_inputs.append(wsl_path)
  #       else:
  #         raise Exception(f"Path not exist:: {wsl_path}")
  #     else:
  #       if os.path.exists(media_path):
  #         media_inputs.append(media_path)
  #       else:
  #         raise Exception(f"Path not exist:: {media_path}")
          
  link_inputs = link_inputs.split(',')
  # print("link_inputs::", link_inputs)
  if link_inputs is not None and len(link_inputs) > 0 and link_inputs[0] != '':
    for url in link_inputs:
      url = url.strip()
      # print('testing url::', url.startswith( 'https://www.youtube.com' ))
      if url.startswith('https://www.youtube.com'):
        media_info = yt_dlp.YoutubeDL().extract_info(url, download=False)
        download_path = f"{os.path.join(youtube_temp_dir, media_info['title'])}.mp4"
        youtube_download(url, download_path)
        media_inputs.append(download_path) 
    
  if srt_inputs is not None and len(srt_inputs)> 0:
    for srt in srt_inputs:
      os.system(f"mv {srt.name} {srt_temp_dir}/")
  global total_input
  global total_output
  if media_inputs is not None and len(media_inputs)> 0:
    total_input = media_inputs
    for media in media_inputs:
      result = translate_from_media(media, s2t_method, t2t_method, t2s_method, vc_method, disable_timeline, YOUR_HF_TOKEN, preview, WHISPER_MODEL_SIZE, batch_size, chunk_size, compute_type, SOURCE_LANGUAGE, TRANSLATE_AUDIO_TO, min_speakers, max_speakers, tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05, AUDIO_MIX_METHOD, progress)
      total_output.append(result)
      output.append(result)
  return output

def upsampling(file):
  global upsampler
  if not upsampler:
    upsampler = Predictor()
    upsampler.setup(model_name="speech")
  filepath = os.path.join("audio2", file[0])
  print("upsampling:", filepath)
  audio_data, sample_rate = sf.read(filepath)
  source_duration = len(audio_data) / sample_rate
  data = upsampler.predict(
      filepath,
      ddim_steps=50,
      guidance_scale=3.5,
      seed=42
  )
  ## Trim duration to match source duration
  target_samples = int(source_duration * 48000)
  sf.write(filepath, data=data[:target_samples], samplerate=48000)
  return file

def tts(segment, speaker_to_voice, TRANSLATE_AUDIO_TO, t2s_method, disable_timeline):
    text = segment['text']
    start = segment['start']
    end = segment['end']

    try:
        speaker = segment['speaker']
        print("speaker::", speaker)
    except KeyError:
        segment['speaker'] = "SPEAKER_99"
        speaker = segment['speaker']
        print(f"NO SPEAKER DETECT IN SEGMENT: TTS auxiliary will be used in the segment time {segment['start'], segment['text']}")

    # make the tts audio
    filename = f"audio/{start}.wav"

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
    print("change speed::", porcentaje, duration_tts, duration_true)
    # Smooth and round
    porcentaje = math.floor(porcentaje * 10000) / 10000
    porcentaje = 0.8 if porcentaje <= 0.8 else porcentaje + 0.005
    porcentaje = 1.5 if porcentaje >= 1.5 else porcentaje
    porcentaje = 1.0 if disable_timeline else porcentaje     
    # apply aceleration or opposite to the audio file in audio2 folder
    os.system(f"ffmpeg -y -loglevel panic -i {filename} -filter:a atempo={porcentaje} audio2/{filename}")
    gc.collect(); torch.cuda.empty_cache()
    # duration_create = librosa.get_duration(filename=f"audio2/{filename}")
    return (filename, speaker) 
  
def translate_from_media(
    media_input,
    s2t_method,
    t2t_method,
    t2s_method,
    vc_method,
    disable_timeline,
    YOUR_HF_TOKEN,
    preview=False,
    WHISPER_MODEL_SIZE="large-v2",
    batch_size=16,
    chunk_size=5,
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
    file_name, file_extension = os.path.splitext(os.path.basename(media_input.strip().replace(' ','_')))
    mix_audio = os.path.join(temp_dir, f"{file_name}.mp3") 
    media_output_name = f"{file_name}-{TRANSLATE_AUDIO_TO}{file_extension}"
    media_output = os.path.join(temp_dir, media_output_name)
    source_media_output_basename = os.path.join(temp_dir, f'{file_name}-{SOURCE_LANGUAGE}')
    target_media_output_basename = os.path.join(temp_dir, f'{file_name}-{TRANSLATE_AUDIO_TO}') 
    audio_wav = f"{source_media_output_basename}.wav"
    audio_webm = f"{source_media_output_basename}.webm"
    translated_output_file = os.path.join(temp_dir, f"{target_media_output_basename}.wav")
    
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
    progress(0.30, desc="Speech to Text...")

    SOURCE_LANGUAGE = None if SOURCE_LANGUAGE == 'Automatic detection' else SOURCE_LANGUAGE

    # 1. Transcribe with original whisper (batched)
    print("Start transcribing source language::")
    with capture.capture_output() as cap:
      model = whisperx.load_model(
          WHISPER_MODEL_SIZE,
          device,
          compute_type=compute_type,
          language= SOURCE_LANGUAGE,
          )
      del cap
    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(WHISPER_MODEL_SIZE, audio, batch_size=batch_size, chunk_size=chunk_size)
    gc.collect(); torch.cuda.empty_cache(); del model
    print("Transcript complete::", len(result["segments"]))

    
    # # 2. Align whisper output for source language
    # print("Start aligning source language::")
    # progress(0.45, desc="Aligning source language...")
    # DAMHF.update(DAMT) #lang align
    # EXTRA_ALIGN = {
    #     "hi": "theainerd/Wav2Vec2-large-xlsr-hindi"
    # } # add new align models here
    # #print(result['language'], DAM.keys(), EXTRA_ALIGN.keys())
    # SOURCE_LANGUAGE = result['language']
    # if not result['language'] in DAMHF.keys() and not result['language'] in EXTRA_ALIGN.keys():
    #     audio = result = None
    #     print("Automatic detection: Source language not compatible")
    #     print(f"Detected language {result['language']}  incompatible, you can select the source language to avoid this error.")
    #     return
      
    ## Start aligning source language
    # model_a, metadata = whisperx.load_align_model(
    #     language_code=result["language"],
    #     device=device,
    #     model_name = None if result["language"] in DAMHF.keys() else EXTRA_ALIGN[result["language"]]
    #     )
    # print("whisperx align model loaded::")
    # result = whisperx.align(
    #     result["segments"],
    #     model_a,
    #     metadata,
    #     audio,
    #     device,
    #     return_char_alignments=True,
    #     )
    # print("Align source language complete::", result["segments"])
    # gc.collect(); torch.cuda.empty_cache(); del model_a

    if result['segments'] == []:
        print('No active speech found in audio')
        return

    # 3. Assign speaker labels
    print("Start Diarizing::")
    progress(0.50, desc="Diarizing...")
    if max_speakers > 1:
      with capture.capture_output() as cap:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
        del cap
      diarize_segments = diarize_model(
          audio_wav,
          min_speakers=min_speakers,
          max_speakers=max_speakers)
      result_diarize = whisperx.assign_word_speakers(diarize_segments, result)
      gc.collect(); torch.cuda.empty_cache(); del diarize_model
    else:
      result_diarize = result
      result_diarize['segments'] = [{**item, 'speaker': "SPEAKER_00"} for item in result_diarize['segments']]
    print("Diarize complete::")

    # 4. Translate to target language
    print("Start translating::")
    progress(0.6, desc="Translating...")
    if TRANSLATE_AUDIO_TO == "zh":
        TRANSLATE_AUDIO_TO = "zh-CN"
    if TRANSLATE_AUDIO_TO == "he":
        TRANSLATE_AUDIO_TO = "iw"
    # print("os.path.splitext(media_input)[0]::", os.path.splitext(media_input)[0])
    ## Write source segment and srt,txt to file

    with open(f'{source_media_output_basename}.json', 'a', encoding='utf-8') as srtFile:
      srtFile.write(json.dumps(result_diarize['segments']))
    segments_to_srt(result_diarize['segments'], f'{source_media_output_basename}.srt')
    result_diarize['segments'] = concise_srt(result_diarize['segments'], 375 if t2t_method == "LLM" else 500)
    segments_to_txt(result_diarize['segments'], f'{source_media_output_basename}.txt')
    # segments_to_srt(result_diarize['segments'], f'{media_output_basename}-{SOURCE_LANGUAGE}-concise.srt')
    target_srt_inputpath = os.path.join(tempfile.gettempdir(), "vgm-translate", 'srt', f'{file_name}-{TRANSLATE_AUDIO_TO}-SPEAKER.srt')
    if os.path.exists(target_srt_inputpath):
      # Start convert from srt if srt found
      print("srt file exist::", target_srt_inputpath)
      result_diarize['segments'] = srt_to_segments(target_srt_inputpath)
      result_diarize['segments'] = concise_srt(result_diarize['segments'])
    else:
      # Start translate if srt not found
      result_diarize['segments'] = translate_text(result_diarize['segments'], TRANSLATE_AUDIO_TO, t2t_method, user_settings['llm_url'],user_settings['llm_model'])
      print("translated segments::", result_diarize['segments'])
    ## Write target segment and srt to file
    segments_to_srt(result_diarize['segments'], f'{target_media_output_basename}.srt')
    with open(f'{target_media_output_basename}.json', 'a', encoding='utf-8') as srtFile:
      srtFile.write(json.dumps(result_diarize['segments']))
    # ## Sort segments by speaker
    # result_diarize['segments'] = sorted(result_diarize['segments'], key=lambda x: x['speaker'])
    print("Translation complete")

    # 5. TTS target language
    progress(0.7, desc="Text_to_speech...")
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
    
    N_JOBS = os.getenv('TTS_JOBS', round(CUDA_MEM*0.5/1000000000))
    print("Start TTS:: concurrency =", N_JOBS)
    with joblib.parallel_config(backend="loky", prefer="threads", n_jobs=int(N_JOBS) if max_speakers == 1 else 1):
      tts_results = Parallel(verbose=100)(delayed(tts)(segment, speaker_to_voice, TRANSLATE_AUDIO_TO, t2s_method, disable_timeline) for (segment) in tqdm(result_diarize['segments']))
    
    if os.getenv('UPSAMPLING_ENABLE', '') == "true":
      progress(0.75, desc="Upsampling...")
      print("Start Upsampling::")
      with joblib.parallel_config(backend="loky", prefer="threads", n_jobs=1):
        tts_results = Parallel(verbose=100)(delayed(upsampling)(file) for (file) in tts_results)
      global upsampler
      upsampler = None; gc.collect(); torch.cuda.empty_cache()
    # tts_results = []
    # for segment in tqdm(result_diarize['segments']):
    #   tts_result = tts(segment, speaker_to_voice, TRANSLATE_AUDIO_TO, t2s_method, disable_timeline)
    #   tts_results.append(tts_result)
      
    audio_files = [result[0] for result in tts_results]
    speakers_list = [result[1] for result in tts_results]
    print("audio_files:",len(audio_files))
    print("speakers_list:",len(speakers_list),speakers_list)
    
    # 6. Convert to target voices
    if vc_method == 'SVC':
        progress(0.80, desc="Applying SVC customized voices...")
        print("start SVC::")
        svc_voices(speakers_list, audio_files)
        
    if vc_method == 'RVC':
        progress(0.80, desc="Applying RVC customized voices...")
        print("start RVC::")
        rvc_voices(speakers_list, audio_files)

    # replace files with the accelerates
    os.system("mv -f audio2/audio/*.wav audio/")

    os.system(f"rm -rf {translated_output_file}")
    
    # 7. Join target language audio files
    progress(0.85, desc="Creating final translated media...")
    create_translated_audio(result_diarize, audio_files, translated_output_file, disable_timeline)

    # # 8. Transribe target language for smaller chunk
    # print("Start transcribing target language::")
    # progress(0.90, desc="Transcribing target language...")
    # with capture.capture_output() as cap:
    #   model = whisperx.load_model(
    #       WHISPER_MODEL_SIZE,
    #       device,
    #       compute_type=compute_type,
    #       language= SOURCE_LANGUAGE,
    #       )
    #   del cap
    # audio = whisperx.load_audio(translated_output_file)
    # result = model.transcribe(WHISPER_MODEL_SIZE, audio, batch_size=batch_size, chunk_size=chunk_size)
    # ## Write target segment and srt to file
    # segments_to_srt(result['segments'], f'{target_media_output_basename}.srt')
    # gc.collect(); torch.cuda.empty_cache(); del model
    # print("Transcribe target language complete::", len(result["segments"]),result["segments"])

    # 8. Combine final audio and video
    print("Mixing source and target voices::")
    progress(0.95, desc="Mixing final video...")
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

    print("Mixing target audio and video::")
    os.system(f"rm -rf {media_output}")
    if is_video:
      os.system(f"ffmpeg -i {OutputFile} -i {mix_audio} -c:v copy -c:a aac -map 0:v -map 1:a -shortest {media_output}")
    os.remove(OutputFile)
    if media_input.startswith('/tmp'):
      os.remove(media_input)
    ## Archve all files and return output
    archive_path = os.path.join(Path(temp_dir).parent.absolute(), os.path.splitext(os.path.basename(media_output))[0])
    shutil.make_archive(archive_path, 'zip', temp_dir)
    shutil.rmtree(temp_dir)
    final_output = f"{archive_path}.zip"
    
    ### Copy to temporary directory
    try:
      target_dir = os.getenv('COPY_OUTPUT_DIR', '')
      if target_dir and os.path.isdir(target_dir):
        os.system(f"cp '{final_output}' '{target_dir}'")
    except:
      print('copy to target dir failed')
    
  
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

js = """
function initLocalStorage() {
  if (localStorage.key("user_settings")) {
    try {
      const settings = JSON.parse(localStorage.getItem("user_settings"));
      for (const [key, value] of Object.entries(settings)) {
        document.getElementById(key).getElementsByTagName(value.type)[0].value =
          value.value;
        if (key === "vc") {
          const voiceNum = 6;
          for (let i = 0; i < voiceNum; i++) {
            const svcElem = document.getElementById(`svc_voice0${i}`);
            const rvcElem = document.getElementById(`rvc_voice0${i}`);
            console.log("svcElem:", svcElem, value.value);
            setTimeout(() => {
              svcElem.classList.toggle("hidden", value.value !== "SVC");
              rvcElem.classList.toggle("hidden", value.value !== "RVC");
            }, 500);
          }
        }
      }
    } catch (error) {
      console.log("error:", error);
    }
  }

  // Event listener for settings change
  document
    .getElementById("save_setting_btn")
    .addEventListener("click", function () {
      const settings = {
        s2t: {
          type: "input",
          value: document.getElementById("s2t").getElementsByTagName("input")[0]
            .value,
        },
        t2t: {
          type: "input",
          value: document.getElementById("t2t").getElementsByTagName("input")[0]
            .value,
        },
        t2s: {
          type: "input",
          value: document.getElementById("t2s").getElementsByTagName("input")[0]
            .value,
        },
        vc: {
          type: "input",
          value: document.getElementById("vc").getElementsByTagName("input")[0]
            .value,
        },
        llm_url: {
          type: "textarea",
          value: document
            .getElementById("llm_url")
            .getElementsByTagName("textarea")[0].value,
        },
        llm_model: {
          type: "input",
          value: document
            .getElementById("llm_model")
            .getElementsByTagName("input")[0].value,
        },
      };
      localStorage.setItem("user_settings", JSON.stringify(settings));
      console.log("setting saved!!", settings);
    });

  return "LocalStorage Initialized!!";
}
"""
theme="Taithrah/Minimal"
demo = gr.Blocks(title="VGM Translate",theme=theme, js=js)
with demo:
  gr.Markdown(title)
  gr.Markdown(description)
  with gr.Tabs():
    with gr.Tab("Audio Translation for a Video"):
        with gr.Row():
            with gr.Column():
                media_input = gr.Files(label="VIDEO|AUDIO", file_types=['audio','video'])
                # path_input = gr.Textbox(label="Import Windows Path",info="Example: M:\\warehouse\\video.mp4", placeholder="Windows path goes here, seperate by comma...")        
                link_input = gr.Textbox(label="Youtube Link",info="Example: https://www.youtube.com/watch?v=M2LksyGYPoc,https://www.youtube.com/watch?v=DrG2c1vxGwU", placeholder="URL goes here, seperate by comma...")        
                srt_input = gr.Files(label="SRT(Optional)", file_types=['.srt'])
                # gr.ClearButton(components=[media_input,link_input,srt_input], size='sm')
                disable_timeline = gr.Checkbox(label="Disable",container=False, info='Disable timeline matching with origin language?')
                ## media_input change function
                # link = gr.HTML()
                # media_input.change(submit_file_func, media_input, [media_input, link], show_progress='full')

                with gr.Row():
                  SOURCE_LANGUAGE = gr.Dropdown(['Automatic detection', 'Arabic (ar)', 'Cantonese (yue)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='English (en)',label = 'Source language', info="This is the original language of the video", scale=1)
                  TRANSLATE_AUDIO_TO = gr.Dropdown(['Arabic (ar)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='Vietnamese (vi)',label = 'Target language', info="Select the target language for translation", scale=1)

                # line_ = gr.HTML("<hr>")
                gr.Markdown("Select how many people are speaking in the video.")
                min_speakers = gr.Slider(1, MAX_TTS, value=1, step=1, label="min_speakers", visible=False)
                max_speakers = gr.Slider(1, MAX_TTS, value=1, step=1, label="Max speakers")
                gr.Markdown("Select the voice you want for each speaker.")
                def update_speaker_visibility(value):
                    visibility_dict = {
                        f'tts_voice{i:02d}_row': gr.update(visible=i < value) for i in range(6)
                    }
                    return [value for value in visibility_dict.values()]
                with gr.Row() as tts_voice00_row:
                  tts_voice00 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 1', visible=True)
                  svc_voice00 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='SVC Speaker 1', visible=False, elem_id="svc_voice00")
                  rvc_voice00 = gr.Dropdown(choices=list_rvc, value=list_rvc[0], label='RVC Speaker 1', visible=False, elem_id="rvc_voice00")
                with gr.Row(visible=False) as tts_voice01_row:
                  tts_voice01 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 2', visible=True)
                  svc_voice01 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='SVC Speaker 2', visible=False, elem_id="svc_voice01")
                  rvc_voice01 = gr.Dropdown(choices=list_rvc, value=list_rvc[0], label='RVC Speaker 2', visible=False, elem_id="rvc_voice01")
                with gr.Row(visible=False) as tts_voice02_row:
                  tts_voice02 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 3', visible=True)
                  svc_voice02 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='SVC Speaker 3', visible=False, elem_id="svc_voice02")
                  rvc_voice02 = gr.Dropdown(choices=list_rvc, value=list_rvc[0], label='RVC Speaker 3', visible=False, elem_id="rvc_voice02")
                with gr.Row(visible=False) as tts_voice03_row:
                  tts_voice03 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 4', visible=True)
                  svc_voice03 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='SVC Speaker 4', visible=False, elem_id="svc_voice03")
                  rvc_voice03 = gr.Dropdown(choices=list_rvc, value=list_rvc[0], label='RVC Speaker 4', visible=False, elem_id="rvc_voice03")
                with gr.Row(visible=False) as tts_voice04_row:
                  tts_voice04 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 5', visible=True)
                  svc_voice04 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='SVC Speaker 5', visible=False, elem_id="svc_voice04")
                  rvc_voice04 = gr.Dropdown(choices=list_rvc, value=list_rvc[0], label='RVC Speaker 5', visible=False, elem_id="rvc_voice04")
                with gr.Row(visible=False) as tts_voice05_row:
                  tts_voice05 = gr.Dropdown(choices=list_vtts, value=list_vtts[0], label='TTS Speaker 6', visible=True)
                  svc_voice05 = gr.Dropdown(choices=list_svc, value=list_svc[0], label='SVC Speaker 6', visible=False, elem_id="svc_voice05")
                  rvc_voice05 = gr.Dropdown(choices=list_rvc, value=list_rvc[0], label='RVC Speaker 6', visible=False, elem_id="rvc_voice05")
                max_speakers.change(update_speaker_visibility, max_speakers, [tts_voice00_row, tts_voice01_row, tts_voice02_row, tts_voice03_row, tts_voice04_row, tts_voice05_row])

                with gr.Column():
                      with gr.Accordion("Advanced Settings", open=False):

                          AUDIO_MIX = gr.Dropdown(['Mixing audio with sidechain compression', 'Adjusting volumes and mixing audio'], value='Adjusting volumes and mixing audio', label = 'Audio Mixing Method', info="Mix original and translated audio files to create a customized, balanced output with two available mixing modes.")

                          # gr.HTML("<hr>")
                          gr.Markdown("Default configuration of Whisper.")
                          with gr.Row():
                            WHISPER_MODEL_SIZE = gr.Dropdown(['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'], value=whisper_model_default, label="Whisper model",  scale=1)
                            compute_type = gr.Dropdown(list_compute_type, value=compute_type_default, label="Compute type",  scale=1)
                          with gr.Row():
                            batch_size = gr.Slider(1, 32, value=round(CUDA_MEM*0.65/1000000000), label="Batch size", step=1)
                            chunk_size = gr.Slider(2, 30, value=5, label="Chunk size", step=1)
                          # gr.HTML("<hr>")
                          # MEDIA_OUTPUT_NAME = gr.Textbox(label="Translated file name" ,value="media_output.mp4", info="The name of the output file")
                          PREVIEW = gr.Checkbox(label="Preview", visible=False, info="Preview cuts the video to only 10 seconds for testing purposes. Please deactivate it to retrieve the full video duration.")
                
                ## update_output_filename if media_input or TRANSLATE_AUDIO_TO change
                # def update_output_filename(file,lang):
                #     file_name, file_extension = os.path.splitext(os.path.basename(file.name.strip().replace(' ','_')))
                #     output_name = f"{file_name}-{LANGUAGES[lang]}{file_extension}"
                #     return gr.update(value=output_name)
                # media_input.change(update_output_filename, [media_input,TRANSLATE_AUDIO_TO], [MEDIA_OUTPUT_NAME])
                # TRANSLATE_AUDIO_TO.change(update_output_filename, [media_input,TRANSLATE_AUDIO_TO], [MEDIA_OUTPUT_NAME])
                
            with gr.Column(variant='compact'):
                def update_output_list():
                  global total_input
                  global total_output
                  return total_output if len(total_output) < len(total_input) else []
                with gr.Row():
                    video_button = gr.Button("TRANSLATE", )
                with gr.Row():    
                    media_output = gr.Files(label="PROGRESS BAR") #gr.Video()
                with gr.Row():
                    tmp_output = gr.Files(label="TRANSLATED VIDEO", every=10, value=update_output_list) #gr.Video()
                    
                ## Clear Button
                def reset_param():
                  global total_input
                  global total_output
                  total_input = []
                  total_output = []
                  return gr.update(label="PROGRESS BAR", visible=True), gr.update(label="TRANSLATED VIDEO", visible=True)
                clear_btn = gr.ClearButton(components=[media_input,link_input,srt_input,media_output,tmp_output], size='sm')
                clear_btn.click(reset_param,[],[media_output,tmp_output])
                line_ = gr.HTML("<hr>")
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
              s2t_method = gr.Dropdown(["Whisper"], label='S2T', value=user_settings['s2t'], visible=True, elem_id="s2t")
              t2t_method = gr.Dropdown(["Google", "VB", "T5", "LLM"], label='T2T', value=user_settings['t2t'], visible=True, elem_id="t2t")
              t2s_method = gr.Dropdown(["Google", "Edge", "VietTTS"], label='T2S', value=user_settings['t2s'], visible=True, elem_id="t2s")
              vc_method = gr.Dropdown(["None", "SVC", "RVC"], label='Voice Conversion', value=user_settings['vc'], visible=True, elem_id="vc")
            ## update t2s method
            def update_t2s_list(method):
              list_tts = list_vtts if method == 'VietTTS' else list_gtts
              visibility_dict = {
                  f'tts_voice{i:02d}': gr.update(choices=list_tts, value=list_tts[0]) for i in range(6)
              }
              return [value for value in visibility_dict.values()]
            t2s_method.change(update_t2s_list, [t2s_method], [tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05])
            
          ## Config LLM Settings
          def update_llm_model(llm_url):
            models = get_llm_models(llm_url)
            return gr.update(choices=models)
          with gr.Accordion("LLM Settings", open=True):
            with gr.Row():
              llm_url = gr.Textbox(label="LLM Endpoint", placeholder="LLM Endpoint goes here...", value=user_settings['llm_url'], elem_id="llm_url")
              llm_model = gr.Dropdown(label="LLM Model", choices=user_settings['llm_models'], value=user_settings['llm_model'], elem_id="llm_model")        
              llm_url.blur(update_llm_model, [llm_url], [llm_model])
          with gr.Row():
            def save_setting_fn(s2t_method,t2t_method,t2s_method,vc_method,llm_url,llm_model):
              # print("settings:", s2t_method,t2t_method,t2s_method,vc_method,llm_url,llm_model)
              user_settings['s2t'] = s2t_method
              user_settings['t2t'] = t2t_method
              user_settings['t2s'] = t2s_method
              user_settings['vc'] = vc_method
              user_settings['llm_url'] = llm_url
              user_settings['llm_models'] = get_llm_models(llm_url)
              user_settings['llm_model'] = llm_model
              save_settings(settings=user_settings)
              return gr.Info("Settings saved!!")
            save_setting_btn = gr.Button("Save Settings", elem_id="save_setting_btn")
            save_setting_btn.click(save_setting_fn, [s2t_method,t2t_method,t2s_method,vc_method,llm_url,llm_model], [])

        # with gr.Column():
        #   with gr.Accordion("Download RVC Models", open=False):
        #     url_links = gr.Textbox(label="URLs", value="",info="Automatically download the RVC models from the URL. You can use links from HuggingFace or Drive, and you can include several links, each one separated by a comma. Example: https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.pth, https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.index", placeholder="urls here...", lines=1)
        #     download_finish = gr.HTML()
        #     download_button = gr.Button("DOWNLOAD MODELS")

        #     def update_models():
        #       models, index_paths = upload_model_list()
        #       for i in range(8):                      
        #         dict_models = {
        #             f'model_voice_path{i:02d}': gr.update(choices=models) for i in range(8)
        #         }
        #         dict_index = {
        #             f'file_index2_{i:02d}': gr.update(choices=index_paths) for i in range(8)
        #         }
        #         dict_changes = {**dict_models, **dict_index}
        #         return [value for value in dict_changes.values()]

        # with gr.Column(visible=False) as svc_setting:
        #   with gr.Accordion("SVC Setting", open=False):
        #     with gr.Column(variant='compact'):
        #       with gr.Column():
        #         gr.Markdown("### 1. To enable its use, mark it as enable.")
        #         # enable_svc_custom_voice = gr.Checkbox(label="ENABLE", value=True, info="Check this to enable the use of the models.")
        #         # enable_svc_custom_voice.change(custom_rvc_model_voice_enable, [enable_svc_custom_voice], [])

        #         gr.Markdown("### 2. Select a voice that will be applied to each TTS of each corresponding speaker and apply the configurations.")
        #         gr.Markdown('Depending on how many "TTS Speaker" you will use, each one needs its respective model. Additionally, there is an auxiliary one if for some reason the speaker is not detected correctly.')
        #         gr.Markdown("Voice to apply to the first speaker.")
        #         with gr.Row():
        #           model_voice_path00 = gr.Dropdown(models, label = 'Model-1', visible=True)
        #           file_index2_00 = gr.Dropdown(index_paths, label = 'Index-1', visible=True)
        #           name_transpose00 = gr.Number(label = 'Transpose-1', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("Voice to apply to the second speaker.")
        #         with gr.Row():
        #           model_voice_path01 = gr.Dropdown(models, label='Model-2', visible=True)
        #           file_index2_01 = gr.Dropdown(index_paths, label='Index-2', visible=True)
        #           name_transpose01 = gr.Number(label='Transpose-2', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("Voice to apply to the third speaker.")
        #         with gr.Row():
        #           model_voice_path02 = gr.Dropdown(models, label='Model-3', visible=True)
        #           file_index2_02 = gr.Dropdown(index_paths, label='Index-3', visible=True)
        #           name_transpose02 = gr.Number(label='Transpose-3', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("Voice to apply to the fourth speaker.")
        #         with gr.Row():
        #           model_voice_path03 = gr.Dropdown(models, label='Model-4', visible=True)
        #           file_index2_03 = gr.Dropdown(index_paths, label='Index-4', visible=True)
        #           name_transpose03 = gr.Number(label='Transpose-4', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("Voice to apply to the fifth speaker.")
        #         with gr.Row():
        #           model_voice_path04 = gr.Dropdown(models, label='Model-5', visible=True)
        #           file_index2_04 = gr.Dropdown(index_paths, label='Index-5', visible=True)
        #           name_transpose04 = gr.Number(label='Transpose-5', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("Voice to apply to the sixth speaker.")
        #         with gr.Row():
        #           model_voice_path05 = gr.Dropdown(models, label='Model-6', visible=True)
        #           file_index2_05 = gr.Dropdown(index_paths, label='Index-6', visible=True)
        #           name_transpose05 = gr.Number(label='Transpose-6', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("- Voice to apply in case a speaker is not detected successfully.")
        #         with gr.Row():
        #           model_voice_path06 = gr.Dropdown(models, label='Model-Aux', visible=True)
        #           file_index2_06 = gr.Dropdown(index_paths, label='Index-Aux', visible=True)
        #           name_transpose06 = gr.Number(label='Transpose-Aux', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         with gr.Row():
        #           f0_method_global = gr.Dropdown(f0_methods_voice, value='harvest', label = 'Global F0 method', visible=True)

        #     with gr.Row(variant='compact'):
        #       button_config = gr.Button("APPLY CONFIGURATION")

        #       confirm_conf = gr.HTML()

        #     button_config.click(rvc_voices.apply_conf, inputs=[
        #         f0_method_global,
        #         model_voice_path00, name_transpose00, file_index2_00,
        #         model_voice_path01, name_transpose01, file_index2_01,
        #         model_voice_path02, name_transpose02, file_index2_02,
        #         model_voice_path03, name_transpose03, file_index2_03,
        #         model_voice_path04, name_transpose04, file_index2_04,
        #         model_voice_path05, name_transpose05, file_index2_05,
        #         model_voice_path06, name_transpose06, file_index2_06,
        #         ], outputs=[confirm_conf])

        # with gr.Column(visible=False) as rvc_setting:
        #   with gr.Accordion("RVC Setting", open=False):
        #     with gr.Column(variant='compact'):
        #       with gr.Column():
        #         # gr.Markdown("### 1. To enable its use, mark it as enable.")
        #         # enable_rvc_custom_voice = gr.Checkbox(label="ENABLE", info="Check this to enable the use of the models.")
        #         # enable_rvc_custom_voice.change(custom_rvc_model_voice_enable, [enable_rvc_custom_voice], [])

        #         gr.Markdown("### 1. Select a voice that will be applied to each TTS of each corresponding speaker and apply the configurations.")
        #         gr.Markdown('Depending on how many "TTS Speaker" you will use, each one needs its respective model. Additionally, there is an auxiliary one if for some reason the speaker is not detected correctly.')
        #         gr.Markdown("Voice to apply to the first speaker.")
        #         with gr.Row():
        #           model_voice_path00 = gr.Dropdown(models, label = 'Model-1', visible=True)
        #           file_index2_00 = gr.Dropdown(index_paths, label = 'Index-1', visible=True)
        #           name_transpose00 = gr.Number(label = 'Transpose-1', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("Voice to apply to the second speaker.")
        #         with gr.Row():
        #           model_voice_path01 = gr.Dropdown(models, label='Model-2', visible=True)
        #           file_index2_01 = gr.Dropdown(index_paths, label='Index-2', visible=True)
        #           name_transpose01 = gr.Number(label='Transpose-2', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("Voice to apply to the third speaker.")
        #         with gr.Row():
        #           model_voice_path02 = gr.Dropdown(models, label='Model-3', visible=True)
        #           file_index2_02 = gr.Dropdown(index_paths, label='Index-3', visible=True)
        #           name_transpose02 = gr.Number(label='Transpose-3', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("Voice to apply to the fourth speaker.")
        #         with gr.Row():
        #           model_voice_path03 = gr.Dropdown(models, label='Model-4', visible=True)
        #           file_index2_03 = gr.Dropdown(index_paths, label='Index-4', visible=True)
        #           name_transpose03 = gr.Number(label='Transpose-4', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("Voice to apply to the fifth speaker.")
        #         with gr.Row():
        #           model_voice_path04 = gr.Dropdown(models, label='Model-5', visible=True)
        #           file_index2_04 = gr.Dropdown(index_paths, label='Index-5', visible=True)
        #           name_transpose04 = gr.Number(label='Transpose-5', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("Voice to apply to the sixth speaker.")
        #         with gr.Row():
        #           model_voice_path05 = gr.Dropdown(models, label='Model-6', visible=True)
        #           file_index2_05 = gr.Dropdown(index_paths, label='Index-6', visible=True)
        #           name_transpose05 = gr.Number(label='Transpose-6', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         gr.Markdown("- Voice to apply in case a speaker is not detected successfully.")
        #         with gr.Row():
        #           model_voice_path06 = gr.Dropdown(models, label='Model-Aux', visible=True)
        #           file_index2_06 = gr.Dropdown(index_paths, label='Index-Aux', visible=True)
        #           name_transpose06 = gr.Number(label='Transpose-Aux', value=0, visible=True)
        #         gr.HTML("<hr></h2>")
        #         with gr.Row():
        #           f0_method_global = gr.Dropdown(f0_methods_voice, value='harvest', label = 'Global F0 method', visible=True)

        #     with gr.Row(variant='compact'):
        #       button_config = gr.Button("APPLY CONFIGURATION")

        #       confirm_conf = gr.HTML()

        #     button_config.click(rvc_voices.apply_conf, inputs=[
        #         f0_method_global,
        #         model_voice_path00, name_transpose00, file_index2_00,
        #         model_voice_path01, name_transpose01, file_index2_01,
        #         model_voice_path02, name_transpose02, file_index2_02,
        #         model_voice_path03, name_transpose03, file_index2_03,
        #         model_voice_path04, name_transpose04, file_index2_04,
        #         model_voice_path05, name_transpose05, file_index2_05,
        #         model_voice_path06, name_transpose06, file_index2_06,
        #         ], outputs=[confirm_conf])


          # with gr.Column():
          #       with gr.Accordion("Test RVC", open=False):

          #         with gr.Row(variant='compact'):
          #           text_test = gr.Textbox(label="Text", value="This is an example",info="write a text", placeholder="...", lines=5)
          #           with gr.Column(): 
          #             tts_test = gr.Dropdown(list_gtts, value='en-GB-ThomasNeural-Male', label = 'TTS', visible=True)
          #             model_voice_path07 = gr.Dropdown(models, label = 'Model', visible=True) #value=''
          #             file_index2_07 = gr.Dropdown(index_paths, label = 'Index', visible=True) #value=''
          #             transpose_test = gr.Number(label = 'Transpose', value=0, visible=True, info="integer, number of semitones, raise by an octave: 12, lower by an octave: -12")
          #             f0method_test = gr.Dropdown(f0_methods_voice, value='harvest', label = 'F0 method', visible=True) 
          #         with gr.Row(variant='compact'):
          #           button_test = gr.Button("Test audio")

          #         with gr.Column():
          #           with gr.Row():
          #             original_ttsvoice = gr.Audio()
          #             ttsvoice = gr.Audio()

          #           button_test.click(rvc_voices.make_test, inputs=[
          #               text_test,
          #               tts_test,
          #               model_voice_path07,
          #               file_index2_07,
          #               transpose_test,
          #               f0method_test,
          #               ], outputs=[ttsvoice, original_ttsvoice])

                # download_button.click(download_list, [url_links], [download_finish]).then(update_models, [], 
                #                   [
                #                     model_voice_path00, model_voice_path01, model_voice_path02, model_voice_path03, model_voice_path04, model_voice_path05, model_voice_path06, model_voice_path07,
                #                     file_index2_00, file_index2_01, file_index2_02, file_index2_03, file_index2_04, file_index2_05, file_index2_06, file_index2_07
                #                   ])
        ## Update VC method
        def update_voice_conversion(method):
          svc_visibility_dict = {
              f'svc_voice{i:02d}_row': gr.update(visible=method=='SVC') for i in range(6)
          }
          rvc_visibility_dict = {
              f'rvc_voice{i:02d}_row': gr.update(visible=method=='RVC') for i in range(6)
          }
          return list(svc_visibility_dict.values()) + list(rvc_visibility_dict.values())
        vc_method.change(update_voice_conversion, [vc_method], [svc_voice00,svc_voice01,svc_voice02,svc_voice03,svc_voice04,svc_voice05,rvc_voice00,rvc_voice01,rvc_voice02,rvc_voice03,rvc_voice04,rvc_voice05])


    with gr.Tab("Help"):
        gr.Markdown(tutorial)
        # gr.Markdown(news)

    # with gr.Accordion("Logs", open = False):
    #     logs = gr.Textbox()
    #     demo.load(read_logs, None, logs, every=1)
    def update_output_visibility():
      return gr.update(label="TRANSLATED VIDEO"),gr.update(visible=False)
    # run
    video_button.click(batch_preprocess, inputs=[
        media_input,
        # path_input,
        link_input,
        srt_input,
        s2t_method,
        t2t_method,
        t2s_method,
        vc_method,
        disable_timeline,
        HFKEY,
        PREVIEW,
        WHISPER_MODEL_SIZE,
        batch_size,
        chunk_size,
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
        rvc_voice00,
        rvc_voice01,
        rvc_voice02,
        rvc_voice03,
        rvc_voice04,
        rvc_voice05,
        svc_voice00,
        svc_voice01,
        svc_voice02,
        svc_voice03,
        svc_voice04,
        svc_voice05,
        AUDIO_MIX,
        ], outputs=media_output, concurrency_limit=1).then(
        #time.sleep(2),
        fn=update_output_visibility,
        inputs=[],
        outputs=[media_output,tmp_output]
        )

if __name__ == "__main__":
  mp.set_start_method('spawn', force=True)
  
  # os.system('rm -rf /tmp/gradio/*')
  # os.system('rm -rf *.wav *.mp3 *.wav *.mp4')
  os.system('mkdir -p downloads')
  os.system(f'rm -rf {os.path.join(tempfile.gettempdir(), "vgm-translate")}/*')
  
  os.system(f'rm -rf audio2/SPEAKER_* audio2/audio/* audio.out audio/*')
  print("CUDA_MEM::", CUDA_MEM)
  print('Working in:: ', device)
  
  auth_user = os.getenv('AUTH_USER', '')
  auth_pass = os.getenv('AUTH_PASS', '')
  demo.queue().launch(
    auth=(auth_user, auth_pass) if auth_user != '' and auth_pass != '' else None,
    show_api=False,
    debug=True,
    inbrowser=True,
    show_error=True,
    server_name="0.0.0.0",
    server_port=6860,
    # quiet=True,
    share=False)