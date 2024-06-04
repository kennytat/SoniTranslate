from .logging_setup import logger
import edge_tts
import asyncio
from .utils import download_manager
from pathlib import Path
import numpy as np
import platform
from typing import Any, Dict
from tqdm import tqdm
import librosa, os, re, torch, gc, subprocess, json
import soundfile as sf

class TTS_OperationError(Exception):
    def __init__(self, message="The operation did not complete successfully."):
        self.message = message
        super().__init__(self.message)

def edge_tts_voices_list():
    voices = []
    formatted_voices = []
    try:
        try:
          completed_process = subprocess.run(
              ["edge-tts", "--list-voices"], capture_output=True, text=True
          )
          lines = completed_process.stdout.strip().split("\n")
        except:
          lines = []

        for line in lines:
            if line.startswith("Name: "):
                voice_entry = {}
                voice_entry["Name"] = line.split(": ")[1]
            elif line.startswith("Gender: "):
                voice_entry["Gender"] = line.split(": ")[1]
                voices.append(voice_entry)

        formatted_voices = [
            f"{entry['Name']}-{entry['Gender']}" for entry in voices
        ]
      
        if not formatted_voices:
            logger.warning(
                "The list of Edge TTS voices could not be obtained, "
                "switching to an alternative method"
            )
            tts_voice_list = asyncio.new_event_loop().run_until_complete(
                edge_tts.list_voices()
            )
            formatted_voices = sorted(
                [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]
            )

        if not formatted_voices:
            logger.error("Can't get EDGE TTS - list voices")
    except Exception as error:
        logger.debug(str(error))
    return formatted_voices

  
# =====================================
# PIPER TTS
# =====================================


def piper_tts_voices_list():
    file_path = download_manager(
        url="https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json",
        path="./model/piperTTS",
    )

    with open(file_path, "r", encoding="utf8") as file:
        data = json.load(file)
    piper_id_models = [key for key in data.keys()]

    return piper_id_models


def replace_text_in_json(file_path, key_to_replace, new_text, condition=None):
    # Read the JSON file
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Modify the specified key's value with the new text
    if key_to_replace in data:
        if condition:
            value_condition = condition
        else:
            value_condition = data[key_to_replace]

        if data[key_to_replace] == value_condition:
            data[key_to_replace] = new_text

    # Write the modified content back to the JSON file
    with open(file_path, "w") as file:
        json.dump(
            data, file, indent=2
        )  # Write the modified data back to the file with indentation for readability


def load_piper_model(
    model: str,
    data_dir: list,
    download_dir: str = "",
    update_voices: bool = False,
):
    from piper import PiperVoice
    from piper.download import ensure_voice_exists, find_voice, get_voices

    try:
        import onnxruntime as rt

        if rt.get_device() == "GPU" and os.environ.get("SONITR_DEVICE") == "cuda":
            logger.debug("onnxruntime device > GPU")
            cuda = True
        else:
            logger.info(
                "onnxruntime device > CPU"
            )  # try pip install onnxruntime-gpu
            cuda = False
    except Exception as error:
        raise TTS_OperationError(f"onnxruntime error: {str(error)}")

    # Disable CUDA in Windows
    if platform.system() == "Windows":
        logger.info("Employing CPU exclusivity with Piper TTS")
        cuda = False

    if not download_dir:
        # Download to first data directory by default
        download_dir = data_dir[0]
    else:
        data_dir = [os.path.join(data_dir[0], download_dir)]

    # Download voice if file doesn't exist
    model_path = Path(model)
    if not model_path.exists():
        # Load voice info
        voices_info = get_voices(download_dir, update_voices=update_voices)

        # Resolve aliases for backwards compatibility with old voice names
        aliases_info: Dict[str, Any] = {}
        for voice_info in voices_info.values():
            for voice_alias in voice_info.get("aliases", []):
                aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

        voices_info.update(aliases_info)
        ensure_voice_exists(model, data_dir, download_dir, voices_info)
        model, config = find_voice(model, data_dir)

        replace_text_in_json(
            config, "phoneme_type", "espeak", "PhonemeType.ESPEAK"
        )

    # Load voice
    voice = PiperVoice.load(model, config_path=config, use_cuda=cuda)

    return voice


def synthesize_text_to_audio_np_array(voice, text, synthesize_args):
    audio_stream = voice.synthesize_stream_raw(text, **synthesize_args)

    # Collect the audio bytes into a single NumPy array
    audio_data = b""
    for audio_bytes in audio_stream:
        audio_data += audio_bytes

    # Ensure correct data type and convert audio bytes to NumPy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    return audio_np


def piper_tts(tts_text, tts_voice, tts_speed, filename):
    """
    Install:
    pip install -q piper-tts==1.2.0 onnxruntime-gpu # for cuda118
    """
	
    data_dir = [
        str(Path.cwd())
    ]  # "Data directory to check for downloaded models (default: current directory)"
    download_dir = os.path.join("model", "piperTTS")
    # model_name = "en_US-lessac-medium" tts_name in a dict like VITS
    update_voices = True  # "Download latest voices.json during startup",

    synthesize_args = {
        "speaker_id": None,
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w": 0.8,
        "sentence_silence": 0.0,
    }

    text = tts_text
    tts_name = tts_voice.replace(" VITS-onnx", "")
    print("piper_tts called::", tts_name, tts_speed, filename)
    model = load_piper_model(
        tts_name, data_dir, download_dir, update_voices
    )
    sampling_rate = model.config.sample_rate

    try:
        # Infer
        speech_output = synthesize_text_to_audio_np_array(
            model, text, synthesize_args
        )

        # Save file
        sf.write(
            file=filename,
            samplerate=sampling_rate,
            data=speech_output,
            format="ogg",
            subtype="vorbis",
        )

    except Exception as error:
        print("Error piperTTS::", error, tts_text)
    gc.collect()
    torch.cuda.empty_cache()
    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        logger.error(str(error))
        gc.collect()
        torch.cuda.empty_cache()

# try:
#   piper_tts("Có lẽ loại trợ giúp đọc Kinh Thánh thông thường nhất là loại dự phần hằng ngày, hằng năm và trong phần hướng dẫn ngắn này tôi muốn chỉ cho bạn thấy cách thức, thật dễ dàng như thế nào để có được phần dự phần hằng ngày của bạn, sử dụng bảng dữ liệu mà chúng tôi thiết lập trong một hướng dẫn trước đây, hãy nhìn vào màn hình của tôi, bạn sẽ thấy tôi đã mở ra trang chủ rồi, nếu trang chủ của bạn chưa được mở ra", "vi_VN-25hours_single-low", 1, "audio/107.927.wav")
# except Exception as error:
# 	print('An exception occurred::', error)