from .logging_setup import logger
import edge_tts
import asyncio
from .utils import download_manager
from pathlib import Path
import numpy as np
import platform
import threading
from typing import Any, Dict
from tqdm import tqdm
import librosa, os, re, torch, gc, subprocess, json

# Piper voice sessions are expensive to build (ONNX session + voice file read).
# Cache them per voice name — synthesize calls are cheap afterwards and
# onnxruntime sessions are safe to share across threads for sequential use.
_piper_voice_cache: Dict[str, Any] = {}
_piper_cache_lock = threading.Lock()


def get_cached_piper_voice(model_name, data_dir, download_dir, update_voices=True):
    key = f"{model_name}::{download_dir}"
    with _piper_cache_lock:
        voice = _piper_voice_cache.get(key)
        if voice is None:
            voice = load_piper_model(
                model_name, data_dir, download_dir, update_voices
            )
            _piper_voice_cache[key] = voice
        return voice
import soundfile as sf
from pydub import AudioSegment

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
    """Load a Piper voice for piper-tts >= 1.4 (new module layout, no piper.download)."""
    from piper import PiperVoice

    # Disable CUDA in Windows (untested path)
    cuda = (
        platform.system() != "Windows"
        and os.environ.get("SONITR_DEVICE") == "cuda"
    )
    logger.info(f"onnxruntime device > {'GPU' if cuda else 'CPU'}")

    if not download_dir:
        download_dir = data_dir[0]

    model_path = Path(os.path.join(download_dir, f"{model}.onnx"))
    config_path = Path(str(model_path) + ".json")

    if not model_path.exists():
        from piper.download_voices import download_voice
        download_voice(model, Path(download_dir))

    # Keep eager-phonemizer compatibility if the config declares the old type
    try:
        replace_text_in_json(
            config_path, "phoneme_type", "espeak", "PhonemeType.ESPEAK"
        )
    except Exception as error:
        logger.debug(f"piper phoneme_type patch skipped: {error}")

    voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=cuda)

    return voice


def synthesize_text_to_audio_np_array(voice, text, synthesize_args):
    """piper-tts >= 1.4: build int16 numpy audio via SynthesisConfig."""
    from piper import SynthesisConfig

    syn_config = SynthesisConfig(
        speaker_id=synthesize_args.get("speaker_id"),
        length_scale=synthesize_args.get("length_scale", 1.0),
        noise_scale=synthesize_args.get("noise_scale", 0.667),
        noise_w_scale=synthesize_args.get("noise_w", 0.8),
    )
    chunks = list(voice.synthesize(text, syn_config=syn_config))
    if not chunks:
        raise RuntimeError("piper returned no audio")
    audio_np = np.concatenate([c.audio_int16_array for c in chunks])
    return audio_np, chunks[0].sample_rate


def piper_tts(tts_text, tts_voice, tts_speed):
    """
    piper-tts >= 1.4 API. Voice sessions are cached per voice name
    (see get_cached_piper_voice).
    """
    # model_name = "en_US-lessac-medium" tts_name in a dict like VITS
    download_dir = os.path.join("model", "piperTTS")

    text = tts_text
    tts_name = tts_voice.replace(" VITS-onnx", "")
    print("piper_tts called::", tts_name, tts_speed)
    model = get_cached_piper_voice(tts_name, [str(Path.cwd())], download_dir)

    # Speed: piper uses length_scale (duration factor) -> scale = 1/speed
    synthesize_args = {
        "speaker_id": None,
        "length_scale": 1.0 / max(tts_speed, 0.1),
        "noise_scale": 0.667,
        "noise_w": 0.8,
        "sentence_silence": 0.0,
    }

    try:
        audio_np, sampling_rate = synthesize_text_to_audio_np_array(
            model, text, synthesize_args
        )
    except Exception as error:
        print("Error piperTTS::", error, tts_text)
        raise
    gc.collect()
    return AudioSegment(
          data=audio_np.tobytes(),
          sample_width=2,
          frame_rate=sampling_rate,
          channels=1
      )


# try:
#   piper_tts("Có lẽ loại trợ giúp đọc Kinh Thánh thông thường nhất là loại dự phần hằng ngày, hằng năm và trong phần hướng dẫn ngắn này tôi muốn chỉ cho bạn thấy cách thức, thật dễ dàng như thế nào để có được phần dự phần hằng ngày của bạn, sử dụng bảng dữ liệu mà chúng tôi thiết lập trong một hướng dẫn trước đây, hãy nhìn vào màn hình của tôi, bạn sẽ thấy tôi đã mở ra trang chủ rồi, nếu trang chủ của bạn chưa được mở ra", "vi_VN-25hours_single-low", 1, "audio/107.927.wav")
# except Exception as error:
# 	print('An exception occurred::', error)