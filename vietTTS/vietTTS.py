import os
import torch  # isort:skip
torch.manual_seed(42)
import soundfile as sf
import json
import re
import unicodedata
from types import SimpleNamespace
from pydub import AudioSegment
from langdetect import detect
import gc
import numpy as np
import regex
from vietTTS.models import DurationNet, SynthesizerTrn
from vietTTS.utils import normalize, num_to_str, read_number, replace_dict

TTS_MODEL_DIR = os.path.join(os.getcwd(),"model","vits")

device = "cuda" if torch.cuda.is_available() else "cpu"
space_re = regex.compile(r"\s+")
number_re = regex.compile("([0-9]+)")
num_re = regex.compile(r"([0-9.,]*[0-9])")
alphabet = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵbcdđghklmnpqrstvx"
keep_text_and_num_re = regex.compile(rf"[^\s{alphabet}.,0-9]")
keep_text_re = regex.compile(rf"[^\s{alphabet}]")
voice_data = {}

def text_to_phone_idx(text, phone_set, sil_idx):
    # lowercase
    text = text.lower()
    # unicode normalize
    text = normalize(text)
    text = unicodedata.normalize("NFKC", text)
    text = num_to_str(text)
    text = re.sub(r"[\s\.]+(?=\s)", " . ", text)
    text = text.replace(".", " . ")
    text = text.replace("-", " - ")
    text = text.replace(",", " , ")
    text = text.replace(";", " ; ")
    text = text.replace(":", " : ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace("(", " ( ")
    text = num_re.sub(r" \1 ", text)
    words = text.split()
    words = [read_number(w) if num_re.fullmatch(w) else w for w in words]
    text = " ".join(words)

    # remove redundant spaces
    text = re.sub(r"\s+", " ", text)
    # remove leading and trailing spaces
    text = text.strip()
    # convert words to phone indices
    tokens = []
    for c in text:
        # if c is "," or ".", add <sil> phone
        if c in ":,.!?;(":
            tokens.append(sil_idx)
        elif c in phone_set:
            tokens.append(phone_set.index(c))
        elif c == " ":
            # add <sep> phone
            tokens.append(0)
    if tokens[0] != sil_idx:
        # insert <sil> phone at the beginning
        tokens = [sil_idx, 0] + tokens
    if tokens[-1] != sil_idx:
        tokens = tokens + [0, sil_idx]
    return tokens

def inference(duration_net, generator, text, phone_set, hps, speed, max_word_length=750):
    assert phone_set[0][1:-1] == "SEP"
    assert "sil" in phone_set
    sil_idx = phone_set.index("sil")
    # prevent too long text
    if len(text) > max_word_length:
        text = text[:max_word_length]

    phone_idx = text_to_phone_idx(text, phone_set, sil_idx)
    batch = {
        "phone_idx": np.array([phone_idx]),
        "phone_length": np.array([len(phone_idx)]),
    }

    # predict phoneme duration
    phone_length = torch.from_numpy(batch["phone_length"].copy()).long().to(device)
    phone_idx = torch.from_numpy(batch["phone_idx"].copy()).long().to(device)
    with torch.inference_mode():
        phone_duration = duration_net(phone_idx, phone_length)[:, :, 0] * 1000 / speed
    phone_duration = torch.where(
        phone_idx == sil_idx, torch.clamp_min(phone_duration, 200), phone_duration
    )
    phone_duration = torch.where(phone_idx == 0, 0, phone_duration)

    # generate waveform
    end_time = torch.cumsum(phone_duration, dim=-1)
    start_time = end_time - phone_duration
    start_frame = start_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
    end_frame = end_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
    spec_length = end_frame.max(dim=-1).values
    pos = torch.arange(0, spec_length.item(), device=device)
    attn = torch.logical_and(
        pos[None, :, None] >= start_frame[:, None, :],
        pos[None, :, None] < end_frame[:, None, :],
    ).float()
    with torch.inference_mode():
        y_hat = generator.infer(
            phone_idx, phone_length, spec_length, attn, max_len=None, noise_scale=0.667
        )[0]
    wave = y_hat[0, 0].data.cpu().numpy()
    return (wave * (2**15)).astype(np.int16)

def load_models(model_path, hps):
    duration_model_path=os.path.join(model_path,"duration.pth")
    lightspeed_model_path = os.path.join(model_path,"vits.pth")
    duration_net = DurationNet(hps.data.vocab_size, 64, 4).to(device)
    duration_net.load_state_dict(torch.load(duration_model_path, map_location=device))
    duration_net = duration_net.eval()
    generator = SynthesizerTrn(
        hps.data.vocab_size,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **vars(hps.model),
    ).to(device)
    del generator.enc_q
    ckpt = torch.load(lightspeed_model_path, map_location=device)
    params = {}
    for k, v in ckpt["net_g"].items():
        k = k[7:] if k.startswith("module.") else k
        params[k] = v
    generator.load_state_dict(params, strict=False)
    del ckpt, params
    generator = generator.eval()
    return duration_net, generator
        
def text_to_speech(text, output_file, model_name,speed = 1):
    global voice_data
    if model_name not in voice_data:
      voice_data[model_name] = {"config": None, "phone_set": None}
    tts_voice_ckpt_dir = os.path.join(TTS_MODEL_DIR, model_name)
    print("Starting TTS {}".format(output_file))
    ### load hifigan config
    config_file = os.path.join(tts_voice_ckpt_dir,"config.json")
    if not voice_data[model_name]["config"]:
      with open(config_file, "rb") as f:
        voice_data[model_name]["config"] = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
    sample_rate = voice_data[model_name]["config"].data.sampling_rate
    ### load phoneset
    phone_set_file = os.path.join(tts_voice_ckpt_dir,"phone_set.json")
    if not voice_data[model_name]["phone_set"]:
      with open(phone_set_file, "r") as f:
        voice_data[model_name]["phone_set"] = json.load(f)
        
    print("tts text::", text)
    if re.sub(r'^sil\s+','',text).isnumeric():
        silence_duration = int(re.sub(r'^sil\s+','',text)) * 1000
        print("Got integer::", text, silence_duration) 
        print("\n\n\n ==> Generating {} seconds of silence at {}".format(silence_duration, output_file))
        second_of_silence = AudioSegment.silent(duration=silence_duration) # or be explicit
        second_of_silence = second_of_silence.set_frame_rate(sample_rate)
        second_of_silence.export(output_file, format="wav")
    elif text == "♪":
        second_of_silence = AudioSegment.silent(duration=2000) # or be explicit
        second_of_silence = second_of_silence.set_frame_rate(sample_rate)
        second_of_silence.export(output_file, format="wav")
    else:
      duration_net, generator = load_models(tts_voice_ckpt_dir, voice_data[model_name]["config"])
      text = text if detect(text) == 'vi' else ' . '
      text = replace_dict(text)
      tts_result = inference(duration_net, generator, text, voice_data[model_name]["phone_set"], voice_data[model_name]["config"], speed)
      wav = np.concatenate([tts_result])
      # Equalize and Normalize
      wav = wav / 32768.0  # Convert to range [-1, 1]
      # Apply a simple high-pass filter for equalization
      # This is a very basic approach - for a more complex equalization, more sophisticated filtering would be required
      # Boosting higher frequencies
      alpha = 0.8
      filtered_data = np.array(wav)
      for i in range(1, len(wav)):
          filtered_data[i] = alpha * filtered_data[i] + (1 - alpha) * wav[i]
      # Normalize the audio
      max_val = np.max(np.abs(filtered_data))
      wav = filtered_data / max_val
      wav = (wav * 32767).astype(np.int16)
      ## Write wav to file        
      sf.write(output_file, wav, samplerate=sample_rate)
      print("Wav segment written at: {}".format(output_file))
    gc.collect(); torch.cuda.empty_cache(); del duration_net; del generator
    return "Done"
  
# if __name__ == '__main__':
#   model_name = "vn_han_male"
#   raw_str = """Và trong tài liệu hướng dẫn ngắn này, tôi muốn cho bạn thấy nó dễ dàng như thế nào để có được yêu thích hàng ngày của bạn sử dụng các thẻ bảng điều khiển chúng tôi thiết lập trong một hướng dẫn trước đó."""
#   result = text_to_speech(raw_str, "/mnt/ssd256/Projects/SoniTranslate/test/viettts.wav", model_name)
#   # print("result::", result)
