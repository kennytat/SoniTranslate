from dotenv import load_dotenv
import os
import sys
import gc
from pathlib import Path
import atexit
from datetime import datetime
import argparse
import shutil
import tempfile
import librosa
import math
import torch  # isort:skip
import torch.multiprocessing as mp
torch.manual_seed(42)
import soundfile as sf
import json
import re
import unicodedata
from types import SimpleNamespace
import joblib
from joblib import Parallel, delayed
from pydub import AudioSegment
from langdetect import detect
from queue import Queue
import gradio as gr
import numpy as np
import regex
from vietTTS.models import DurationNet, SynthesizerTrn
from vietTTS.utils import normalize, num_to_str, read_number, pad_zero, encode_filename, new_dir_now, file_to_paragraph, txt_to_paragraph, combine_wav_segment
from vietTTS.upsample import Predictor
load_dotenv()


## Exit Hooks called when app terminating
class ExitHooks(object):
    def __init__(self):
        self.exit_code = None
        self.exception = None

    def hook(self):
        self._orig_exit = sys.exit
        sys.exit = self.exit
        sys.excepthook = self.exc_handler

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args):
        self.exception = exc

hooks = ExitHooks()
hooks.hook()

class WavStruct():
    def __init__(self, wav_path, start_time):
        self.wav_path = wav_path
        self.start_time = start_time
        
class CONFIG():
    """Configurations"""
    # ckpt
    os_tmp = Path(os.path.join(tempfile.gettempdir(), "tts"))
    empty_wav = Path(os.path.join(f'{os_tmp}', "test.wav"))
    tts_ckpt_dir = Path(os.path.join(os.getcwd(), "model", "vits"))
    convert_ckpt_dir = Path(os.path.join(os.getcwd(), "model", "svc"))
    # salt = Path(os.path.join(os.getcwd(), "model","tts", "salt.salt"))
    key = "^VGMAI*607#"


device = "cuda" if torch.cuda.is_available() else "cpu"
space_re = regex.compile(r"\s+")
number_re = regex.compile("([0-9]+)")
num_re = regex.compile(r"([0-9.,]*[0-9])")
alphabet = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵbcdđghklmnpqrstvx"
keep_text_and_num_re = regex.compile(rf"[^\s{alphabet}.,0-9]")
keep_text_re = regex.compile(rf"[^\s{alphabet}]")

class TTS():
  def __init__(self):
    self.device = device
    self.upsampler = None
        
  def text_to_phone_idx(self, text, phone_set, sil_idx):
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


  def text_to_speech(self, duration_net, generator, text, model_path, hps, speed, max_word_length=750):
      phone_set_file = os.path.join(model_path,"phone_set.json")
      # load phone set json file
      with open(phone_set_file, "r") as f:
          phone_set = json.load(f)

      assert phone_set[0][1:-1] == "SEP"
      assert "sil" in phone_set
      sil_idx = phone_set.index("sil")
      # prevent too long text
      if len(text) > max_word_length:
          text = text[:max_word_length]

      phone_idx = self.text_to_phone_idx(text, phone_set, sil_idx)
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
      del phone_duration; del duration_net; del generator; gc.collect(); torch.cuda.empty_cache()
      return (wave * (2**15)).astype(np.int16)


  def load_models(self, model_path, hps):
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

          
  def tts(self, text, output_file, tts_voice_ckpt_dir, speed = 1, desired_duration = 0, start_time = 0):
      try:
        print("Starting TTS {}".format(output_file), desired_duration, start_time)
        ### Get hifigan path
        config_file = os.path.join(tts_voice_ckpt_dir,"config.json")
        with open(config_file, "rb") as f:
          hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
        sample_rate = hps.data.sampling_rate
        print("tts text::", text)

        if re.sub(r'^sil\s+','',text).isnumeric():
            silence_duration = int(re.sub(r'^sil\s+','',text)) * 1000
            print("Got integer::", text, silence_duration) 
            print("\n\n\n ==> Generating {} seconds of silence at {}".format(silence_duration, output_file))
            second_of_silence = AudioSegment.silent(duration=silence_duration) # or be explicit
            second_of_silence = second_of_silence.set_frame_rate(sample_rate)
            second_of_silence.export(output_file, format="wav")
        else:
          duration_net, generator = self.load_models(tts_voice_ckpt_dir, hps)
          text = text if detect(text) == 'vi' else ' . '
          ## For tts with timeline
          if desired_duration > 0:
            tts_tmp_result = self.text_to_speech(duration_net, generator, text, tts_voice_ckpt_dir, hps, 1)
            wav_tmp = np.concatenate([tts_tmp_result])
            predicted_duration = librosa.get_duration(y=wav_tmp, sr=sample_rate)
            speed = predicted_duration / desired_duration
            speed = math.floor(speed * 10000) / 10000
            speed = 0.8 if speed <= 0.8 else speed + 0.005
            speed = 1.5 if speed >= 1.5 else speed
            print("tts speed::",speed)
            tts_result = self.text_to_speech(duration_net, generator, text, tts_voice_ckpt_dir, hps, speed)
            wav = np.concatenate([tts_result])
          else:
            tts_result = self.text_to_speech(duration_net, generator, text, tts_voice_ckpt_dir, hps, speed)
            # clips.append(silence)
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

          # Save the processed file
          sf.write(output_file, wav, samplerate=sample_rate)
          print("Wav segment written at: {}".format(output_file))
        del duration_net; del generator; gc.collect(); torch.cuda.empty_cache()
      except Exception as error:
        print("tts error::", text, "\n", error)
      return WavStruct(output_file, start_time)

  def upsampling(self, file):
    if not self.upsampler:
      self.upsampler = Predictor()
      self.upsampler.setup(model_name="speech")
    audio_data, sample_rate = sf.read(file.wav_path)
    source_duration = len(audio_data) / sample_rate
    data = self.upsampler.predict(
        file.wav_path,
        ddim_steps=50,
        guidance_scale=3.5,
        seed=42
    )
    ## Trim duration to match source duration
    target_samples = int(source_duration * 48000)
    sf.write(file.wav_path, data=data[:target_samples], samplerate=48000)
    return file
    
  def convert_voice(self, input_dir, model_dir):
    print("start convert_voice::", input_dir, model_dir)
    model_path = os.path.join(model_dir, "G.pth")
    config_path = os.path.join(model_dir, "config.json")
    output_dir = f'{input_dir}.out'
    os.system(f'svc infer -re -m {model_path} -c {config_path} {input_dir}')
    if os.path.exists(input_dir): shutil.rmtree(input_dir, ignore_errors=True)
    shutil.move(output_dir, input_dir)
    gc.collect(); torch.cuda.empty_cache()
    
  def synthesize(self, output_dir_name, input, is_file, speed, method, tts_voice_ckpt_dir, convert_voice_ckpt_dir):
      print("start synthesizing::", output_dir_name, input, is_file, speed, tts_voice_ckpt_dir, convert_voice_ckpt_dir)
      filepath = ""
      paragraphs = ""
      file_name_only = ""
      basename, ext = os.path.splitext(os.path.basename(input))
      print(basename, ext)
      if is_file:
        file_name_only = Path(basename)
        filepath = encode_filename(input)
        paragraphs = file_to_paragraph(input)
      else:
        filepath = "{}".format(new_dir_now())
        file_name_only = encode_filename(filepath)
        paragraphs = txt_to_paragraph(input)

      ### Put segments in temp dir for concatnation later
      tmp_dirname = os.path.join(CONFIG.os_tmp, output_dir_name, filepath)
      # print("filename::", filepath, paragraphs, file_name_only, tmp_dirname)
      Path(tmp_dirname).mkdir(parents=True, exist_ok=True)
      final_name = "{}.wav".format(file_name_only)
      final_output = os.path.join(CONFIG.os_tmp, output_dir_name, final_name)
      log_output = None
      print("Output Temp: ", final_output)
      temp_output = ''

      # process_list = []
      queue_list = Queue()
      # wav_list = []
      results = []
      for (no, para) in enumerate(paragraphs):
          name = "{}.wav".format(pad_zero(no, 5))
          print("Prepare normalized text: ", para.text)
          temp_output = os.path.join(tmp_dirname, name)
          print("paragraph:: ", para.text.strip(), temp_output, para.total_duration, para.start_time)
          queue_list.put((para.text.strip(), temp_output, para.total_duration, para.start_time))
          
      # print("Parallel processing {} tasks".format(len(process_list)))
      print("Queue list:: ", queue_list.qsize())
      CUDA_MEM = int(torch.cuda.get_device_properties(0).total_memory)
      N_JOBS = os.getenv('TTS_JOBS', round(CUDA_MEM*0.5/1000000000))
      
      print("Start TTS:: concurrency =", N_JOBS)
      with joblib.parallel_config(backend="loky", prefer="threads", n_jobs=int(N_JOBS)):
        results = Parallel(verbose=100)(delayed(self.tts)(text, output_file, tts_voice_ckpt_dir, speed, total_duration, start_silence) for (text, output_file, total_duration, start_silence) in queue_list.queue)
        
      print("Start Upsampling::")
      with joblib.parallel_config(backend="loky", prefer="threads", n_jobs=1):
        results = Parallel(verbose=100)(delayed(self.upsampling)(file) for (file) in results)
      del self.upsampler; gc.collect(); torch.cuda.empty_cache()
      
      print("TTS Done::")
      if torch.cuda.is_available() and convert_voice_ckpt_dir != "none":
        print("Start Voice Convertion::")
        self.convert_voice(tmp_dirname, convert_voice_ckpt_dir)
        
      if method == 'join':
        result_path, log_path = combine_wav_segment(results, final_output)
        print("combine_wav_segment result::", result_path, log_path)
        final_output = result_path
        log_output = log_path
      if method == 'split':
        archive_path = re.sub(r'\.wav$', '', final_output)
        shutil.make_archive(archive_path, 'zip', tmp_dirname)
        final_output = "{}.zip".format(archive_path)
      print("final_output_path::", final_output)
      
      ## Remove tmp files
      if os.path.exists(tmp_dirname): shutil.rmtree(tmp_dirname, ignore_errors=True)
      return (final_output, log_output)

  def speak(
    self,
    input_files,
    input_text,
    tts_voice="",
    convert_voice="none",
    speed=1,
    method="join"
    ):
      output_dir_name = new_dir_now()
      output_dir_path = os.path.join(CONFIG.os_tmp, output_dir_name)
      Path(output_dir_path).mkdir(parents=True, exist_ok=True)
      print("start speak_fn:", tts_voice)
      tts_voice_ckpt_dir = os.path.join(CONFIG.tts_ckpt_dir, tts_voice)
      convert_voice_ckpt_dir = os.path.join(CONFIG.convert_ckpt_dir, convert_voice) if convert_voice != "none" else "none"
      print("selected TTS voice:", tts_voice_ckpt_dir)
      print("selected Convert voice:", convert_voice_ckpt_dir)
      results_list = []
      result_text = CONFIG.empty_wav
      logs_list = []
      ## Process input_text first
      if input_text:
        try:
            print('input_text::', input_text)
            output_temp_file, log_temp_file = self.synthesize(output_dir_name, input_text, False, speed, method, tts_voice_ckpt_dir, convert_voice_ckpt_dir)
            if log_temp_file:
              logs_list.append(log_temp_file)
            if method == 'join':
              result_text = output_temp_file
            if method == 'split':
              results_list.append(output_temp_file)
        except:
            print("Skip error file while synthesizing input_text")
      ## Process input_files     
      if input_files:
        print("got input files::",input_files)
        file_list = [f.name for f in input_files]
        for file_path in file_list:
            try:
                print('file_path::',file_path)
                output_temp_file, log_temp_file = self.synthesize(output_dir_name, file_path, True, speed, method, tts_voice_ckpt_dir, convert_voice_ckpt_dir)
                results_list.append(output_temp_file)
                if log_temp_file:
                  logs_list.append(log_temp_file)
            except:
                print("Skip error file while synthesizing doc: {}".format(file_path))
      print("[DONE] {} tasks: {}".format(len(results_list), results_list))
      return results_list, result_text, logs_list

  def web_interface(self, port):
    css = """
    .btn-active {background-color: "orange"}
    """
    # title="VGM Text To Speech",
    # description = "A vietnamese text-to-speech by VGM speakers."
    tts_voices = [voice for voice in os.listdir(CONFIG.tts_ckpt_dir) if os.path.isdir(os.path.join(CONFIG.tts_ckpt_dir, voice))]
    convert_voices = ["none"] + [voice for voice in os.listdir(CONFIG.convert_ckpt_dir) if os.path.isdir(os.path.join(CONFIG.convert_ckpt_dir, voice))]
    app = gr.Blocks(title="VGM Text To Speech", theme=gr.themes.Default(), css=css)
    with app:
        gr.Markdown("# VGM Text To Speech" )
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Row():
                    with gr.Column():
                        input_files = gr.Files(label="Upload .doc|.docx|.txt|.srt file(s)", file_types=[".doc", ".docx", ".txt"])
                        textbox = gr.Textbox(label="Text for synthesize")
                        tts_voice = gr.Radio(label="Choose TTS Voice", value=tts_voices[0], choices=tts_voices)
                        convert_voice = gr.Radio(label="Choose Conversion Voice", value="none", choices=convert_voices)
                        duration_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1, step=0.02, label='Speed')
                        method = gr.Radio(label="Method", value="join", choices=["join","split"])
                    with gr.Column():
                        files_output = gr.Files(label="Files Audio Output")
                        audio_output = gr.Audio(label="Text Audio Output", elem_id="tts-audio")
                        logs_output = gr.Files(label="Error Audio Logs")
                        with gr.Row():
                          gr.ClearButton([input_files,textbox,files_output,audio_output,logs_output])
                          btn = gr.Button(value="Generate!", variant="primary")
                          btn.click(self.speak,
                                  inputs=[input_files, textbox, tts_voice, convert_voice, duration_slider, method],
                                  outputs=[files_output, audio_output, logs_output], concurrency_limit=1)
    auth_user = os.getenv('AUTH_USER', '')
    auth_pass = os.getenv('AUTH_PASS', '')
    app.queue().launch(
      auth=(auth_user, auth_pass) if auth_user != '' and auth_pass != '' else None,
      show_api=False,
      debug=False,
      inbrowser=True,
      show_error=True,
      server_name="0.0.0.0",
      server_port=port,
      share=False)

@atexit.register
def cleanup_tmp():
  if hooks.exit_code is not None:
      print("atexit call:: death by sys.exit(%d)" % hooks.exit_code)
  elif hooks.exception is not None:
      print("atexit call:: death by exception: %s" % hooks.exception)
  else:
      print("atexit call:: natural death")
      print("closing app:: cleanup_tmp")
      if os.path.exists( CONFIG.os_tmp): shutil.rmtree( CONFIG.os_tmp)
      if sys._MEIPASS2 and os.path.exists(sys._MEIPASS2): shutil.rmtree(sys._MEIPASS2)
  sys.exit()
      
if __name__ == "__main__":
    ## Download model if not exist
    # os.system('/bin/sh update_model.sh')
    ## Initialise app
    print("Application running on::", sys.platform)
    os.makedirs( CONFIG.os_tmp, exist_ok=True)
    os.system(f'rm -rf { CONFIG.os_tmp}/*')
    CONFIG.empty_wav.touch(exist_ok=True)
    ## Set torch multiprocessing
    mp.set_start_method('spawn', force=True)
    
    host = "localhost"
    port = 7901
    ## Parser argurment
    parser = argparse.ArgumentParser(description="VGM TTS application")
    parser.add_argument("-pf", "--platform", help="TTS Platform, default to desktop", default="web")
    parser.add_argument("-m", "--model", help="Custom path for model directory, default to current folder")
    parser.add_argument("-f", "--file", help="Input file for TTS")
    parser.add_argument("-t", "--text", help="Input text for TTS")
    parser.add_argument("-s", "--speed", help="Speed for TTS: default to 1", default=1)
    parser.add_argument("-mt", "--method", help="Method for TTS: join|split", default="join")
    parser.add_argument("-v", "--voice", help="Voice for TTS: male_name|female_name")
    parser.add_argument("-o", "--output", help="Output directory")
    args = parser.parse_args()
    ## Change ckpt_dir path if provided
    if args.model:
        CONFIG.tts_ckpt_dir = args.model
        print("ckpt_dir:",  CONFIG.tts_ckpt_dir)
    ## Execute app
    if args.platform == "web" and args.file:
        raise TypeError("Could not TTS from WEB and CLI at same time")
    elif args.platform == "cli" and args.file and args.text:
        raise TypeError("Could not TTS-CLI text and file at same time")
    elif args.platform == "web":
        root = TTS()
        root.web_interface(port)
    elif args.platform == "desktop":
        pass
        # start_desktop_interface(host, port)
    elif (args.platform == "cli" and args.file) or (args.platform == "cli" and args.text):
        pass
        # tts(file=args.file, text=args.text, voice=args.voice, speed=args.speed, method=args.method, output=args.output)
    else:
        raise TypeError("Not enough or wrong argument, please try again")
    sys.exit()