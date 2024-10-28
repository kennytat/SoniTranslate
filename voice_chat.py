# import argparse
# import re
# import subprocess
import logging
import os
import queue
import sys
import tempfile
import time
import warnings
from typing import List
import wave
import torch
from pathlib import Path
import shutil
import ffmpeg
import emoji
import numpy
import whisperx
from llm_translate import LLM
from text_to_speech import TTSClient, voice_conversion
from datetime import datetime
import pygame
import audioop
import pyaudio
from termcolor import colored

# ------------------------------------- global variable initialization ---------------------------------------------- #
warnings.filterwarnings("ignore")
pygame.init()
logger = logging.getLogger(__name__)
# global variable to store the audio device choices.

# ----------------------------------------- decorator libraries ----------------------------------------------------- #
emoji_man = "\U0001F9D4"
emoji_women = emoji.emojize(":woman:")
emoji_system = emoji.emojize(":robot:")
emoji_user = emoji.emojize(":supervillain:")
emoji_speaking = emoji.emojize(":speaking_head:")
emoji_sparkiles = emoji.emojize(":sparkles:")
emoji_jack_o_lantern = emoji.emojize(":jack-o-lantern:")
emoji_microphone = emoji.emojize(":studio_microphone:")
emoji_rocket = emoji.emojize(":rocket:")

temp_dir = os.getenv("APP_TEMP_DIR", os.path.join(tempfile.gettempdir(), "vgm_voice_chat"))
Path(temp_dir).mkdir(parents=True, exist_ok=True)
if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"
compute_type = "float32" if device == "cpu" else "float16"
pygame.mixer.init()

def play_sound(file):
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()

def stop_sound():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

def is_silent(data_chunk, threshold):
    """Check if the audio chunk is below the silence threshold."""
    rms = audioop.rms(data_chunk, 2)  # width=2 for paInt16
    return rms < threshold
      
def new_dir_now():
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d%H%M%S")
    return date_time  
  
class Whisper:
    def __init__(self, whisper_model="", device="", compute_type=compute_type if device == "cpu" else "float16", language=None):
        self.model = whisperx.load_model(
            whisper_arch=whisper_model,
            device=device,
            compute_type=compute_type,
            language=language,
            )

    def stt(self, audio_bytes:numpy.ndarray = [], batch_size=16, chunk_size=5):
        try:
          result = self.model.transcribe(audio_bytes, batch_size=batch_size, chunk_size=chunk_size, print_progress=True)
          speech = ", ".join([ segment["text"].strip() for segment in result["segments"]])
          return speech
        except Exception as e:
          print('Error stt::', e)
          return ""
      
class VoiceChat():
    def __init__(self):
      self.q: queue.Queue = queue.Queue()
      self.whisper = Whisper(whisper_model="large-v3", device=device, language="en")
      self.llm = LLM()
      self.llm.initLLM( 
        endpoints="https://infer-2.vn.chattrust.ai/v1", 
        model="trast-ai/trust-translator-llama3-5b4e", ## "nampdn-ai/vietmistral-chatvgm-3072" "nampdn-ai/vietmistral-bible-translation"
        temp=0.3,
        k=10
      )
      self.tts_client = TTSClient()
      self.tts_client.init_tts_client(client="VietTTS")
    # --------------------------------- supplemented util to get the record --------------------------------------------- #

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    # function to take audio input and transcript it into text-file.
    def record_unlimited(self) -> numpy.ndarray:
        filename = os.path.join(temp_dir, f'user_{new_dir_now()}.wav')
        max_duration=30
        silence_threshold=5000
        silence_duration=5
        sample_rate=44100
        channels=2
        chunk=1024
        audio = pyaudio.PyAudio()

        # Open stream
        stream = audio.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk)

        print("Recording... (Press Ctrl+C to stop)")

        frames = []
        silent_chunks = 0
        speech_start = False
        start_time = time.time()

        try:
            while True:
                data = stream.read(chunk)
                silence = is_silent(data, silence_threshold)

                if not silence:
                  speech_start = True
                  stop_sound()
                  
                if speech_start:
                  frames.append(data)
                else:
                  frames = [data]
                  
                if silence and speech_start:
                    silent_chunks += 1

                # Stop if silence is detected for more than silence_duration
                if silent_chunks > silence_duration * sample_rate / chunk:
                    silent_chunks = 0
                    speech_start = False
                    print(f"Silence over {silence_duration} seconds. Stopping recording.")
                    break

                # Stop if maximum duration is reached
                if time.time() - start_time > max_duration:
                    silent_chunks = 0
                    speech_start = False
                    print("Maximum duration reached. Stopping recording.")
                    break

        except KeyboardInterrupt:
            print("Recording stopped by user")

        finally:
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            audio.terminate()

            # Save the recorded data as a WAV file
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()     

        try:
            y, _ = (
                ffmpeg.input(os.path.abspath(filename), threads=0)
                .filter('afftdn', nr=10, nt='w', om='o')
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
        os.remove(filename)
        return numpy.frombuffer(y, numpy.int16).flatten().astype(numpy.float32) / 32768.0


    def speech_to_text(self, audio_input) -> str:
        output_text = self.whisper.stt(audio_bytes=audio_input)
        print(f"{emoji_man} ------ User ------ {emoji_man}")
        print(colored(output_text, 'yellow'))
        return output_text

    # transcript the generated chatbot word to audio output so the user will hear the result.
    def text_to_audio(self, tts_text="", play_sound_wait=0):
        tts_voice="vn_han_male"
        tts_speed=1
        filename=os.path.join(temp_dir, f"assistant_{new_dir_now()}.wav")
        language="vi"
        t2s_method="VietTTS"
        self.tts_client.make_voice_gradio(tts_text=tts_text, tts_voice=tts_voice, tts_speed=tts_speed, filename=filename, language=language, t2s_method=t2s_method)
        print(f"{emoji_man} ------ Assistant ------ {emoji_man}")
        print(colored(tts_text, 'green'))
        play_sound(filename)
        time.sleep(play_sound_wait)

    def chat_with_bot(self, input):
        return self.llm.process(text=input)
      
    def start(self):
        print(f"{emoji_jack_o_lantern} ------ Welcome to the VGM chatroom ------ {emoji_jack_o_lantern}")
        print(emoji_sparkiles, end="")
        print(f"{emoji_sparkiles} Say something with 'exit', 'quit', 'bye', or 'see you' to leave the chatroom {emoji_sparkiles}")
        print(" ------------------ ")
        ## Greeting
        welcome_prompt = "Xin chào, tôi là một trở lý dịch thuật, xin hãy nói bằng tiếng anh và tôi sẽ dịch sang tiếng việt"
        self.text_to_audio(tts_text=welcome_prompt)

        while True:
            ## Comment this line to disable Enter to start talking
            # input("  Press Enter to start talking and press Ctrl+C to stop the recording:")
            audio_input = self.record_unlimited()

            start = time.time()
            format_input = self.speech_to_text(audio_input=audio_input)
            logger.info(f"Time spent on transcribing: {time.time() - start}")

            # for natural exit, both bot is expected to send greeting message.
            if "exit" in format_input.lower() or "quit" in format_input.lower() or "bye" in format_input.lower() or "see you" in format_input.lower():
                ending_prompt = "Cám ơn bạn đã tin tưởng và sử dụng dịch vụ Vi Tri Em Chat, hẹn gặp lại lần sau nhé."
                self.text_to_audio(tts_text=ending_prompt, play_sound_wait=7)
                break
              
            if len(format_input) > 1:
              output = self.chat_with_bot(input=format_input)
              self.text_to_audio(tts_text=output)
            else:
              no_speech_prompt = "Xin lỗi tôi không nghe được bạn, xin hãy thử lại nhé"
              self.text_to_audio(tts_text=no_speech_prompt)

        ## Remove temp_dir when finish
        shutil.rmtree(temp_dir)     


# ---------------------------------------- The program will run from below: ------------------------------------------#
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-e", "--endpoint", type=str, help="Xinference endpoint, required", required=True
    # )
    # args = parser.parse_args()
    # endpoint = args.endpoint
    
    vc = VoiceChat()
    vc.start()
    

