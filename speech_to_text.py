import torch
import os
import whisperx
torch.cuda.set_per_process_memory_fraction(0.5)
# Check GPU
YOUR_HF_TOKEN = os.getenv("YOUR_HF_TOKEN", "")
    
if torch.cuda.is_available():
    device = "cuda"
    list_compute_type = ['float16', 'float32']
    compute_type_default = 'float16'
    CUDA_MEM = int(torch.cuda.get_device_properties(0).total_memory)
    whisper_model_default = 'large-v3' if CUDA_MEM > 9000000000 else 'medium'
else:
    device = "cpu"
    list_compute_type = ['float32']
    compute_type_default = 'float32'
    whisper_model_default = 'medium'
    
class STTClient():
  def __init__(self, language = 'Automatic detection'):
    self.whisper_model = None
    self.diarize_model = None
    self.language = language
    self.whisper_model = whisperx.load_model(
              whisper_model_default,
              device=device,
              compute_type=compute_type_default,
              language=self.language,
              )
    self.diarize_model = "pyannote/speaker-diarization-3.1" ## "pyannote/speaker-diarization-3.1" "pyannote/speaker-diarization@2.1"
    self.diarize_model = whisperx.DiarizationPipeline(model_name=self.diarize_model, use_auth_token=YOUR_HF_TOKEN, device=device)
    
  def init_model(self, language):
    if self.language != language:
      self.language = language
      self.whisper_model = whisperx.load_model(
                whisper_model_default,
                device=device,
                compute_type=compute_type_default,
                language=self.language,
                )
    return self.whisper_model
        
  def transcribe(self, audio_wav, batch_size, chunk_size):
      audio = whisperx.load_audio(audio_wav)
      result = self.whisper_model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size, print_progress=True)
      return result
    
  def diarize(self, audio_wav, transcript, min_speakers, max_speakers):
      diarize_segments = self.diarize_model(
          audio_wav,
          min_speakers=min_speakers,
          max_speakers=max_speakers)
      result = whisperx.assign_word_speakers(diarize_segments, transcript)
      return result