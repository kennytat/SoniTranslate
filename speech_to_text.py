import torch
import whisperx
torch.cuda.set_per_process_memory_fraction(0.5)
# Check GPU
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
    self.model = None
    self.language = language
    self.model = whisperx.load_model(
              whisper_model_default,
              device=device,
              compute_type=compute_type_default,
              language=self.language,
              )
    
  def init_model(self, language):
    if self.language != language:
      self.language = language
      self.model = whisperx.load_model(
                whisper_model_default,
                device=device,
                compute_type=compute_type_default,
                language=self.language,
                )
    return self.model
        
  def transcribe(self, audio_wav, batch_size, chunk_size):
      audio = whisperx.load_audio(audio_wav)
      result = self.model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size, print_progress=True)
      return result