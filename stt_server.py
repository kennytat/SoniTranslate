from fastapi import FastAPI, Request, Header, UploadFile, HTTPException, Response, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import os
from pathlib import Path
import uuid
import tempfile
import torch
from pydantic import BaseModel
from typing import Optional
import whisperx
from sherpa_onnx_tts.stts import STTS
import numpy as np
import soundfile as sf
import re
from num2words import num2words
from speechbrain.inference.text import GraphemeToPhoneme
from lameenc import Encoder
from dotenv import load_dotenv
import json
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

temp_dir = os.getenv("APP_TEMP_DIR", os.path.join(tempfile.gettempdir(), "stt_server"))
TMP_FILE_DIRECTORY = os.path.join(temp_dir, "output")
whisper_model_default = os.getenv("WHISPER_MODEL",  "medium.en")
compute_type_default = os.getenv("COMPUTE_TYPE",  "float16")
tts_model = os.getenv("TTS_MODEL",  "csukuangfj/vits-coqui-en-vctk|109 speakers")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
  
class Whisper:
    def __init__(self, whisper_model="", device="", compute_type=compute_type_default, language='en'):
        self.current_model = whisper_model
        self.current_language = language
        self.device = device
        self.g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", run_opts={"device":"cuda"})
        self.model = whisperx.load_model(
            whisper_arch=whisper_model,
            device=device,
            compute_type=compute_type,
            language=None if language == 'Automatic detection' else language,
            )

    def stt(self, file_path="", align=False, batch_size=16, chunk_size=5):
        try:
          audio_bytes = whisperx.load_audio(file_path)
          # audio_bytes = np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0
          result = self.model.transcribe(audio_bytes, batch_size=batch_size, chunk_size=chunk_size, print_progress=True)
          if align:
            for segment in result['segments']:
              segment['text'] = convert_numbers_in_text(segment['text'])
            model_a, metadata = whisperx.load_align_model( language_code=result["language"], device=self.device, model_name=None)
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio_bytes,
                self.device,
                return_char_alignments=True,
                print_progress=False,
            )
            del model_a
          return result
        except Exception as e:
          logger.info('Error stt::', e)
          return ""
      
app = FastAPI(title="Audio Processing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the full error details
    error_detail = exc.errors()
    logger.error(f"Validation error: {error_detail}")
    
    # Return detailed error response
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": error_detail},
    )
    
stt_client = Whisper(whisper_model=whisper_model_default, device=device)
tts_client = STTS()



PHONEME_TO_VISEME = {
    # Viseme 0: Silence (not explicitly mapped)

    # Viseme 1: AE, AX, AH
    "AE": 1, "AH": 1, "AX": 1,
    
    # Viseme 2: AA
    "AA": 2,
    
    # Viseme 3: AO
    "AO": 3,
    
    # Viseme 4: EH, EY, UH
    "EH": 4, "EY": 4, "UH": 4,
    
    # Viseme 5: ER
    "ER": 5,
    
    # Viseme 6: IH, IY
    "IH": 6, "IY": 6,
    
    # Viseme 7: W, UW
    "W": 7, "UW": 7,
    
    # Viseme 8: OW
    "OW": 8,
    
    # Viseme 9: AW
    "AW": 9,
    
    # Viseme 10: AY
    "AY": 10,
    
    # Viseme 11: B, P, M
    "B": 11, "P": 11, "M": 11,
    
    # Viseme 12: CH, SH, JH, ZH
    "CH": 12, "SH": 12, "JH": 12, "ZH": 12,
    
    # Viseme 13: S, Z
    "S": 13, "Z": 13,
    
    # Viseme 14: TH, DH
    "TH": 14, "DH": 14,
    
    # Viseme 15: F, V
    "F": 15, "V": 15,
    
    # Viseme 16: D, T, N, L
    "D": 16, "T": 16, "N": 16, "L": 16,
    
    # Viseme 17: G, K, NG
    "G": 17, "K": 17, "NG": 17,
    
    # Viseme 18: R
    "R": 18,
    
    # Viseme 19: Y
    "Y": 19,
    
    # Viseme 20: HH
    "HH": 20
}

def convert_numbers_in_text(text):
    # Function to convert a number match to words
    def replace_num(match):
        number = match.group(0)
        try:
            # Convert to float first to handle both integers and decimals
            return num2words(float(number))
        except ValueError:
            # If conversion fails, return the original text
            return number
    
    # Use regex to find numbers in the text
    # This pattern matches integers and decimal numbers
    pattern = r'\b\d+(\.\d+)?\b'
    
    # Replace all numbers with their word equivalents
    result = re.sub(pattern, replace_num, text)
    return result
  
async def save_upload_file(audio_buffer, filename: str):
    temp_file = os.path.join(temp_dir, f"{uuid.uuid4()}{Path(filename).suffix}.ogg")
    with open(f"{temp_file}.raw", 'wb') as file:
        file.write(audio_buffer)
    cmd = f"ffmpeg -f s16le -ac 1 -acodec pcm_s16le -ar 16000 -i {temp_file}.raw {temp_file}"
    os.system(cmd)
    Path(f"{temp_file}.raw").unlink(missing_ok=True)
    return temp_file

# --------------- Route start here ---------------
@app.post("/stt")
async def stt(request: Request, x_audio_metadata: Optional[str] = Header(None)) -> JSONResponse:
    """
    Endpoint to receive audio buffer data with metadata header.
    
    Args:
        request: Request object containing the raw audio buffer
        x_audio_metadata: Header containing metadata about the audio
        
    Returns:
        JSON response with acknowledgment and metadata info
    """
    # Parse the metadata header if present
    metadata = {}
    if x_audio_metadata:
        try:
            metadata = json.loads(x_audio_metadata)
            logger.info(f"Received audio metadata: {metadata}")
        except json.JSONDecodeError:
            logger.error("Failed to parse X-Audio-Metadata header as JSON")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid metadata format. Expected JSON."}
            )
    try:
        # Get the raw audio buffer data
        audio_bytes = await request.body()
        temp_file = await save_upload_file(audio_buffer=audio_bytes, filename=metadata["filename"])
        try:
          stt_result = stt_client.stt(file_path=temp_file, align=False, batch_size=24, chunk_size=24)
          # Process the audio file
          result = "".join([segment["text"] for segment in  stt_result["segments"]])
          return JSONResponse(
              content={"message": "Success", "text": result},
              status_code=200
          )
        finally:
            Path(temp_file).unlink(missing_ok=True)
    except Exception as e:
        print("error::", e)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

class TTSRequest(BaseModel):
    text: str
    class Config:
        extra='allow'
            
class TTSResponseMeta(TTSRequest):
    name: str
    sample_rate: int
    visemes: list
                
@app.post("/tts")
async def tts(request: TTSRequest) -> Response:
    """
    Process text and return tts results.
    
    Args:
        text: string to tts
    
    Returns:
        JSONResponse containing the processing results or error message
    
    Raises:
        HTTPException: If any processing fails
    """
    print("TTS request::", request)
    if not request.text:
        raise HTTPException(status_code=400, detail="No text received")
    filename = request.filename if 'filename' in request else uuid.uuid4()
    try:
        try:
            # Process the audio file
            result = tts_client.predict(text=request.text, outpath="", repo_id=tts_model, sid="94", speed=1.0)
            # Normalize and scale if samples are float
            
            if isinstance(result.samples, list) or samples.dtype == np.float32:
                samples = np.array(result.samples, dtype=np.float32)
                samples = (samples * 32767).astype(np.int16)
            # Initialize MP3 encoder
            encoder = Encoder()
            encoder.set_bit_rate(128)  # Set desired bit rate (e.g., 192 kbps)
            encoder.set_in_sample_rate(result.sample_rate)
            encoder.set_channels(1 if samples.ndim == 1 else 2)
            encoder.set_out_sample_rate(result.sample_rate)
            # Encode samples to MP3
            mp3_data = encoder.encode(samples.tobytes())
            mp3_data += encoder.flush()
            temp_file = os.path.join(TMP_FILE_DIRECTORY, f"{filename}.mp3")
            with open(temp_file, "wb") as mp3_file:
              mp3_file.write(mp3_data)
            
            ## Extract visemes
            stt_result = stt_client.stt(file_path=temp_file, align=True, batch_size=24, chunk_size=24)
            stt_result['segments'] = [{'text': segment['text'], 'start': segment['start'], 'end': segment['end'], 'words': [{ 'word': word['word'], 'phonemes': stt_client.g2p(word['word']), 'start': word['start'], 'end': segment['words'][wordIndex + 1]['start'] if (wordIndex < len(segment['words'])-1) else word['end'], 'duration': (segment['words'][wordIndex + 1]['start'] if (wordIndex < len(segment['words'])-1) else word['end']) - word['start']} for (wordIndex, word) in enumerate(segment['words'])] } for segment in stt_result['segments']]
            logger.info("\nsegment::\n", stt_result['segments'])
            visemes = [
                {'shape': phoneme, 'duration': round(word['duration']/len(word['phonemes']), 4)}
                for segment in stt_result['segments']
                for word in segment['words'] 
                for phoneme in word['phonemes']
            ]
            logger.info("\nvisemes::\n", visemes)
            metadata = TTSResponseMeta(
                name=f"{filename}.mp3",
                sample_rate=result.sample_rate,
                visemes=visemes,
                **request.model_dump()
            )
            headers = {"X-Audio-Metadata": metadata.model_dump_json()}
            return FileResponse(
                path=temp_file,
                filename=f"{filename}.mp3",
                media_type=None,  # Let FastAPI guess the content type
                headers=headers
            )
        except Exception as error:
            logger.info("error tts::", error)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
      
@app.get("/file")
async def serve_file(name: Optional[str] = None):
    """
    Serve files from the specified directory.
    
    Args:
        name (str): The name of the file to serve
        
    Returns:
        FileResponse: The requested file
        
    Raises:
        HTTPException: If file is not found or name is not provided
    """
    if not name:
        raise HTTPException(status_code=400, detail="File name is required")
    
    try:
        file_path = os.path.join(TMP_FILE_DIRECTORY, name)
        
        # Ensure the file exists and is within the allowed directory
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Ensure the resolved path is within the allowed directory
        if not Path(file_path).resolve().is_relative_to(Path(TMP_FILE_DIRECTORY).resolve()):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(
            path=file_path,
            filename=name,
            media_type=None  # Let FastAPI guess the content type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
            
@app.get("/health")
async def health_check() -> JSONResponse:
    """Simple health check endpoint."""
    return JSONResponse(content={"status": "healthy"}, status_code=200)

def download_espeak_ng_data():
    os.system(
        """
    cd /tmp
    wget -qq https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
    tar xf espeak-ng-data.tar.bz2
    """
    )

if __name__ == "__main__":
    download_espeak_ng_data()
    Path(TMP_FILE_DIRECTORY).mkdir(exist_ok=True, parents=True)
    os.system(f"rm -rf {TMP_FILE_DIRECTORY}/*")
    import uvicorn
    port = os.getenv("PORT", 8008)
    uvicorn.run(app, host="0.0.0.0", port=port)
