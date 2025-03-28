from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
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
from lameenc import Encoder
from dotenv import load_dotenv
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
        self.model = whisperx.load_model(
            whisper_arch=whisper_model,
            device=device,
            compute_type=compute_type,
            language=None if language == 'Automatic detection' else language,
            )

    def stt(self, file_path="", batch_size=16, chunk_size=5):
        try:
          audio_bytes = whisperx.load_audio(file_path)
          result = self.model.transcribe(audio_bytes, batch_size=batch_size, chunk_size=chunk_size, print_progress=True)
          return result
        except Exception as e:
          print('Error stt::', e)
          return ""
      
app = FastAPI(title="Audio Processing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stt_client = Whisper(whisper_model=whisper_model_default, device=device)
tts_client = STTS()

async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save uploaded file to disk and return the file path."""
    temp_file = os.path.join(temp_dir, f"{uuid.uuid4()}{Path(upload_file.filename).suffix}")
    print("file::", upload_file, temp_file)
    
    try:
        with Path(temp_file).open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    return temp_file

@app.post("/stt")
async def stt(file: UploadFile) -> JSONResponse:
    """
    Process an uploaded audio file and return the results.
    
    Args:
        file: The uploaded audio file
    
    Returns:
        JSONResponse containing the processing results or error message
    
    Raises:
        HTTPException: If the file is invalid or processing fails
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Save the uploaded file
        temp_file = await save_upload_file(file)
        try:
            # Process the audio file
            stt_result = stt_client.stt(file_path=temp_file, batch_size=24, chunk_size=24)
            result = "".join([segment["text"] for segment in  stt_result["segments"]])           
            return JSONResponse(
                content={"message": "Success", "result": result},
                status_code=200
            )
        finally:
            # Clean up: remove the temporary file
            Path(temp_file).unlink(missing_ok=True)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

class TTSRequest(BaseModel):
    text: str
    filename: str
    
@app.post("/tts")
async def tts(request: TTSRequest) -> FileResponse:
    """
    Process text and return tts results.
    
    Args:
        text: string to tts
    
    Returns:
        JSONResponse containing the processing results or error message
    
    Raises:
        HTTPException: If any processing fails
    """
    if not request.text or not request.filename:
        raise HTTPException(status_code=400, detail="No text received") 
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
            filename = f"{request.filename}.mp3"
            tmp_file = os.path.join(temp_dir, "output", filename)
            with open(tmp_file, "wb") as mp3_file:
              mp3_file.write(mp3_data)
            return FileResponse(
                path=tmp_file,
                filename=filename,
                media_type=None  # Let FastAPI guess the content type
            )
        except Exception as error:
            print("error tts::", error)
            
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
