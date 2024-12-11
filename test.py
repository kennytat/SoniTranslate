
# from utils.utils import download_manager
# from speech_to_text import STTClient
from tasks import app, stt
import joblib
from joblib import Parallel, delayed

audio_paths = ("/home/vgm/Downloads/logos.mp4", "/home/vgm/Downloads/logos.mp4", "/home/vgm/Downloads/logos.mp4", "/home/vgm/Downloads/logos.mp4", "/home/vgm/Downloads/logos.mp4")

with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(5)):
	stt_results = Parallel(verbose=100)(delayed(stt.delay)(audio_path, "en", 24, 24) for audio_path in audio_paths)

for index, result in enumerate(stt_results):
  print(f"result {index}::", result.get())

# result = stt.delay(audio_paths[0], "en", 24, 24)

# res = AsyncResult(result_id, app=app)

# stt_client = STTClient("en")
# # stt_client.init_model()
# result = stt_client.transcribe(audio_path, 24, 24)

    
# print("result::", result.get())
