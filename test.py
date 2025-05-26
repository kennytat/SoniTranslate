import numpy as np
from inference import VisemeRegressor

pb_filepath = "./visemenet_frozen.pb"
wav_file_path = "./test_audio.wav"
out_txt_path = "./maya_viseme_outputs.txt"

viseme_regressor = VisemeRegressor(pb_filepath=pb_filepath)

viseme_outputs = viseme_regressor.predict_outputs(wav_file_path=wav_file_path)

np.savetxt(out_txt_path, viseme_outputs, '%.4f')