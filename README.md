# 🎥 SoniTranslate 🈷️

🎬 Video Translation with Synchronized Audio 🌐

SonyTranslate is a powerful and user-friendly web application that allows you to easily translate videos into different languages. This repository hosts the code for the SonyTranslate web UI, which is built with the Gradio library to provide a seamless and interactive user experience.

| Description       | Link                                                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📙 Colab Notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/R3gm/SoniTranslate/blob/main/SoniTranslate_Colab.ipynb)         |
| 🎉 Repository     | [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?style=flat-square&logo=github)](https://github.com/R3gm/SoniTranslate/)                                    |
| 🚀 Online DEMO    | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/r3gm/SoniTranslate_translate_audio_of_a_video_content) |

## SonyTranslate's web UI, which features a browser interface built on the Gradio library.

![image](https://github.com/R3gm/SoniTranslate/assets/114810545/53800b08-3a18-4f8a-be15-8710dc9102ec)

## Supported languages for translation

| Language Code | Language   |
| ------------- | ---------- |
| en            | English    |
| fr            | French     |
| de            | German     |
| es            | Spanish    |
| it            | Italian    |
| ja            | Japanese   |
| zh            | Chinese    |
| nl            | Dutch      |
| uk            | Ukrainian  |
| pt            | Portuguese |

## Installation

```
sudo apt-get update -y && apt-get upgrade -y
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -y
sudo apt-get install -y aria2 build-essential ffmpeg wget curl git vim cmake unzip cifs-utils tmux cudnn9-cuda-12 nvidia-cuda-toolkit
wget \
	https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \
	&& mkdir ~/.conda \
	&& bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b \
	&& rm -f Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
~/miniconda3/bin/conda init
conda create -n soni python=3.10.12 -y
conda activate soni
pip install -r requirements_stt.txt --no-cache-dir --resume-retries 10
pip install -r requirements_tts.txt --no-cache-dir --resume-retries 10
pip install -r requirements_ttt.txt --no-cache-dir --resume-retries 10
python -m spacy download en_core_web_sm
python
import nltk
nltk.download("punkt_tab")

export LD_LIBRARY_PATH=/home/vgm/miniconda3/envs/soni/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

rm -rf $HOME/miniconda3/envs/soni/lib/python3.10/site-packages/data/checkpoints
ln -s model/neuspell $HOME/miniconda3/envs/soni/lib/python3.10/site-packages/data/checkpoints
python app.py

mount smb:
sudo mount -t cifs //192.168.1.12/vgm-translate/ai1 /home/vgm/mount/output -o username=Administrator,password=,uid=1000,gid=1000,forceuid,forcegid,_netdev,auto,x-systemd.automount,x-systemd.mount-timeout=30
```

## Example:

### Original audio

https://github.com/R3gm/SoniTranslate/assets/114810545/db9e78c0-b228-4e81-9704-e62d5cc407a3

### Translated audio

https://github.com/R3gm/SoniTranslate/assets/114810545/6a8ddc65-a46f-4653-9726-6df2615f0ef9

## 📖 News

🔥 2023/07/26: New UI and add mix options.

🔥 2023/07/27: Fix some bug processing the video and audio.

🔥 2023/08/01: Add options for use RVC models.

🔥 2023/08/02: Added support for Arabic, Czech, Danish, Finnish, Greek, Hebrew, Hungarian, Korean, Persian, Polish, Russian, Turkish, Urdu, Hindi, and Vietnamese languages. 🌐

🔥 2023/08/03: Changed default options and added directory view of downloads.

## Contributing

Welcome to contributions from the community! If you have any ideas, bug reports, or feature requests, please open an issue or submit a pull request. For more information, please refer to the contribution guidelines.

## License

Although the code is licensed under Apache 2, the models or weights may have commercial restrictions, as seen with pyannote diarization.

## Docker

```
docker compose build
docker compose up
```
