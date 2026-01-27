@ECHO OFF
TITLE VGM-TTS
ECHO "Starting application, please wait..."
wsl -u vgm /bin/bash -ic "if lsof -t -i:7901;then kill -9 $(lsof -t -i:7901);fi"
wsl -u vgm /bin/bash -ic "cd ~/Projects/sonitranslate && conda activate soni && python tts.py"
PAUSE