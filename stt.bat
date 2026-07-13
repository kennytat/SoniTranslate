@ECHO OFF
TITLE VGM-STT
ECHO "Starting application, please wait..."
wsl -u vgm /bin/bash -ic "if lsof -t -i:3100;then kill -9 $(lsof -t -i:3100);fi"
wsl -u vgm /bin/bash -ic "cd ~/Projects/sonitranslate && conda activate soni && python stt.py"

:: cd /d D:\vgm-translate
:: docker compose run --rm -p 3100:3100 soni python stt.py

PAUSE