#!/bin/sh

## Start TTS
python tts.py &
## Start STT
python stt.py &
## Start upsample
python upsample.py &
## Start vgm-translate
python app.py "$@"