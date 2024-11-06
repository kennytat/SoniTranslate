#!/bin/bash
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
pip install llama-cpp-python --upgrade --force-reinstall llama-cpp-python --no-cache-dir
pip install pip==24.0
pip install -r requirements_stt.txt --force-reinstall --no-cache-dir
pip install -r requirements_ttt.txt --force-reinstall --no-cache-dir
pip install -r requirements_tts.txt --force-reinstall --no-cache-dir
pip install -r requirements_extra.txt --force-reinstall --no-cache-dir
