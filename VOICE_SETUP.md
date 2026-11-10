# VieneuTTS voice setup

The Gradio UI and the HTTP `/voice` endpoint use the same local voice
discovery: `vieneu.list_available_voice_names()`. The list contains 25
bundled Turbo presets followed by the four existing LoRA voices (29 local
voices total). IDs stay plain strings; no API or settings format changes.

## Discovery is offline

The core package reads the Turbo manifest from the local Hugging Face cache
(`models--pnnbao-ump--VieNeu-TTS-v3-Turbo`) or a local model directory. It
never downloads voices automatically.

To use a different local copy, set an explicit directory containing
`manifest.json` and one `<voice-id>/` folder per preset with
`speaker.emb.txt` and `ref_codes.txt`:

```bash
export VIENEU_TURBO_VOICES_DIR=/path/to/turbo/voices
```

LoRA voices keep using the existing full adapter IDs and require the PyTorch
(PEFT) backend. Turbo presets do not need a LoRA adapter.
