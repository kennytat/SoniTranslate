# Vendored package: `vieneu`

This directory vendors the `vieneu` TTS engine source so this repository's
install path (`pip install -r requirements_tts.txt`) needs **no access to the
private `git.vgm.cloud/models/VGM-TTS` repository** (and no PyPI dependency on
the upstream package, which is a different build).

## Origin

- Upstream project: <https://github.com/pnnbao97/VieNeu-TTS> (Apache-2.0)
- Vendored copy: the private VGM-TTS fork at commit
  `de1a2af2ffa5c0964807dc6b0c42e74c48617507`, version `3.2.8` (fork keeps the
  upstream version number but diverges from the published PyPI build).
- Package content is source-only (29 `.py` files in two packages: `vieneu`
  engine + `vieneu_utils` phonemizer helpers, ~430 KB). Model checkpoints,
  ONNX codecs and voice assets are downloaded from Hugging Face Hub at runtime
  by `huggingface_hub`, so nothing large or licensed is vendored here.

## Update procedure

Pull the private fork, verify the commit, and re-copy the tree:

```bash
cd /path/to/VGM-TTS && git rev-parse HEAD
rm -rf /path/to/SoniTranslate/vendor/vieneu/vieneu \
       /path/to/SoniTranslate/vendor/vieneu/vieneu_utils
cp -r /path/to/VGM-TTS/src/vieneu /path/to/SoniTranslate/vendor/vieneu/vieneu
cp -r /path/to/VGM-TTS/src/vieneu_utils /path/to/SoniTranslate/vendor/vieneu/vieneu_utils
find /path/to/SoniTranslate/vendor/vieneu -type d -name __pycache__ \
    -exec rm -rf {} +
# also clean transient build artifacts if you test-installed locally:
rm -rf /path/to/SoniTranslate/vendor/vieneu/build /path/to/SoniTranslate/vendor/vieneu/*.egg-info
```

then update the pinned commit reference in `VENDORED.md` and this note, and
re-run the smoke test of `vieneuTTS/`.
