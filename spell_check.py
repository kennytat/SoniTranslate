
## download models from gdrive to 
## ~/miniconda3/envs/soni/lib/python3.10/site-packages/data/checkpoints/
import os
import pickle

import neuspell
import torch
from neuspell.commons import ARXIV_CHECKPOINTS
from neuspell.seq_modeling.downloads import download_pretrained_model

# Same IDs as neuspell.seq_modeling.downloads.URL_MAPPINGS_BIG_FILES["scrnn-probwordnoise"]
_SCRNN_GDRIVE = {
    "model.pth.tar": "1cG0mduVmF7ChR2AOf58XKm0gsVB5d9aC",
    "vocab.pkl": "1M7MH3bL0pvnN5OoIBIxZV-F7G-XXi7qU",
}


def _file_starts_like_html(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as f:
        head = f.read(512)
    return head.lstrip().startswith(b"<")


def _torch_checkpoint_ok(path: str) -> bool:
    try:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return isinstance(ck, dict) and "model_state_dict" in ck


def _vocab_ok(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            v = pickle.load(f)
    except Exception:
        return False
    return isinstance(v, dict) and "token2idx" in v


def _scrnn_checkpoint_ok(ckpt_dir: str) -> bool:
    mp = os.path.join(ckpt_dir, "model.pth.tar")
    vp = os.path.join(ckpt_dir, "vocab.pkl")
    if not os.path.isfile(mp) or not os.path.isfile(vp):
        return False
    if _file_starts_like_html(mp) or _file_starts_like_html(vp):
        return False
    return _torch_checkpoint_ok(mp) and _vocab_ok(vp)


def _purge_scrnn(ckpt_dir: str) -> None:
    for name in ("model.pth.tar", "vocab.pkl"):
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            os.remove(p)


def _download_scrnn_via_gdown(ckpt_dir: str) -> bool:
    try:
        import gdown
    except ImportError:
        print("gdown not installed; install with: pip install gdown")
        return False
    os.makedirs(ckpt_dir, exist_ok=True)
    for fname, fid in _SCRNN_GDRIVE.items():
        dest = os.path.join(ckpt_dir, fname)
        url = f"https://drive.google.com/uc?id={fid}"
        try:
            gdown.download(url, dest, quiet=False, fuzzy=True)
        except TypeError:
            gdown.download(url, dest, quiet=False)
    return True


def _ensure_scrnn_checkpoint() -> None:
    """
    Neuspell's built-in Google Drive download often saves an HTML page as
    model.pth.tar (invalid load key '<'). gdown handles confirmations for
    large files; we verify with torch.load/pickle before use.
    """
    ckpt_dir = ARXIV_CHECKPOINTS["scrnn-probwordnoise"]
    if _scrnn_checkpoint_ok(ckpt_dir):
        return

    print("Neuspell scrnn checkpoint missing or invalid; re-downloading...")
    _purge_scrnn(ckpt_dir)

    try:
        _download_scrnn_via_gdown(ckpt_dir)
    except Exception as e:
        print(f"gdown download failed: {e}")

    if _scrnn_checkpoint_ok(ckpt_dir):
        return

    print("Retrying with neuspell's downloader...")
    _purge_scrnn(ckpt_dir)
    download_pretrained_model(ckpt_dir)

    if _scrnn_checkpoint_ok(ckpt_dir):
        return

    raise RuntimeError(
        "Neuspell scrnn-probwordnoise checkpoint is still invalid (often an HTML "
        "page from Google Drive). Try: pip install -U gdown && rm -rf '%s'/* "
        "then rerun, or symlink a manual download per README (model/neuspell)."
        % ckpt_dir
    )


class SpellCheck():
  def __init__(self):
    _ensure_scrnn_checkpoint()
    self.checker = neuspell.SclstmChecker()
    self.checker.from_pretrained()
        
  def correct(self, text):
    return self.checker.correct(text)
