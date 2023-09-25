import os
from argparse import Namespace
from pathlib import Path
from typing import NamedTuple
import tempfile

from jax.numpy import ndarray


class FLAGS(Namespace):
    """Configurations"""
    duration_lstm_dim = 256
    vocab_size = 256
    duration_embed_dropout_rate = 0.5
    num_training_steps = 200_000
    postnet_dim = 512
    acoustic_decoder_dim = 512
    acoustic_encoder_dim = 256

    # dataset
    max_phoneme_seq_len = 256 * 1
    assert max_phoneme_seq_len % 256 == 0  # prevent compilation error on Colab T4 GPU
    max_wave_len = 1024 * 64 * 3

    # Montreal Forced Aligner
    special_phonemes = ["sil", "sp", "spn", ""]  # [sil], [sp] [spn] [word end]
    sil_index = special_phonemes.index("sil")
    sp_index = sil_index  # no use of "sp"
    word_end_index = special_phonemes.index("")
    _normal_phonemes = (
        []
        + ["a", "b", "c", "d", "e", "g", "h", "i", "k", "l"]
        + ["m", "n", "o", "p", "q", "r", "s", "t", "u", "v"]
        + ["x", "y", "à", "á", "â", "ã", "è", "é", "ê", "ì"]
        + ["í", "ò", "ó", "ô", "õ", "ù", "ú", "ý", "ă", "đ"]
        + ["ĩ", "ũ", "ơ", "ư", "ạ", "ả", "ấ", "ầ", "ẩ", "ẫ"]
        + ["ậ", "ắ", "ằ", "ẳ", "ẵ", "ặ", "ẹ", "ẻ", "ẽ", "ế"]
        + ["ề", "ể", "ễ", "ệ", "ỉ", "ị", "ọ", "ỏ", "ố", "ồ"]
        + ["ổ", "ỗ", "ộ", "ớ", "ờ", "ở", "ỡ", "ợ", "ụ", "ủ"]
        + ["ứ", "ừ", "ử", "ữ", "ự", "ỳ", "ỵ", "ỷ", "ỹ"]
    )

    # dsp
    mel_dim = 80
    n_fft = 1024
    sample_rate = 22050
    fmin = 0.0
    fmax = 8000

    # training
    batch_size = 64
    learning_rate = 1e-4
    duration_learning_rate = 1e-4
    max_grad_norm = 1.0
    weight_decay = 1e-4
    token_mask_prob = 0.1

    # ckpt
    os_tmp = Path(os.path.join(tempfile.gettempdir(), "vgm-translate"))
    empty_wav = Path(os.path.join(f'{os_tmp}', "test.wav"))
    tts_ckpt_dir = Path(os.path.join(os.getcwd(), "model", "vietTTS"))
    convert_ckpt_dir = Path(os.path.join(os.getcwd(), "model", "svc"))
    salt = Path(os.path.join(os.getcwd(), "model","vietTTS", "salt.salt"))
    key = "^VGMAI*607#"

class DurationInput(NamedTuple):
    phonemes: ndarray
    lengths: ndarray
    durations: ndarray

class AcousticInput(NamedTuple):
    phonemes: ndarray
    lengths: ndarray
    durations: ndarray
    wavs: ndarray
    wav_lengths: ndarray
    mels: ndarray