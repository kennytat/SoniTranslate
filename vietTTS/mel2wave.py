import json
import os
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from vietTTS.crypto import decrypt_byte

from vietTTS.config import FLAGS
from vietTTS.hifigan_model import Generator

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def mel2wave(
    mel,
    config_file=os.path.join(FLAGS.tts_ckpt_dir, "config.json"),
    ckpt_file=os.path.join(FLAGS.tts_ckpt_dir, "hk_hifi.pickle"),
):
    print("starting mel2wav::")
    MAX_WAV_VALUE = 32768.0
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    @hk.transform_with_state
    def forward(x):
        net = Generator(h)
        return net(x)

    rng = next(hk.PRNGSequence(42))

    with open(ckpt_file, "rb") as f:
        encrypted = f.read()
        decrypted = decrypt_byte(encrypted, FLAGS.key)
        # load encrypted checkpoint file: pickle.load(f)
        # params = pickle.load(f)
        # load decrypted checkpoint bytes: pickle.loads(decrypt)
        params = pickle.loads(decrypted)
    aux = {}
    wav, aux = forward.apply(params, aux, rng, mel)
    print("Completed synthesize to wav")
    # print("wav:", wav, "aux:", aux)
    wav = jnp.squeeze(wav)
    audio = jax.device_get(wav)
    return audio