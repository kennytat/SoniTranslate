"""Coqui TTS XTTS does `from transformers import BeamSearchScorer`.

Hugging Face transformers 4.57+ no longer re-exports BeamSearchScorer at package root
(it remains in transformers.generation.beam_search). Import this module before any
`TTS` import so the legacy import inside Coqui TTS succeeds.
"""

import transformers
from transformers.generation.beam_search import BeamSearchScorer

if not hasattr(transformers, "BeamSearchScorer"):
    transformers.BeamSearchScorer = BeamSearchScorer
