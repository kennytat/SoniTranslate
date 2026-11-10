"""Single parent-side source for VieneuTTS voice IDs.

Discovery itself lives in the ``vieneu`` package so the Gradio UI and the
HTTP API see the same local presets.  This module only delegates; it looks
the function up on every call so refreshes are never served from a list
frozen at import time.
"""

from __future__ import annotations


def vieneu_voice_names():
  """Return the current VieneuTTS voice IDs (Turbo presets + LoRA voices)."""
  from vieneu import list_available_voice_names
  return list(list_available_voice_names())
