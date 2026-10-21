import os
import gc
import json
import re
import threading
import torch
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from vieneu import Vieneu
from peft.peft_model import PeftModel


TTS_MODEL_DIR = os.path.join(os.getcwd(), "model", "vienue")


class VieneuTTS():
    def __init__(self):
        self.name = "VietTTS"  # Keep same name for UI compatibility
        self.tts_engine = None      # V3TurboVieNeuTTS instance (high-level API)
        self.current_lora = None    # Currently loaded LoRA path string
        self._base_model = None     # Original backbone before any LoRA attachment
        self._voice_presets = {}    # Cache: model_name -> voice dict
        self.sample_rate = 48000    # Vieneu v3-turbo outputs at 48 kHz
        # Vieneu v3-turbo engine is NOT thread-safe for independent calls:
        # _generate_codes() holds no lock and _load_lora/set_adapter() mutate
        # shared engine state. Concurrent joblib/ThreadPool workers calling
        # text_to_speech() can swap the active adapter mid-generation and crash
        # the CUDA context (device-side assert). All engine access is serialized
        # through this lock — batched inference (infer_batch) remains the only
        # supported parallelism (done inside engine.infer).
        self._infer_lock = threading.Lock()

    def release(self):
        """Release all resources (GPU memory, model weights)."""
        if self.tts_engine is not None:
            if hasattr(self.tts_engine, 'close'):
                self.tts_engine.close()
            self.tts_engine = None
        self.current_lora = None
        self._base_model = None
        self._voice_presets = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_engine(self):
        """Lazy-load the base Vieneu v3-turbo engine."""
        if self.tts_engine is None:
            print("Loading Vieneu v3-turbo base model...")
            self.tts_engine = Vieneu(mode="v3turbo")
            # Save reference to the original backbone model
            self._base_model = self.tts_engine.engine.model
            print("Vieneu v3-turbo base model loaded.")

    def _load_voice_preset(self, model_name):
        """Load voice preset (speaker_emb + codes) from LoRA directory."""
        if model_name in self._voice_presets:
            return self._voice_presets[model_name]

        preset_path = Path(TTS_MODEL_DIR) / model_name / "voice_preset.json"
        if not preset_path.exists():
            raise FileNotFoundError(f"Voice preset not found: {preset_path}")

        with open(preset_path, "r", encoding="utf-8") as f:
            preset = json.load(f)

        voice_dict = {
            "speaker_emb": np.array(preset["speaker_emb"], dtype=np.float32),
        }
        if "codes" in preset and preset["codes"] is not None:
            voice_dict["codes"] = np.array(preset["codes"], dtype=np.int64)

        self._voice_presets[model_name] = voice_dict
        return voice_dict

    def _load_lora(self, model_name):
        """
        Load a LoRA adapter from model/vienue/{model_name}/.
        Handles switching: attaches adapters additively (non-destructive) so the
        base model stays pristine. IMPORTANT: never use merge_and_unload() here —
        it mutates the base model in place, so switching back would stack adapters
        on an already-merged base and corrupt the output voice. Uses load_adapter +
        set_adapter instead.
        """
        lora_path = Path(TTS_MODEL_DIR) / model_name
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")

        self._ensure_engine()

        if self.current_lora == str(lora_path):
            return  # Already loaded

        adapter_name = "lora_" + re.sub(r"[^A-Za-z0-9_]", "_", model_name)
        engine_model = self.tts_engine.engine.model

        if isinstance(engine_model, PeftModel):
            # Switching: add the new adapter to the existing PeftModel
            print(f"Switching LoRA: {self.current_lora} -> {model_name}")
            engine_model.load_adapter(str(lora_path), adapter_name=adapter_name)
            engine_model.set_adapter(adapter_name)
            final_model = engine_model
        else:
            # First attach: wrap the pristine base model (base stays untouched)
            print(f"Loading LoRA adapter: {model_name}")
            final_model = PeftModel.from_pretrained(self._base_model, str(lora_path), adapter_name=adapter_name)
            self.tts_engine.engine.model = final_model

        final_model.eval()
        for p in final_model.parameters():
            p.requires_grad_(False)

        self.current_lora = str(lora_path)
        print(f"LoRA adapter active: {model_name}")

    def _apply_speed(self, audio_array, speed):
        """Apply speed change using librosa time_stretch (preserves pitch)."""
        import librosa
        audio_float = np.asarray(audio_array, dtype=np.float32)
        stretched = librosa.effects.time_stretch(audio_float, rate=speed)
        return stretched

    def text_to_speech(self, text, model_name, speed=1):
        """
        Generate speech audio matching VietTTS interface.

        Args:
            text: Input text (Vietnamese)
            model_name: LoRA directory name (e.g., "VGM-Han-LoRA")
            speed: Speech speed multiplier (default 1.0)

        Returns:
            pydub.AudioSegment compatible with TTSClient.make_voice_gradio()
        """
        # Handle silence requests (same as VietTTS)
        import re
        if re.sub(r'^sil\s+', '', text).isnumeric():
            silence_duration = int(re.sub(r'^sil\s+', '', text)) * 1000
            return AudioSegment.silent(duration=silence_duration).set_frame_rate(self.sample_rate)
        elif text == "♪":
            return AudioSegment.silent(duration=2000).set_frame_rate(self.sample_rate)

        # Serialize ALL shared-state work (adapter switch + engine inference).
        # The GpuJobSlot/VRAM gating may already be held by the caller from a
        # parallel worker thread — only one thread runs engine code at a time.
        with self._infer_lock:
            # Load LoRA if not already loaded or if changed
            lora_path = str(Path(TTS_MODEL_DIR) / model_name)
            if self.tts_engine is None or self.current_lora != lora_path:
                self._load_lora(model_name)

            # Load voice preset for this speaker
            voice_preset = self._load_voice_preset(model_name)

            # Run inference — returns numpy array (float32, normalized, 48 kHz)
            with torch.inference_mode():
                audio_array = self.tts_engine.infer(
                    text=text,
                    voice=voice_preset,
                    use_ref_codes="codes" in voice_preset,
                    apply_watermark=False,
                )

        # Apply speed change if needed
        if speed != 1.0 and speed > 0:
            audio_array = self._apply_speed(audio_array, speed)

        # Convert to int16 for pydub
        audio_array = np.asarray(audio_array, dtype=np.float32)
        max_val = np.abs(audio_array).max()
        if max_val > 1.0:
            audio_array = audio_array / max_val
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Create AudioSegment
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,  # 16-bit = 2 bytes
            channels=1       # Mono
        )
        return audio_segment

    def text_to_speech_batch(self, texts, model_name, speed=1):
        """
        Batched synthesis: many sentences of ONE speaker in one GPU pass.

        Uses `Vieneu.infer_batch` (static batching engine): chunks from all
        texts are flattened and run in length-bucketed groups of
        `max_batch_size` (default 32) — GPU stays saturated instead of
        running many tiny sequential autoregressive loops.

        Args:
            texts: list of Vietnamese strings
            model_name: LoRA speaker directory name
            speed: speed multiplier applied to every output

        Returns:
            list of pydub.AudioSegment, same order as `texts`
        """
        import re as _re
        static = _re.compile(r'^(sil \d+|♪)$')
        real_texts, out_order = [], []
        for t in texts:
            if static.match(t or ""):
                out_order.append(("static", t or ""))
            else:
                out_order.append(("tts", len(real_texts)))
                real_texts.append(t or "")

        # Serialize shared-state work the same way as text_to_speech()
        with self._infer_lock:
            lora_path = str(Path(TTS_MODEL_DIR) / model_name)
            if self.tts_engine is None or self.current_lora != lora_path:
                self._load_lora(model_name)
            voice_preset = self._load_voice_preset(model_name)

            if real_texts:
                with torch.inference_mode():
                    audio_arrays = self.tts_engine.infer_batch(
                        texts=real_texts,
                        voice=voice_preset,
                        use_ref_codes="codes" in voice_preset,
                        apply_watermark=False,
                    )
            else:
                audio_arrays = []

        def _to_segment(arr):
            arr = np.asarray(arr, dtype=np.float32)
            if speed != 1.0 and speed > 0:
                arr = self._apply_speed(arr, speed)
            max_val = np.abs(arr).max() if arr.size else 0.0
            if max_val > 1.0:
                arr = arr / max_val
            audio_int16 = (arr * 32767).astype(np.int16)
            return AudioSegment(
                audio_int16.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1,
            )

        results = []
        for kind, val in out_order:
            if kind == "static":
                if val.startswith("sil "):
                    results.append(AudioSegment.silent(duration=int(val.split()[1]) * 1000).set_frame_rate(self.sample_rate))
                else:
                    results.append(AudioSegment.silent(duration=2000).set_frame_rate(self.sample_rate))
            else:
                results.append(_to_segment(audio_arrays[val]))
        return results
