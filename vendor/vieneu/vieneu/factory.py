

def Vieneu(mode="v3turbo", **kwargs):
    """
    Factory function for VieNeu-TTS v3 Turbo.

    Args:
        mode: 'v3turbo' (DEFAULT) — VieNeu-TTS v3 Turbo, 48 kHz.
        **kwargs: Arguments for V3TurboVieNeuTTS

    Returns:
        V3TurboVieNeuTTS: An instance of VieNeu-TTS v3 Turbo.
    """
    match mode:
        case "v3turbo":
            from .v3turbo import V3TurboVieNeuTTS
            return V3TurboVieNeuTTS(**kwargs)
        case _:
            raise ValueError(
                f"Chỉ hỗ trợ mode='v3turbo' cho pipeline v3 Turbo (nhận được mode='{mode}')."
            )