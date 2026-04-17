from faster_whisper import WhisperModel

from app.services.text_emotion import text_emotion


_whisper = None


def _whisper_lazy():
    global _whisper
    if _whisper is None:
        # medium is a decent Chinese/English speed-quality tradeoff on a single GPU;
        # swap to small/base for tighter latency budgets
        _whisper = WhisperModel("medium", device="cuda", compute_type="float16")
    return _whisper


def transcribe(audio_path: str, language: str = "zh") -> str:
    segments, _info = _whisper_lazy().transcribe(
        audio_path, language=language, beam_size=5, vad_filter=True,
    )
    return "".join(s.text for s in segments).strip()


async def audio_to_emotion(audio_path: str, language: str = "zh") -> dict:
    text = transcribe(audio_path, language)
    label = await text_emotion(text)
    return {"text": text, "label": label}
