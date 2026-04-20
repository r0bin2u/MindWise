from tenacity import retry, stop_after_attempt, wait_exponential

from app.services.text_emotion import text_emotion


_whisper = None


def _whisper_lazy():
    """Lazy-load faster-whisper. `medium` is the sweet spot for zh/en on a
    single GPU; swap to `small`/`base` if the latency budget tightens."""
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel("medium", device="cuda", compute_type="float16")
    return _whisper


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=2))
def _transcribe(audio_path: str, language: str) -> str:
    segments, _info = _whisper_lazy().transcribe(
        audio_path, language=language, beam_size=5, vad_filter=True,
    )
    return "".join(s.text for s in segments).strip()


async def audio_to_emotion(audio_path: str, language: str = "zh") -> dict:
    """Transcribe audio and classify emotion.

    Falls back to empty transcript + '正常' label on any exception so the
    upstream fusion engine can just treat this as a missing modality.
    """
    try:
        text = _transcribe(audio_path, language)
    except Exception as e:
        return {"text": "", "label": None, "error": f"transcribe_failed: {e!s}"[:200]}
    if not text:
        return {"text": "", "label": None, "error": "empty_transcript"}
    label = await text_emotion(text)
    return {"text": text, "label": label}
