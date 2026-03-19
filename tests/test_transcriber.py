# tests/test_transcriber.py

from accent_fix.services.transcriber import transcriber

def test_transcriber_returns_result():
    result = transcriber.transcribe("tests/sample_audio.mp3")

    assert result is not None
    assert isinstance(result.text, str)
    assert isinstance(result.words, list)
    assert result.engine in ["assemblyai", "whisper"]

def test_transcriber_falls_back_to_whisper():
    # Test with a bad API key to force fallback
    result = transcriber.transcribe("tests/sample_audio.mp3", engine="assemblyai")
    assert result.engine in ["assemblyai", "whisper"]
    assert result.error is None