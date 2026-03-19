from accent_fix.services.error_detector import error_detector
from accent_fix.services.transcriber import TranscriptResult
from accent_fix.services.accent_detector import AccentResult

def test_error_detector_returns_result():
    transcript = TranscriptResult(
        text="I want to confirm the recede",
        words=[
            {"word": "I", "start": 0, "end": 100, "confidence": 0.99},
            {"word": "want", "start": 100, "end": 300, "confidence": 0.98},
            {"word": "to", "start": 300, "end": 400, "confidence": 0.99},
            {"word": "confirm", "start": 400, "end": 700, "confidence": 0.97},
            {"word": "the", "start": 700, "end": 800, "confidence": 0.99},
            {"word": "recede", "start": 800, "end": 1100, "confidence": 0.60},
        ],
        engine="assemblyai",
        duration=1.1,
        error=None
    )

    accent = AccentResult(
        accent="Nigerian English",
        confidence=0.92,
        all_scores={"Nigerian English": 0.92},
        error=None
    )

    result = error_detector.detect(transcript, accent)

    assert result is not None
    assert isinstance(result.errors, list)
    assert result.total_words == 6
    assert result.error is None

def test_error_detector_no_words():
    transcript = TranscriptResult(
        text="",
        words=[],
        engine="whisper",
        duration=None,
        error=None
    )
    accent = AccentResult(
        accent="Nigerian English",
        confidence=0.92,
        all_scores={},
        error=None
    )
    result = error_detector.detect(transcript, accent)
    assert result.error is not None