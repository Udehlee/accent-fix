from accent_fix.services.accent_detector import accent_detector

def test_accent_detector_returns_result():
    result = accent_detector.detect("tests/sample_audio.mp3")
    
    assert result is not None
    assert result.accent is not None
    assert isinstance(result.accent, str)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.all_scores, dict)

def test_accent_detector_handles_missing_file():
    result = accent_detector.detect("nonexistent_file.mp3")
    assert result.error is not None