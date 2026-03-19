# tests/test_output_builder.py

from accent_fix.services.output import output_builder
from accent_fix.services.accent_detector import AccentResult
from accent_fix.services.transcriber import TranscriptResult
from accent_fix.services.error_detector import ErrorDetectionResult
from accent_fix.services.corrector import CorrectionResult

def test_output_builder_no_corrections():
    accent = AccentResult("Nigerian English", 0.92, {}, None)
    transcript = TranscriptResult("Hello world", [], "assemblyai", 2.0, None)
    errors = ErrorDetectionResult([], 2, 0, 0.0, "Nigerian English", None)
    corrections = CorrectionResult("Hello world", "Hello world", [], 0, "Nigerian English", None)

    result = output_builder.build(accent, transcript, errors, corrections)

    assert result.accent == "Nigerian English"
    assert result.original_transcript == "Hello world"
    assert result.corrected_transcript == "Hello world"
    assert result.total_corrections_applied == 0
    assert result.error is None