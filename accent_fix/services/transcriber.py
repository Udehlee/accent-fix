import assemblyai as aai
import whisper
import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# What this service returns
# ─────────────────────────────────────────
@dataclass
class TranscriptResult:
    text: str                  # full transcript text
    words: list                # each word with timestamps [{word, start, end, confidence}]
    engine: str                # which engine was used: "assemblyai" or "whisper"
    duration: Optional[float]  # audio duration in seconds
    error: Optional[str]       # None if success


# ─────────────────────────────────────────
# Service
# ─────────────────────────────────────────
class Transcriber:
    """
    Transcribes audio using AssemblyAI with Whisper as local fallback.

    Fallback order:
    AssemblyAI (cloud) → fails → Whisper (local, no internet needed)

    Why this fallback order:
    - AssemblyAI is faster, more accurate, and returns proper word-level
      timestamps with confidence scores
    - Whisper is free, runs locally, needs no API key, and always works
      even when there is no internet or API credits are exhausted
    """

    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

    # Whisper model size
    # "tiny"   → fastest, least accurate
    # "base"   → good balance for development
    # "small"  → better accuracy, still reasonable speed
    # "medium" → high accuracy, slower
    # "large"  → best accuracy, slowest
    WHISPER_MODEL_SIZE = "base"

    def __init__(self):
        # Configure AssemblyAI
        if self.ASSEMBLYAI_API_KEY:
            aai.settings.api_key = self.ASSEMBLYAI_API_KEY
            logger.info("AssemblyAI configured")
        else:
            logger.warning("ASSEMBLYAI_API_KEY not found — will fall back to Whisper")

        # Load Whisper model once into memory when app starts
        # Not reloaded per request — that would be too slow
        logger.info(f"Loading Whisper model: {self.WHISPER_MODEL_SIZE}")
        self.whisper_model = whisper.load_model(self.WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded and ready as fallback")

    # ─────────────────────────────────────
    # AssemblyAI
    # ─────────────────────────────────────
    def _transcribe_assemblyai(self, audio_path: str) -> TranscriptResult:
        """
        Transcribes audio using AssemblyAI.

        Returns full text and word-level timestamps with confidence scores.
        """
        logger.info(f"Transcribing with AssemblyAI: {audio_path}")

        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            language_code="en",
            punctuate=True,
            format_text=True,
        )

        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_path)

        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"AssemblyAI error: {transcript.error}")

        # Extract word-level timestamps
        words = []
        if transcript.words:
            for word in transcript.words:
                words.append({
                    "word": word.text,
                    "start": word.start,        # milliseconds
                    "end": word.end,            # milliseconds
                    "confidence": round(word.confidence, 4)
                })

        return TranscriptResult(
            text=transcript.text or "",
            words=words,
            engine="assemblyai",
            duration=transcript.audio_duration,
            error=None
        )

    # ─────────────────────────────────────
    # Whisper (local fallback)
    # ─────────────────────────────────────
    def _transcribe_whisper(self, audio_path: str) -> TranscriptResult:
        """
        Local fallback using OpenAI Whisper.

        Runs entirely on your machine.
        No API key. No internet. Completely free.
        Used only when AssemblyAI fails.

        Note on word timestamps:
        Whisper returns word-level timestamps when word_timestamps=True.
        Confidence scores are not available from Whisper so we default to 1.0.
        Start and end times are converted from seconds to milliseconds
        to stay consistent with AssemblyAI's format.
        """
        logger.info(f"AssemblyAI failed — falling back to local Whisper: {audio_path}")

        result = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True    # get word-level timestamps
        )

        # Extract word-level data from Whisper segments
        words = []
        for segment in result.get("segments", []):
            for word_obj in segment.get("words", []):
                words.append({
                    "word": word_obj["word"].strip(),
                    "start": word_obj["start"] * 1000,   # convert seconds → ms
                    "end": word_obj["end"] * 1000,
                    "confidence": 1.0                    # Whisper has no confidence score
                })

        # If word timestamps came back empty for any reason
        # fall back to splitting the full text into words
        if not words:
            words = [
                {"word": w, "start": 0, "end": 0, "confidence": 1.0}
                for w in result["text"].split()
            ]

        return TranscriptResult(
            text=result["text"].strip(),
            words=words,
            engine="whisper",
            duration=None,          # Whisper does not return duration
            error=None
        )

    # ─────────────────────────────────────
    # Main method
    # ─────────────────────────────────────
    def transcribe(self, audio_path: str, engine: str = "assemblyai") -> TranscriptResult:
        """
        Main method your route calls.

        Always tries AssemblyAI first.
        If AssemblyAI fails for any reason, automatically falls back to Whisper.

        Args:
            audio_path: path to audio file on disk
            engine: kept for interface consistency but always starts with assemblyai

        Returns:
            TranscriptResult with full text, word timestamps, and engine used
        """
        # Step 1 — try AssemblyAI
        try:
            logger.info("Attempting transcription with AssemblyAI")
            result = self._transcribe_assemblyai(audio_path)
            logger.info("Transcription successful with AssemblyAI")
            return result

        except Exception as e:
            logger.warning(f"AssemblyAI failed: {e} — switching to Whisper fallback")

        # Step 2 — fall back to Whisper
        try:
            result = self._transcribe_whisper(audio_path)
            logger.info("Transcription successful with Whisper fallback")
            return result

        except Exception as e:
            logger.error(f"Whisper fallback also failed: {e}")
            return TranscriptResult(
                text="",
                words=[],
                engine="none",
                duration=None,
                error=f"Both AssemblyAI and Whisper failed. Last error: {str(e)}"
            )


# ─────────────────────────────────────────
# Single instance
# ─────────────────────────────────────────
transcriber = Transcriber()