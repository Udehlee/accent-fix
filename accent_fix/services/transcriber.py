import assemblyai as aai
import whisper
import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)



@dataclass
class TranscriptResult:
    text: str               
    words: list               
    engine: str                
    duration: Optional[float]  
    error: Optional[str]       


class Transcriber:
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    WHISPER_MODEL_SIZE = "base"

    def __init__(self):
        if self.ASSEMBLYAI_API_KEY:
            aai.settings.api_key = self.ASSEMBLYAI_API_KEY
            logger.info("AssemblyAI configured")
        else:
            logger.warning("ASSEMBLYAI_API_KEY not found — will fall back to Whisper")

        # Load Whisper model once into memory when app starts
        logger.info(f"Loading Whisper model: {self.WHISPER_MODEL_SIZE}")
        self.whisper_model = whisper.load_model(self.WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded and ready as fallback")


    def transcribe_assemblyai(self, audio_path: str) -> TranscriptResult:
    
        logger.info(f"Transcribing with AssemblyAI: {audio_path}")

        config = aai.TranscriptionConfig(
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
                    "start": word.start,        
                    "end": word.end,          
                    "confidence": round(word.confidence, 4)
                })

        return TranscriptResult(
            text=transcript.text or "",
            words=words,
            engine="assemblyai",
            duration=transcript.audio_duration,
            error=None
        )

    def transcribe_whisper(self, audio_path: str) -> TranscriptResult:
        logger.info(f"AssemblyAI failed — falling back to local Whisper: {audio_path}")

        result = self.whisper_model.transcribe(
                 audio_path,
                 word_timestamps=True,
                 language="en",
                 task="transcribe"
)

        # Extract word-level data from Whisper segments
        words = []
        for segment in result.get("segments", []):
            for word_obj in segment.get("words", []):
                words.append({
                    "word": word_obj["word"].strip(),
                    "start": word_obj["start"] * 1000,   
                    "end": word_obj["end"] * 1000,
                    "confidence": 1.0                    
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
            duration=None,         
            error=None
        )

        

    def transcribe(self, audio_path: str, engine: str = "assemblyai") -> TranscriptResult:
        try:
            logger.info("Attempting transcription with AssemblyAI")
            result = self.transcribe_assemblyai(audio_path)
            logger.info("Transcription successful with AssemblyAI")
            return result

        except Exception as e:
            logger.warning(f"AssemblyAI failed: {e} — switching to Whisper fallback")

        try:
            result = self.transcribe_whisper(audio_path)
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



transcriber = Transcriber()