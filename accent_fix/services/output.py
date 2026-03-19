import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CorrectionHighlight:
    index: int             
    original_word: str     
    corrected_word: str    
    confidence: float      
    explanation: str      


@dataclass
class OutputResult:
    accent: str                     
    accent_confidence: float         
    engine_used: str                 
    original_transcript: str         
    corrected_transcript: str        
    total_words: int                
    total_errors_found: int          
    total_corrections_applied: int   
    correction_rate: float        
    highlights: list                
    summary: str                     
    error: Optional[str] = None    


class Output:
    def build_highlights(self, corrections: list) -> list:
        highlights = []

        for correction in corrections:
            highlights.append(CorrectionHighlight(
                index=correction.index,
                original_word=correction.original_word,
                corrected_word=correction.corrected_word,
                confidence=correction.confidence,
                explanation=correction.explanation
            ))

        return highlights

    def build_summary(self,accent: str,accent_confidence: float,engine_used: str,total_words: int, total_errors_found: int,
        total_corrections_applied: int
    ) -> str:
        """
        Builds a human readable summary of what the pipeline did

        This is what the user reads to understand what happened
        to their audio in plain English.

        Args:
            accent: detected accent label
            accent_confidence: how confident the accent detection was
            engine_used: which transcription engine was used
            total_words: total words in transcript
            total_errors_found: how many errors were detected
            total_corrections_applied: how many corrections were applied

        Returns:
            human readable summary string
        """
        summary_parts = []

        summary_parts.append(
            f"Detected accent: {accent} ({accent_confidence:.0%} confidence)."
        )

        # Transcription engine used
        if engine_used == "assemblyai":
            summary_parts.append("Transcribed using AssemblyAI.")
        elif engine_used == "whisper":
            summary_parts.append("Transcribed using Whisper (local fallback).")
        else:
            summary_parts.append(f"Transcribed using {engine_used}.")

        # Correction summary
        if total_errors_found == 0:
            summary_parts.append(
                f"No transcription errors detected across {total_words} words."
            )
        elif total_corrections_applied == 0:
            summary_parts.append(
                f"{total_errors_found} potential errors were detected across "
                f"{total_words} words but none passed the confidence threshold "
                f"to be corrected automatically."
            )
        elif total_corrections_applied == total_errors_found:
            summary_parts.append(
                f"{total_corrections_applied} transcription "
                f"{'error was' if total_corrections_applied == 1 else 'errors were'} "
                f"detected and corrected across {total_words} words."
            )
        else:
            summary_parts.append(
                f"{total_errors_found} potential errors were detected across "
                f"{total_words} words. {total_corrections_applied} "
                f"{'was' if total_corrections_applied == 1 else 'were'} "
                f"corrected after context validation."
            )

        return " ".join(summary_parts)

    def calculate_correction_rate(self,total_corrections: int,total_words: int) -> float:
        if total_words == 0:
            return 0.0
        return round(total_corrections / total_words, 4)

    def build(self, accent_result, transcript_result, error_detection_result, correction_result) -> OutputResult:
        """
        Takes all four pipeline results and assembles them into
        one clean structured output ready to send to the user.
        """
        try:
            logger.info("Building final output")

            accent = accent_result.accent
            accent_confidence = accent_result.confidence
            engine_used = transcript_result.engine
            total_words = error_detection_result.total_words
            total_errors_found = error_detection_result.error_count
            total_corrections_applied = correction_result.total_corrections
            original_transcript = correction_result.original_text
            corrected_transcript = correction_result.corrected_text

            # Build highlights from corrections
            highlights = self.build_highlights(correction_result.corrections)

            # Calculate correction rate
            correction_rate = self._calculate_correction_rate(
                total_corrections_applied,
                total_words
            )

            # Build human readable summary
            summary = self.build_summary(
                accent=accent,
                accent_confidence=accent_confidence,
                engine_used=engine_used,
                total_words=total_words,
                total_errors_found=total_errors_found,
                total_corrections_applied=total_corrections_applied
            )

            logger.info(f"Output built — {total_corrections_applied} corrections across {total_words} words")

            return OutputResult(
                accent=accent,
                accent_confidence=accent_confidence,
                engine_used=engine_used,
                original_transcript=original_transcript,
                corrected_transcript=corrected_transcript,
                total_words=total_words,
                total_errors_found=total_errors_found,
                total_corrections_applied=total_corrections_applied,
                correction_rate=correction_rate,
                highlights=highlights,
                summary=summary,
                error=None
            )

        except Exception as e:
            logger.error(f"Output builder failed: {e}")
            return OutputResult(
                accent=accent_result.accent if accent_result else "unknown",
                accent_confidence=0.0,
                engine_used="unknown",
                original_transcript="",
                corrected_transcript="",
                total_words=0,
                total_errors_found=0,
                total_corrections_applied=0,
                correction_rate=0.0,
                highlights=[],
                summary="Output builder failed.",
                error=str(e)
            )


output_builder = Output()