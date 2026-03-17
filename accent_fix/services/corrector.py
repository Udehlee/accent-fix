import os
import logging
from dataclasses import dataclass
from typing import Optional
from groq import Groq
import re

logger = logging.getLogger(__name__)

@dataclass
class Correction:
    index: int            
    corrected_word: str     
    context: str            
    confidence: float       
    explanation: str        

@dataclass
class CorrectionResult:
    original_text: str        
    corrected_text: str         
    corrections: list           
    total_corrections: int      
    accent: str                
    error: Optional[str] = None 


class Corrector:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    AUTO_CORRECT_THRESHOLD = 0.75

    def __init__(self):
        if self.GROQ_API_KEY:
            self.groq_client = Groq(api_key=self.GROQ_API_KEY)
            logger.info("Groq client initialized for context validation")
        else:
            self.groq_client = None
            logger.warning("GROQ_API_KEY not found — corrections will use T5 prediction directly without LLM validation")

  
    def validate_with_llm(
        self,
        original_word: str,
        suggested_word: str,
        context: str,
        accent: str
    ) -> tuple[bool, float, str]:
        
        if not self.groq_client:
            # No Groq available — trust T5 prediction directly
            logger.warning("No Groq client — applying T5 prediction without validation")
            return True, 0.80, "T5 prediction applied without LLM validation"

        prompt = f"""You are a transcription correction assistant specializing in accented English speech.

A speech-to-text system transcribed audio from a speaker with a {accent} accent and likely made an error.

Sentence context: "{context}"
Word that may be wrong: "{original_word}"
Suggested correction: "{suggested_word}"

Your job:
1. Read the sentence context carefully
2. Decide if replacing "{original_word}" with "{suggested_word}" makes the sentence correct and natural
3. Respond in this exact format with nothing else:
   APPLY: yes or no
   CONFIDENCE: a number between 0.0 and 1.0
   REASON: one short sentence

Only respond with those three lines. No extra text."""

        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,     # low temperature = deterministic responses
                max_tokens=80
            )

            raw = response.choices[0].message.content.strip()
            lines = raw.split("\n")

            apply = False
            confidence = 0.0
            explanation = "No explanation provided"

            for line in lines:
                line = line.strip()
                if line.startswith("APPLY:"):
                    apply = "yes" in line.lower()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        confidence = 0.7
                elif line.startswith("REASON:"):
                    explanation = line.split(":", 1)[1].strip()

            logger.debug(
                f"Groq validation: '{original_word}' → '{suggested_word}' | "
                f"apply={apply} confidence={confidence:.2f}"
            )

            return apply, round(confidence, 4), explanation

        except Exception as e:
            logger.warning(f"Groq validation failed: {e} — applying T5 prediction directly")
            return True, 0.75, f"LLM validation failed ({e}) — T5 prediction applied"

 
    # Rebuild Transcript
    def _rebuild_transcript(self, original_text: str, corrections: list) -> str:   
        corrected_text = original_text

        for correction in corrections:
            original = correction.original_word
            corrected = correction.corrected_word

            if original[0].isupper():
                corrected = corrected.capitalize()

            # Whole word replacement only — \b is word boundary
            pattern = r'\b' + re.escape(original) + r'\b'
            corrected_text = re.sub(
                pattern,
                corrected,
                corrected_text,
                count=1,
                flags=re.IGNORECASE
            )

        return corrected_text

    def correct(
        self,
        transcript_result,        
        error_detection_result,  
        accent_result            
    ) -> CorrectionResult:
        try:
            accent = accent_result.accent
            original_text = transcript_result.text
            errors = error_detection_result.errors

            if not errors:
                logger.info("No errors to correct — returning original transcript")
                return CorrectionResult(
                    original_text=original_text,
                    corrected_text=original_text,
                    corrections=[],
                    total_corrections=0,
                    accent=accent,
                    error=None
                )

            logger.info(f"Processing {len(errors)} flagged errors for accent: {accent}")

            applied_corrections = []

            for detected_error in errors:
                original_word = detected_error.original_word
                context = detected_error.context

                # T5 already predicted the correct word in the error detector
                # We just read it directly — no lookup needed
                suggested_word = detected_error.predicted_word

                if not suggested_word or suggested_word == original_word.lower():
                    # T5 had no useful prediction — skip
                    logger.debug(f"No useful T5 prediction for: '{original_word}' — skipping")
                    continue

                # Validate with Groq that the correction makes sense in context
                should_apply, confidence, explanation = self._validate_with_llm(
                    original_word=original_word,
                    suggested_word=suggested_word,
                    context=context,
                    accent=accent
                )

                # Apply only if Groq approved and confidence is above threshold
                if should_apply and confidence >= self.AUTO_CORRECT_THRESHOLD:
                    applied_corrections.append(Correction(
                        index=detected_error.index,
                        original_word=original_word,
                        corrected_word=suggested_word,
                        context=context,
                        confidence=confidence,
                        explanation=explanation
                    ))
                    logger.info(
                        f"Correction applied: '{original_word}' → '{suggested_word}' "
                        f"({confidence:.0%} confidence)"
                    )
                else:
                    logger.info(
                        f"Correction rejected: '{original_word}' → '{suggested_word}' "
                        f"(apply={should_apply}, confidence={confidence:.0%})"
                    )

            # Rebuild full transcript with all accepted corrections applied
            corrected_text = self._rebuild_transcript(original_text, applied_corrections)

            logger.info(
                f"Correction complete — {len(applied_corrections)} corrections applied "
                f"out of {len(errors)} flagged errors"
            )

            return CorrectionResult(
                original_text=original_text,
                corrected_text=corrected_text,
                corrections=applied_corrections,
                total_corrections=len(applied_corrections),
                accent=accent,
                error=None
            )

        except Exception as e:
            logger.error(f"Correction failed: {e}")
            return CorrectionResult(
                original_text=transcript_result.text if transcript_result else "",
                corrected_text=transcript_result.text if transcript_result else "",
                corrections=[],
                total_corrections=0,
                accent=accent_result.accent if accent_result else "unknown",
                error=str(e)
            )

corrector = Corrector()