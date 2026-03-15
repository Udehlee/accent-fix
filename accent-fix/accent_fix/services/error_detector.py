import logging
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DetectedError:
    index: int          
    original_word: str  
    context: str         
    predicted_word: str 
    confidence: float    


@dataclass
class ErrorDetectionResult:
    errors: list                
    total_words: int             
    error_count: int            
    error_rate: float            
    accent: str                 
    error: Optional[str] = None 

class ErrorDetector:
    T5_MODEL_NAME = "t5-base"
    DIVERGENCE_THRESHOLD = 0.6

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load_T5_model()

    def load_T5_model(self):
       
        logger.info(f"Loading T5 error detection model: {self.T5_MODEL_NAME}")

        self.tokenizer = T5Tokenizer.from_pretrained(self.T5_MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(self.T5_MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"T5 model loaded on {self.device}")

    def get_context(self, words: list, index: int, window: int = 5) -> str:
        start = max(0, index - window)
        end = min(len(words), index + window + 1)

        context_words = []
        for i in range(start, end):
            if i == index:
                context_words.append("<extra_id_0>")
            else:
                context_words.append(words[i]["word"])

        return " ".join(context_words)

    def build_t5_input(self, context: str, accent: str) -> str:
        """
        Builds the full input string T5 expects.

        Returns:
            formatted T5 input string
        """
        return f"detect error accent: {accent} context: {context}"

    def predict_word(self, t5_input: str) -> str:
        """
        Runs T5 and returns its predicted word for the masked position.
        """
        input_ids = self.tokenizer(
            t5_input,
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=10,
                num_beams=4,        
                early_stopping=True
            )

        predicted = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).lower().strip()

        return predicted

    def calculate_divergence(self, original: str, predicted: str) -> float:
        """
        Measures how different the predicted word is from the original.

        Uses Jaccard character similarity:
        - Compare the set of characters in both words

        Returns:
            divergence score between 0.0 and 1.0
        """
        if not original or not predicted:
            return 0.0

        # If T5 predicted the exact same word — definitely not an error
        if original == predicted:
            return 0.0

        original_chars = set(original)
        predicted_chars = set(predicted)
        intersection = original_chars & predicted_chars
        union = original_chars | predicted_chars

        similarity = len(intersection) / len(union) if union else 1.0
        divergence = 1.0 - similarity

        return round(divergence, 4)

    def detect(self,transcript_result, accent_result) -> ErrorDetectionResult:
        """
        Scans every word in the transcript through T5
        Words where T5's prediction diverges significantly are flagged
        """
        try:
            accent = accent_result.accent
            words = transcript_result.words

            if not words:
                return ErrorDetectionResult(
                    errors=[],
                    total_words=0,
                    error_count=0,
                    error_rate=0.0,
                    accent=accent,
                    error="No words found in transcript"
                )

            logger.info(f"Scanning {len(words)} words with T5 | accent: {accent}")

            errors = []

            for i, word_obj in enumerate(words):
                word = word_obj.get("word", "").strip()

                if not word:
                    continue

                # build context with word masked
                context = self.get_context(words, i)

                # build T5 input
                t5_input = self.build_t5_input(context, accent)

                # get T5 prediction
                predicted_word = self.predict_word(t5_input)

                # measure divergence between transcript and prediction
                original_clean = word.lower().strip(".,!?;:")
                divergence = self.calculate_divergence(original_clean, predicted_word)

                # flag if divergence is above threshold
                if divergence >= self.DIVERGENCE_THRESHOLD:
                    logger.debug(
                        f"Flagged: '{word}' → T5 predicted '{predicted_word}' "
                        f"(divergence: {divergence:.2f})"
                    )
                    errors.append(DetectedError(
                        index=i,
                        original_word=word,
                        context=" ".join([w["word"] for w in words[
                            max(0, i - 4): min(len(words), i + 5)
                        ]]),
                        predicted_word=predicted_word,
                        confidence=round(divergence, 4)
                    ))

            error_rate = round(len(errors) / len(words), 4) if words else 0.0

            logger.info(
                f"Detection complete — {len(errors)} errors "
                f"out of {len(words)} words ({error_rate:.1%})"
            )

            return ErrorDetectionResult(
                errors=errors,
                total_words=len(words),
                error_count=len(errors),
                error_rate=error_rate,
                accent=accent,
                error=None
            )

        except Exception as e:
            logger.error(f"Error detection failed: {e}")
            return ErrorDetectionResult(
                errors=[],
                total_words=0,
                error_count=0,
                error_rate=0.0,
                accent=accent_result.accent if accent_result else "unknown",
                error=str(e)
            )


error_detector = ErrorDetector()