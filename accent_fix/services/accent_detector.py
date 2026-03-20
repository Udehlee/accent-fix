import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from dataclasses import dataclass
from typing import Optional
import logging


logger = logging.getLogger(__name__)

@dataclass
class AccentResult:
    accent:str
    confidence:float
    all_accents_scores:dict
    error:Optional[str]

class AccentDetector:
    MODEL_NAME = "dima806/english_accents_classification"
    TARGET_SAMPLE_RATE = 16000

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = None
        self.model = None
        self.id2label = None
        self.load_model()


    def load_model(self):
        logger.info("Loading accent model")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_NAME)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

        logger.info(f"Accent model loaded on {self.device}")

    def prepare_audio(self, audio_path: str):
        waveform, sample_rate = torchaudio.load(audio_path)

        #change from stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
     
        #resample audio
        if sample_rate != self.TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        #truncate to 30 seconds
        audio = waveform.squeeze().numpy()
        max_samples = 30 * self.TARGET_SAMPLE_RATE
        return audio[:max_samples]
    
    def detect_accent(self, audio_path:str):
        try:
            audio = self.prepare_audio(audio_path)
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
           
            new_inputs = {}

            for k, v in inputs.items():
             new_inputs[k] = v.to(self.device)

            inputs = new_inputs

            # run inference 
            with torch.no_grad():
                outputs = self.model(**inputs)

            # convert raw numbers to probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.squeeze().cpu().numpy()

            # get the top accent
            top_id = np.argmax(probs)
            accent = self.id2label[top_id]
            confidence = round(float(probs[top_id]), 4)

            # build all scores sorted highest to lowest
            all_scores = dict(
                sorted(
                    {
                        self.id2label[i]: round(float(probs[i]), 4)
                        for i in range(len(probs))
                    }.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

            return AccentResult(
                accent=accent,
                confidence=confidence,
                all_accents_scores=all_scores,
                error=None
            )

        except Exception as e:
            logger.error(f"Accent detection failed: {e}")
            return AccentResult(
                accent="unknown",
                confidence=0.0,
                all_accents_scores={},
                error=str(e)
            )
        

accent_detector = AccentDetector()