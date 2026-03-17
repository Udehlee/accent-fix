import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from services import (accent_detector, transcriber, error_detector,corrector, output)
from db import (get_cached_result, set_cached_result, save_transcript_log, create_tables )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    logger.info("Database tables ready")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "AccentFix is running"}


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    contents = await file.read()

    # check If same audio was processed before return instantly
    cached = get_cached_result(contents)
    if cached:
        logger.info("Returning cached result")
        return cached

    # Save to temp folder
    file_extension = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    audio_path = os.path.join(TEMP_DIR, unique_filename)

    try:
        with open(audio_path, "wb") as f:
            f.write(contents)

        accent_result = accent_detector.detect(audio_path)
        transcript_result = transcriber.transcribe(audio_path)
        error_detection_result = error_detector.detect(transcript_result, accent_result)
        correction_result = corrector.correct(transcript_result, error_detection_result, accent_result)

        # build output
        output = output.build(
            accent_result,
            transcript_result,
            error_detection_result,
            correction_result
        )

        # Build response dict
        result = {
            "accent": output.accent,
            "accent_confidence": output.accent_confidence,
            "engine_used": output.engine_used,
            "original_transcript": output.original_transcript,
            "corrected_transcript": output.corrected_transcript,
            "total_corrections": output.total_corrections_applied,
            "highlights": [
                {
                    "original_word": h.original_word,
                    "corrected_word": h.corrected_word,
                    "confidence": h.confidence,
                    "explanation": h.explanation
                }
                for h in output.highlights
            ],
            "summary": output.summary
        }

        # Save to Redis cache 
        set_cached_result(contents, result)

        # ── Save to PostgreSQL ──
        save_transcript_log(
            id=str(uuid.uuid4()),
            accent=output.accent,
            accent_confidence=output.accent_confidence,
            engine_used=output.engine_used,
            total_words=output.total_words,
            total_errors_found=output.total_errors_found,
            total_corrections_applied=output.total_corrections_applied
        )

        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)




