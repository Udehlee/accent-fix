from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from accent_fix.services import accent_detector, transcriber, error_detector, corrector, output_builder
from accent_fix.db import get_cached_result, set_cached_result, save_transcript_log, create_tables
from accent_fix.db.postgres import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    # Check Redis cache first
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

        # Step 1 — detect accent
        # Fixed: changed detect_accent to detect
        accent_result = accent_detector.detect_accent(audio_path)

        # Step 2 — transcribe
        transcript_result = transcriber.transcribe(audio_path)

        # Step 3 — detect errors
        error_detection_result = error_detector.detect(
            transcript_result,
            accent_result
        )

        # Step 4 — correct errors
        correction_result = corrector.correct(
            transcript_result,
            error_detection_result,
            accent_result
        )

        # Step 5 — build output
        output = output_builder.build(
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

        # Save to database — fixed: pass db session
        db = SessionLocal()
        try:
            save_transcript_log(
                db=db,
                id=str(uuid.uuid4()),
                accent=output.accent,
                accent_confidence=output.accent_confidence,
                engine_used=output.engine_used,
                total_words=output.total_words,
                total_errors_found=output.total_errors_found,
                total_corrections_applied=output.total_corrections_applied
            )
        finally:
            db.close()

        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)