import os
import logging
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

logger = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL")

# Engine — the connection to the database
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      
    pool_size=5,             
    max_overflow=10         
)

# create database sessions
# Each request gets its own session
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# all your database models inherit from this
Base = declarative_base()

# feedback Log Table
class FeedbackLog(Base):
    __tablename__ = "feedback_logs"

    id = Column(String, primary_key=True)          
    original_word = Column(String, nullable=False)  
    corrected_word = Column(String, nullable=False) 
    is_correct = Column(Boolean, nullable=False)    
    accent = Column(String, nullable=False)         
    context = Column(Text, nullable=True)           
    confidence = Column(Float, nullable=True)       
    engine_used = Column(String, nullable=True)     
    created_at = Column(DateTime, default=datetime.utcnow)


# Transcript Log Table
class TranscriptLog(Base):
    __tablename__ = "transcript_logs"

    id = Column(String, primary_key=True)
    accent = Column(String, nullable=False)
    accent_confidence = Column(Float, nullable=True)
    engine_used = Column(String, nullable=True)
    total_words = Column(Float, nullable=True)
    total_errors_found = Column(Float, nullable=True)
    total_corrections_applied = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Create Tables
def create_tables():
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


# Get Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Save Feedback
def save_feedback(
    db,
    id: str,
    original_word: str,
    corrected_word: str,
    is_correct: bool,
    accent: str,
    context: str = None,
    confidence: float = None,
    engine_used: str = None
):
 
    feedback = FeedbackLog(
        id=id,
        original_word=original_word,
        corrected_word=corrected_word,
        is_correct=is_correct,
        accent=accent,
        context=context,
        confidence=confidence,
        engine_used=engine_used
    )
    db.add(feedback)
    db.commit()
    logger.info(f"Feedback saved: '{original_word}' → '{corrected_word}' | correct: {is_correct}")


# Save Transcript Log
def save_transcript_log(
    db,
    id: str,
    accent: str,
    accent_confidence: float,
    engine_used: str,
    total_words: int,
    total_errors_found: int,
    total_corrections_applied: int
):
    log = TranscriptLog(
        id=id,
        accent=accent,
        accent_confidence=accent_confidence,
        engine_used=engine_used,
        total_words=total_words,
        total_errors_found=total_errors_found,
        total_corrections_applied=total_corrections_applied
    )
    db.add(log)
    db.commit()
    logger.info(f"Transcript log saved: {accent} | {total_corrections_applied} corrections")