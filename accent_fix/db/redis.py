import os
import json
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Redis Setup
# ─────────────────────────────────────────
# We try to import and connect to Redis
# If Redis is not available the cache silently
# does nothing — the app keeps working without it
try:
    import redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(
        REDIS_URL,
        decode_responses=True,    # return strings not bytes
        socket_connect_timeout=2, # fail fast if Redis is not running
        socket_timeout=2
    )
    # Test the connection
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connected successfully")

except Exception as e:
    redis_client = None
    REDIS_AVAILABLE = False
    logger.warning(f"Redis not available: {e} — caching disabled, app will continue without it")


# ─────────────────────────────────────────
# Cache TTL (Time To Live)
# ─────────────────────────────────────────
# How long a cached result stays valid
# After this time the cache expires and
# the audio is reprocessed fresh
CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


# ─────────────────────────────────────────
# Generate Cache Key
# ─────────────────────────────────────────
def _generate_cache_key(audio_bytes: bytes) -> str:
    """
    Generates a unique cache key for an audio file.

    Uses MD5 hash of the audio file contents.
    Same audio file always produces the same key.
    Different audio files produce different keys.

    This means if someone uploads the exact same audio file twice
    we can return the cached result instantly without reprocessing.

    Args:
        audio_bytes: raw bytes of the audio file

    Returns:
        cache key string like "accentfix:a3f9b2c1d4e5..."
    """
    file_hash = hashlib.md5(audio_bytes).hexdigest()
    return f"accentfix:{file_hash}"


# ─────────────────────────────────────────
# Get From Cache
# ─────────────────────────────────────────
def get_cached_result(audio_bytes: bytes) -> Optional[dict]:
    """
    Checks if this audio file was already processed.

    If the same audio was processed before and the cache
    has not expired, returns the cached result instantly.
    Otherwise returns None and the pipeline runs normally.

    Args:
        audio_bytes: raw bytes of the audio file

    Returns:
        cached result dict or None if not in cache
    """
    if not REDIS_AVAILABLE:
        return None

    try:
        cache_key = _generate_cache_key(audio_bytes)
        cached = redis_client.get(cache_key)

        if cached:
            logger.info(f"Cache hit — returning cached result for key: {cache_key[:20]}...")
            return json.loads(cached)

        logger.info(f"Cache miss — key not found: {cache_key[:20]}...")
        return None

    except Exception as e:
        logger.warning(f"Cache get failed: {e} — proceeding without cache")
        return None


# ─────────────────────────────────────────
# Set In Cache
# ─────────────────────────────────────────
def set_cached_result(audio_bytes: bytes, result: dict) -> None:
    """
    Saves a pipeline result to the cache.

    Called after every successful pipeline run.
    Next time the same audio is uploaded the result
    is returned from cache without reprocessing.

    Args:
        audio_bytes: raw bytes of the audio file
        result: the pipeline output dict to cache
    """
    if not REDIS_AVAILABLE:
        return

    try:
        cache_key = _generate_cache_key(audio_bytes)
        redis_client.setex(
            name=cache_key,
            time=CACHE_TTL_SECONDS,
            value=json.dumps(result)
        )
        logger.info(f"Result cached for 24 hours — key: {cache_key[:20]}...")

    except Exception as e:
        logger.warning(f"Cache set failed: {e} — result not cached")


# ─────────────────────────────────────────
# Clear Cache (utility)
# ─────────────────────────────────────────
def clear_cache() -> None:
    """
    Clears all AccentFix cached results.
    Useful during development or model updates.
    """
    if not REDIS_AVAILABLE:
        return

    try:
        keys = redis_client.keys("accentfix:*")
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cache cleared — {len(keys)} keys deleted")
        else:
            logger.info("Cache already empty")

    except Exception as e:
        logger.warning(f"Cache clear failed: {e}")