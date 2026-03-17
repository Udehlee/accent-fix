import os
import json
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(
        REDIS_URL,
        decode_responses=True,   
        socket_connect_timeout=2, 
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



# Cache TTL (Time To Live) 
CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours

def generate_cache_key(audio_bytes: bytes) -> str:
    file_hash = hashlib.md5(audio_bytes).hexdigest()
    return f"accentfix:{file_hash}"

def get_cached_result(audio_bytes: bytes) -> Optional[dict]:
    if not REDIS_AVAILABLE:
        return None

    try:
        cache_key = generate_cache_key(audio_bytes)
        cached = redis_client.get(cache_key)

        if cached:
            logger.info(f"Cache hit — returning cached result for key: {cache_key[:20]}...")
            return json.loads(cached)

        logger.info(f"Cache miss — key not found: {cache_key[:20]}...")
        return None

    except Exception as e:
        logger.warning(f"Cache get failed: {e} — proceeding without cache")
        return None

def set_cached_result(audio_bytes: bytes, result: dict) -> None:
    if not REDIS_AVAILABLE:
        return

    try:
        cache_key = generate_cache_key(audio_bytes)
        redis_client.setex(
            name=cache_key,
            time=CACHE_TTL_SECONDS,
            value=json.dumps(result)
        )
        logger.info(f"Result cached for 24 hours — key: {cache_key[:20]}...")

    except Exception as e:
        logger.warning(f"Cache set failed: {e} — result not cached")


def clear_cache() -> None:
    if not REDIS_AVAILABLE:
        return

    try:
        keys = redis_client.keys("accentfix:")
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Cache cleared — {len(keys)} keys deleted")
        else:
            logger.info("Cache already empty")

    except Exception as e:
        logger.warning(f"Cache clear failed: {e}")