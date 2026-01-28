"""
Async embedding generation using OpenAI API.
"""

import os
import json
import asyncio
from typing import List, Optional
import httpx
from dotenv import load_dotenv
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_API_URL = "https://api.openai.com/v1/embeddings"
EMBEDDING_TIMEOUT = 30.0
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_MAX_TEXT_LENGTH = 8000

# Module-level async client (reused)
_http_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    """Get or create the async HTTP client."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT)
    return _http_client


async def close_client():
    """Close the HTTP client."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


def is_configured() -> bool:
    """Check if embedding API is configured."""
    return OPENAI_API_KEY is not None


async def embed_text(text: str) -> Optional[List[float]]:
    """
    Generate embedding for a single text using OpenAI API.
    Returns None if embedding fails.
    """
    if not OPENAI_API_KEY:
        print("[Embeddings] OPENAI_API_KEY not configured")
        return None

    # Truncate text if too long
    if len(text) > EMBEDDING_MAX_TEXT_LENGTH:
        text = text[:EMBEDDING_MAX_TEXT_LENGTH]

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": text,
        "model": EMBEDDING_MODEL
    }

    client = _get_client()

    for attempt in range(EMBEDDING_MAX_RETRIES):
        try:
            response = await client.post(
                EMBEDDING_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            embedding = data["data"][0]["embedding"]
            return embedding

        except httpx.HTTPStatusError as e:
            print(f"[Embeddings] HTTP error (attempt {attempt + 1}): {e.response.status_code}")
            if attempt < EMBEDDING_MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                return None

        except Exception as e:
            print(f"[Embeddings] Error (attempt {attempt + 1}): {e}")
            if attempt < EMBEDDING_MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                return None

    return None


async def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Generate embeddings for multiple texts concurrently.
    Returns a list of embeddings (None for any that failed).
    """
    tasks = [embed_text(text) for text in texts]
    return await asyncio.gather(*tasks)


async def embed_text_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Generate embeddings for multiple texts in a single API call (batch).
    More efficient for many texts.
    """
    if not OPENAI_API_KEY:
        print("[Embeddings] OPENAI_API_KEY not configured")
        return [None] * len(texts)

    if not texts:
        return []

    # Truncate texts if too long
    truncated = [t[:EMBEDDING_MAX_TEXT_LENGTH] if len(t) > EMBEDDING_MAX_TEXT_LENGTH else t for t in texts]

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": truncated,
        "model": EMBEDDING_MODEL
    }

    client = _get_client()

    for attempt in range(EMBEDDING_MAX_RETRIES):
        try:
            response = await client.post(
                EMBEDDING_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            # OpenAI returns embeddings in same order as input
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings

        except httpx.HTTPStatusError as e:
            print(f"[Embeddings] Batch HTTP error (attempt {attempt + 1}): {e.response.status_code}")
            if attempt < EMBEDDING_MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                return [None] * len(texts)

        except Exception as e:
            print(f"[Embeddings] Batch error (attempt {attempt + 1}): {e}")
            if attempt < EMBEDDING_MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                return [None] * len(texts)

    return [None] * len(texts)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import math
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
