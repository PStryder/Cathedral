"""
Async embedding generation for conversation service.

Uses OpenAI API for generating text embeddings compatible with pgvector.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional

import httpx
from dotenv import load_dotenv

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("ConversationEmbeddings")

# Load .env
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
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

    Args:
        text: Text to embed (will be truncated if > 8000 chars)

    Returns:
        List of floats (embedding vector) or None if failed
    """
    if not OPENAI_API_KEY:
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
            if attempt < EMBEDDING_MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                _log.error(f" HTTP error: {e.response.status_code}")
                return None

        except Exception as e:
            if attempt < EMBEDDING_MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                _log.error(f" Error: {e}")
                return None

    return None


async def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Generate embeddings for multiple texts concurrently.

    Args:
        texts: List of texts to embed

    Returns:
        List of embeddings (None for any that failed)
    """
    tasks = [embed_text(text) for text in texts]
    return await asyncio.gather(*tasks)


async def embed_text_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Generate embeddings for multiple texts in a single API call (batch).

    More efficient than embed_texts for many texts as it uses
    OpenAI's batch embedding endpoint.

    Args:
        texts: List of texts to embed

    Returns:
        List of embeddings (None for all if batch fails)
    """
    if not OPENAI_API_KEY:
        return [None] * len(texts)

    if not texts:
        return []

    # Truncate texts if too long
    truncated = [
        t[:EMBEDDING_MAX_TEXT_LENGTH] if len(t) > EMBEDDING_MAX_TEXT_LENGTH else t
        for t in texts
    ]

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
            if attempt < EMBEDDING_MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                _log.error(f" Batch HTTP error: {e.response.status_code}")
                return [None] * len(texts)

        except Exception as e:
            if attempt < EMBEDDING_MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                _log.error(f" Batch error: {e}")
                return [None] * len(texts)

    return [None] * len(texts)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity (-1 to 1, higher = more similar)
    """
    import math
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
