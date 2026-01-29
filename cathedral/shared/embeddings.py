"""
Shared embedding utilities for Cathedral.

Conversation embeddings use the MemoryGate conversation embeddings module.
"""

from typing import Optional, List

from cathedral.MemoryGate.conversation import embeddings as conv_embeddings


def is_configured() -> bool:
    """Check if embedding API is configured."""
    return conv_embeddings.is_configured()


async def embed_text(text: str) -> Optional[List[float]]:
    """Generate embedding for a single text."""
    return await conv_embeddings.embed_text(text)


async def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """Generate embeddings for multiple texts concurrently."""
    return await conv_embeddings.embed_texts(texts)


async def embed_text_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Generate embeddings for multiple texts in a single API call."""
    return await conv_embeddings.embed_text_batch(texts)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    return conv_embeddings.cosine_similarity(a, b)


async def close_client():
    """Close the HTTP client."""
    await conv_embeddings.close_client()
