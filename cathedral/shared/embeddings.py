"""
Shared embedding utilities for Cathedral.

Provides backend-agnostic embedding generation that works with either
Loom or MemoryGate conversation backend.
"""

from typing import Optional, List

from cathedral.Memory import CONVERSATION_BACKEND, ConversationBackend


def _get_loom_embeddings():
    """Get Loom embeddings module."""
    from loom import embeddings as loom_emb
    return loom_emb


def _get_mg_embeddings():
    """Get MemoryGate conversation embeddings module."""
    from cathedral.MemoryGate.conversation import embeddings as mg_emb
    return mg_emb


def is_configured() -> bool:
    """Check if embedding API is configured."""
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        return _get_loom_embeddings().is_configured()
    else:
        return _get_mg_embeddings().is_configured()


async def embed_text(text: str) -> Optional[List[float]]:
    """Generate embedding for a single text."""
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        return await _get_loom_embeddings().embed_text(text)
    else:
        return await _get_mg_embeddings().embed_text(text)


async def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """Generate embeddings for multiple texts concurrently."""
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        return await _get_loom_embeddings().embed_texts(texts)
    else:
        return await _get_mg_embeddings().embed_texts(texts)


async def embed_text_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Generate embeddings for multiple texts in a single API call."""
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        return await _get_loom_embeddings().embed_text_batch(texts)
    else:
        return await _get_mg_embeddings().embed_text_batch(texts)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        return _get_loom_embeddings().cosine_similarity(a, b)
    else:
        return _get_mg_embeddings().cosine_similarity(a, b)


async def close_client():
    """Close the HTTP client."""
    if CONVERSATION_BACKEND == ConversationBackend.LOOM:
        await _get_loom_embeddings().close_client()
    else:
        await _get_mg_embeddings().close_client()
