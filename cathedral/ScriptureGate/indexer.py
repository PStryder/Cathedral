"""
ScriptureGate content extraction and indexing.

Extracts searchable text from files and generates embeddings.
"""

import os
import json
from pathlib import Path
from typing import Optional, List
import asyncio

from cathedral.ScriptureGate.storage import get_full_path, read_text, read_file


async def extract_text_content(
    relative_path: str,
    file_type: str,
    mime_type: str = None
) -> Optional[str]:
    """
    Extract searchable text content from a file.

    Args:
        relative_path: Path relative to scripture root
        file_type: Type category (document, image, audio, etc.)
        mime_type: MIME type of file

    Returns:
        Extracted text or None
    """
    full_path = get_full_path(relative_path)
    if not full_path.exists():
        return None

    ext = full_path.suffix.lower()

    # Text-based documents
    if file_type == "document":
        return await _extract_document_text(full_path, ext, mime_type)

    # Images - use vision API for description
    if file_type == "image":
        return await _extract_image_text(full_path)

    # Audio - transcribe
    if file_type == "audio":
        return await _extract_audio_text(full_path)

    # JSON artifacts
    if ext == ".json":
        try:
            content = read_text(relative_path)
            data = json.loads(content)
            # Flatten JSON to searchable text
            return _flatten_json(data)
        except Exception:
            return None

    # Threads - extract message content
    if file_type == "thread":
        return await _extract_thread_text(full_path)

    return None


async def _extract_document_text(path: Path, ext: str, mime_type: str = None) -> Optional[str]:
    """Extract text from document files."""
    try:
        # Plain text formats
        if ext in (".txt", ".md", ".csv", ".html", ".xml"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        # JSON
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return _flatten_json(data)

        # PDF - try to extract with pdfplumber if available
        if ext == ".pdf":
            try:
                import pdfplumber
                text_parts = []
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                return "\n\n".join(text_parts)
            except ImportError:
                # pdfplumber not available, return filename as fallback
                return f"PDF document: {path.name}"

        return None

    except Exception as e:
        print(f"[ScriptureGate] Error extracting text from {path}: {e}")
        return None


async def _extract_image_text(path: Path) -> Optional[str]:
    """Extract description from image using vision API."""
    try:
        from cathedral.StarMirror import describe_image
        description = await describe_image(
            str(path),
            prompt="Describe this image comprehensively. Include: main subjects, colors, composition, text visible, and overall context."
        )
        return description
    except Exception as e:
        print(f"[ScriptureGate] Error describing image {path}: {e}")
        return f"Image: {path.name}"


async def _extract_audio_text(path: Path) -> Optional[str]:
    """Transcribe audio file."""
    try:
        from cathedral.StarMirror import transcribe_audio
        transcription = await transcribe_audio(str(path))
        return transcription
    except Exception as e:
        print(f"[ScriptureGate] Error transcribing audio {path}: {e}")
        return f"Audio: {path.name}"


async def _extract_thread_text(path: Path) -> Optional[str]:
    """Extract text from thread export."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        thread = data.get("thread", [])
        messages = []
        for msg in thread:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            messages.append(f"{role}: {content}")

        return "\n\n".join(messages)

    except Exception as e:
        print(f"[ScriptureGate] Error extracting thread {path}: {e}")
        return None


def _flatten_json(data, prefix: str = "") -> str:
    """Flatten JSON structure to searchable text."""
    parts = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            parts.append(_flatten_json(value, new_prefix))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_prefix = f"{prefix}[{i}]"
            parts.append(_flatten_json(item, new_prefix))
    else:
        if data is not None:
            parts.append(f"{prefix}: {data}" if prefix else str(data))

    return "\n".join(p for p in parts if p)


async def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for text content."""
    if not text or not text.strip():
        return None

    try:
        from cathedral.shared.embeddings import embed_text, is_configured

        if not is_configured():
            return None

        # Truncate if too long
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars]

        embedding = await embed_text(text)
        return embedding

    except Exception as e:
        print(f"[ScriptureGate] Error generating embedding: {e}")
        return None


def build_searchable_text(
    title: str = None,
    description: str = None,
    extracted_text: str = None,
    tags: List[str] = None
) -> str:
    """Combine all text fields for embedding."""
    parts = []

    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    if extracted_text:
        parts.append(extracted_text)

    return "\n\n".join(parts)
