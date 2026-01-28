"""
MediaGate - Multi-modal content handling for StarMirror.

Supports images (file, URL, base64) and audio (transcription).
Builds message content arrays compatible with OpenAI/OpenRouter vision APIs.
"""

import os
import base64
import httpx
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class MediaType(Enum):
    """Supported media types."""
    IMAGE = "image"
    AUDIO = "audio"


class ImageDetail(Enum):
    """Image detail level for vision models."""
    LOW = "low"      # 512px, faster, fewer tokens
    HIGH = "high"    # Full resolution, more tokens
    AUTO = "auto"    # Model decides


@dataclass
class MediaItem:
    """Represents a media item (image or audio)."""
    media_type: MediaType
    data: str  # base64 data or URL
    mime_type: str
    is_url: bool = False
    detail: ImageDetail = ImageDetail.AUTO
    source_path: Optional[str] = None

    def to_content_part(self) -> dict:
        """Convert to OpenAI message content part format."""
        if self.media_type == MediaType.IMAGE:
            if self.is_url:
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": self.data,
                        "detail": self.detail.value
                    }
                }
            else:
                data_url = f"data:{self.mime_type};base64,{self.data}"
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": self.detail.value
                    }
                }
        elif self.media_type == MediaType.AUDIO:
            # Audio is typically transcribed, not sent directly
            # Return as input_audio for models that support it
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": self.data,
                    "format": self._audio_format()
                }
            }
        return {}

    def _audio_format(self) -> str:
        """Get audio format from mime type."""
        formats = {
            "audio/wav": "wav",
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "audio/webm": "webm",
            "audio/ogg": "ogg",
        }
        return formats.get(self.mime_type, "wav")


# MIME type detection
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".m4a": "audio/mp4",
}


def detect_mime_type(path_or_url: str) -> str:
    """Detect MIME type from file extension."""
    path = Path(path_or_url.split("?")[0])  # Strip query params for URLs
    ext = path.suffix.lower()
    return MIME_TYPES.get(ext, "application/octet-stream")


def is_url(path: str) -> bool:
    """Check if path is a URL."""
    return path.startswith(("http://", "https://", "data:"))


def load_image(
    source: str,
    detail: ImageDetail = ImageDetail.AUTO,
    max_size_mb: float = 20.0
) -> MediaItem:
    """
    Load an image from file path, URL, or base64 string.

    Args:
        source: File path, URL, or base64 string
        detail: Image detail level for vision models
        max_size_mb: Maximum file size in MB

    Returns:
        MediaItem ready for message construction
    """
    # Already base64
    if source.startswith("data:"):
        # Parse data URL
        parts = source.split(",", 1)
        mime_part = parts[0].replace("data:", "").replace(";base64", "")
        return MediaItem(
            media_type=MediaType.IMAGE,
            data=parts[1] if len(parts) > 1 else "",
            mime_type=mime_part,
            is_url=False,
            detail=detail
        )

    # URL - pass through (let API fetch it)
    if is_url(source):
        mime_type = detect_mime_type(source)
        return MediaItem(
            media_type=MediaType.IMAGE,
            data=source,
            mime_type=mime_type,
            is_url=True,
            detail=detail,
            source_path=source
        )

    # File path - load and encode
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {source}")

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Image too large: {size_mb:.1f}MB > {max_size_mb}MB limit")

    mime_type = detect_mime_type(source)
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return MediaItem(
        media_type=MediaType.IMAGE,
        data=data,
        mime_type=mime_type,
        is_url=False,
        detail=detail,
        source_path=str(path)
    )


def load_audio(
    source: str,
    max_size_mb: float = 25.0
) -> MediaItem:
    """
    Load audio from file path or base64 string.

    Args:
        source: File path or base64 string
        max_size_mb: Maximum file size in MB

    Returns:
        MediaItem ready for transcription or direct input
    """
    # Already base64
    if source.startswith("data:"):
        parts = source.split(",", 1)
        mime_part = parts[0].replace("data:", "").replace(";base64", "")
        return MediaItem(
            media_type=MediaType.AUDIO,
            data=parts[1] if len(parts) > 1 else "",
            mime_type=mime_part,
            is_url=False
        )

    # File path
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {source}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Audio too large: {size_mb:.1f}MB > {max_size_mb}MB limit")

    mime_type = detect_mime_type(source)
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return MediaItem(
        media_type=MediaType.AUDIO,
        data=data,
        mime_type=mime_type,
        is_url=False,
        source_path=str(path)
    )


@dataclass
class ContentBuilder:
    """
    Builder for multi-modal message content.

    Usage:
        content = (ContentBuilder()
            .text("What's in this image?")
            .image("path/to/image.jpg")
            .build())
    """
    parts: List[Dict[str, Any]] = field(default_factory=list)

    def text(self, text: str) -> "ContentBuilder":
        """Add text content."""
        self.parts.append({"type": "text", "text": text})
        return self

    def image(
        self,
        source: str,
        detail: ImageDetail = ImageDetail.AUTO
    ) -> "ContentBuilder":
        """Add image from file, URL, or base64."""
        media = load_image(source, detail)
        self.parts.append(media.to_content_part())
        return self

    def image_item(self, item: MediaItem) -> "ContentBuilder":
        """Add pre-loaded MediaItem."""
        self.parts.append(item.to_content_part())
        return self

    def audio(self, source: str) -> "ContentBuilder":
        """Add audio from file or base64."""
        media = load_audio(source)
        self.parts.append(media.to_content_part())
        return self

    def audio_item(self, item: MediaItem) -> "ContentBuilder":
        """Add pre-loaded audio MediaItem."""
        self.parts.append(item.to_content_part())
        return self

    def build(self) -> List[Dict[str, Any]]:
        """Build the content array."""
        return self.parts

    def build_message(self, role: str = "user") -> Dict[str, Any]:
        """Build a complete message dict."""
        return {"role": role, "content": self.parts}


def build_multimodal_message(
    text: str,
    images: List[str] = None,
    audio: List[str] = None,
    role: str = "user",
    image_detail: ImageDetail = ImageDetail.AUTO
) -> Dict[str, Any]:
    """
    Convenience function to build a multi-modal message.

    Args:
        text: Text content
        images: List of image paths/URLs
        audio: List of audio paths
        role: Message role
        image_detail: Detail level for images

    Returns:
        Message dict ready for API
    """
    builder = ContentBuilder()

    # Add text first
    if text:
        builder.text(text)

    # Add images
    if images:
        for img in images:
            builder.image(img, image_detail)

    # Add audio
    if audio:
        for aud in audio:
            builder.audio(aud)

    return builder.build_message(role)


def is_multimodal_content(content: Any) -> bool:
    """Check if content is multi-modal (list of parts) vs plain text."""
    return isinstance(content, list)


def extract_text_from_content(content: Any) -> str:
    """Extract text from content (handles both plain and multi-modal)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if p.get("type") == "text"]
        return " ".join(texts)
    return ""


# Vision-capable models (subset, check OpenRouter for full list)
VISION_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-2024-11-20",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-pro-vision",
    "google/gemini-1.5-pro",
    "google/gemini-1.5-flash",
]


def supports_vision(model: str) -> bool:
    """Check if model supports vision (image) input."""
    return any(v in model for v in VISION_MODELS)
