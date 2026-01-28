"""
ScriptureGate file storage operations.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Tuple, BinaryIO, Union
from datetime import datetime
import uuid
import mimetypes

# Initialize mimetypes
mimetypes.init()

# Storage root
SCRIPTURE_ROOT = Path(__file__).resolve().parents[2] / "data" / "scripture"
SCRIPTURE_ROOT.mkdir(parents=True, exist_ok=True)

# Subdirectories by type
STORAGE_DIRS = {
    "document": SCRIPTURE_ROOT / "documents",
    "image": SCRIPTURE_ROOT / "images",
    "audio": SCRIPTURE_ROOT / "audio",
    "artifact": SCRIPTURE_ROOT / "artifacts",
    "thread": SCRIPTURE_ROOT / "threads",
}

# Ensure all dirs exist
for d in STORAGE_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


def detect_file_type(path: Union[str, Path], mime_type: str = None) -> str:
    """Detect file type category from path or mime type."""
    path = Path(path)
    ext = path.suffix.lower()

    # By extension
    doc_exts = {".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".csv", ".json", ".xml", ".html"}
    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".ico"}
    audio_exts = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".wma"}

    if ext in doc_exts:
        return "document"
    if ext in image_exts:
        return "image"
    if ext in audio_exts:
        return "audio"

    # By mime type
    if mime_type:
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("audio/"):
            return "audio"
        if mime_type.startswith("text/") or mime_type == "application/pdf":
            return "document"

    return "artifact"  # Default


def detect_mime_type(path: Union[str, Path]) -> str:
    """Detect MIME type from file path."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def compute_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_storage_path(file_type: str, original_name: str, scripture_uid: str = None) -> Tuple[Path, str]:
    """
    Generate storage path for a file.
    Returns (full_path, relative_path).
    """
    if scripture_uid is None:
        scripture_uid = str(uuid.uuid4())

    # Get storage directory
    storage_dir = STORAGE_DIRS.get(file_type, STORAGE_DIRS["artifact"])

    # Organize by date
    date_dir = datetime.utcnow().strftime("%Y/%m")
    target_dir = storage_dir / date_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # Preserve extension, use UID as filename
    ext = Path(original_name).suffix
    filename = f"{scripture_uid}{ext}"
    full_path = target_dir / filename

    # Relative path from scripture root
    relative_path = str(full_path.relative_to(SCRIPTURE_ROOT))

    return full_path, relative_path


def store_file(
    source: Union[str, Path, BinaryIO],
    file_type: str = None,
    original_name: str = None,
    scripture_uid: str = None
) -> Tuple[str, str, int, str, str]:
    """
    Store a file in scripture storage.

    Args:
        source: File path, Path object, or file-like object
        file_type: Override file type detection
        original_name: Original filename (required if source is file-like)
        scripture_uid: Optional UID (generated if not provided)

    Returns:
        (scripture_uid, relative_path, file_size, mime_type, content_hash)
    """
    if scripture_uid is None:
        scripture_uid = str(uuid.uuid4())

    # Handle different source types
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        original_name = original_name or source_path.name
        is_file_object = False
    else:
        # File-like object
        if not original_name:
            raise ValueError("original_name required for file-like objects")
        is_file_object = True

    # Detect types
    if file_type is None:
        file_type = detect_file_type(original_name)
    mime_type = detect_mime_type(original_name)

    # Generate storage path
    full_path, relative_path = generate_storage_path(file_type, original_name, scripture_uid)

    # Copy/write file
    if is_file_object:
        with open(full_path, "wb") as f:
            shutil.copyfileobj(source, f)
    else:
        shutil.copy2(source_path, full_path)

    # Get file info
    file_size = full_path.stat().st_size
    content_hash = compute_hash(full_path)

    return scripture_uid, relative_path, file_size, mime_type, content_hash


def store_content(
    content: Union[str, bytes],
    filename: str,
    file_type: str = None,
    scripture_uid: str = None
) -> Tuple[str, str, int, str, str]:
    """
    Store content directly (not from a file).

    Args:
        content: Text or bytes content
        filename: Filename to use (with extension)
        file_type: Override type detection
        scripture_uid: Optional UID

    Returns:
        (scripture_uid, relative_path, file_size, mime_type, content_hash)
    """
    if scripture_uid is None:
        scripture_uid = str(uuid.uuid4())

    if file_type is None:
        file_type = detect_file_type(filename)
    mime_type = detect_mime_type(filename)

    full_path, relative_path = generate_storage_path(file_type, filename, scripture_uid)

    # Write content
    if isinstance(content, str):
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(full_path, "wb") as f:
            f.write(content)

    file_size = full_path.stat().st_size
    content_hash = compute_hash(full_path)

    return scripture_uid, relative_path, file_size, mime_type, content_hash


def get_full_path(relative_path: str) -> Path:
    """Get full path from relative path."""
    return SCRIPTURE_ROOT / relative_path


def read_file(relative_path: str) -> bytes:
    """Read file content as bytes."""
    full_path = get_full_path(relative_path)
    if not full_path.exists():
        raise FileNotFoundError(f"Scripture file not found: {relative_path}")
    with open(full_path, "rb") as f:
        return f.read()


def read_text(relative_path: str, encoding: str = "utf-8") -> str:
    """Read file content as text."""
    full_path = get_full_path(relative_path)
    if not full_path.exists():
        raise FileNotFoundError(f"Scripture file not found: {relative_path}")
    with open(full_path, "r", encoding=encoding) as f:
        return f.read()


def delete_file(relative_path: str) -> bool:
    """Delete a file from storage."""
    full_path = get_full_path(relative_path)
    if full_path.exists():
        full_path.unlink()
        return True
    return False


def file_exists(relative_path: str) -> bool:
    """Check if file exists."""
    return get_full_path(relative_path).exists()
