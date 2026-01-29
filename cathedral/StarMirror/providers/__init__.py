"""
StarMirror provider helpers.
"""

from typing import Any, Dict, List


def _content_to_text(content: Any) -> str:
    """Convert message content (text or multimodal) to a plain-text prompt."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text", "")
                if text:
                    parts.append(text)
            elif part_type == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    parts.append("[image:base64]")
                elif url:
                    trimmed = url if len(url) <= 120 else f"{url[:117]}..."
                    parts.append(f"[image:{trimmed}]")
                else:
                    parts.append("[image]")
            elif part_type == "input_audio":
                parts.append("[audio]")
        return " ".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def serialize_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Serialize Cathedral {role, content} messages into a single prompt string.
    """
    lines: List[str] = []
    for message in messages or []:
        role = str(message.get("role", "user")).upper()
        content = _content_to_text(message.get("content", ""))
        lines.append(f"{role}:\n{content}".rstrip())

    # Encourage CLI tools to respond as the assistant.
    lines.append("ASSISTANT:")
    return "\n\n".join(lines).strip() + "\n"


__all__ = ["serialize_messages"]
