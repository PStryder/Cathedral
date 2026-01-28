"""
Automatic memory extraction and context injection for Cathedral.
"""

from typing import Optional
from cathedral.MemoryGate import (
    store_observation,
    search,
    is_initialized
)


def extract_from_exchange(
    user_input: str,
    assistant_response: str,
    thread_uid: str
) -> Optional[dict]:
    """
    Extract and store a memory from a user/assistant exchange.
    Called after each successful LLM response.

    Returns the stored observation or None if storage failed.
    """
    if not is_initialized():
        return None

    # Truncate to reasonable lengths for the observation
    user_excerpt = user_input[:500].strip()
    response_excerpt = assistant_response[:1000].strip()

    # Create a combined observation text
    observation_text = f"User: {user_excerpt}\nAssistant: {response_excerpt}"

    # Store with conversation domain and thread as evidence
    result = store_observation(
        text=observation_text,
        confidence=0.7,
        domain="conversation",
        evidence=[f"thread:{thread_uid}"]
    )

    return result


def build_memory_context(user_input: str, limit: int = 3, min_confidence: float = 0.5) -> str:
    """
    Build memory context to inject into the prompt.
    Searches for semantically relevant memories based on user input.

    Returns formatted string of relevant memories, or empty string if none found.
    """
    if not is_initialized():
        return ""

    results = search(
        query=user_input,
        limit=limit,
        min_confidence=min_confidence
    )

    if not results:
        return ""

    lines = ["[RELEVANT MEMORIES]"]
    for r in results:
        source_type = r.get("source_type", "memory")
        similarity = r.get("similarity", 0)

        # Extract text based on source type
        if source_type == "observation":
            text = r.get("observation", r.get("text", ""))
        elif source_type == "pattern":
            text = r.get("pattern_text", r.get("text", ""))
        elif source_type == "concept":
            text = r.get("description", r.get("text", ""))
        else:
            text = r.get("text", r.get("observation", ""))

        if text:
            # Truncate long memories
            text_truncated = text[:300] + "..." if len(text) > 300 else text
            confidence = r.get("confidence", 0)
            lines.append(f"- [{source_type}|sim:{similarity:.2f}|conf:{confidence:.1f}] {text_truncated}")

    if len(lines) == 1:  # Only header, no results
        return ""

    return "\n".join(lines)


def format_search_results(results: list, max_per_result: int = 200) -> str:
    """
    Format search results for display to user.
    """
    if not results:
        return "No relevant memories found."

    lines = []
    for i, r in enumerate(results, 1):
        source_type = r.get("source_type", "memory")
        ref = f"{source_type}:{r.get('id', '?')}"

        # Extract text based on source type
        if source_type == "observation":
            text = r.get("observation", "")
        elif source_type == "pattern":
            text = r.get("pattern_text", "")
            category = r.get("category", "")
            name = r.get("pattern_name", "")
            if category and name:
                ref = f"pattern:{category}/{name}"
        elif source_type == "concept":
            text = r.get("description", "")
            name = r.get("name", "")
            if name:
                ref = f"concept:{name}"
        else:
            text = r.get("text", "")

        text_truncated = text[:max_per_result] + "..." if len(text) > max_per_result else text
        similarity = r.get("similarity", 0)
        lines.append(f"{i}. [{ref}] (sim:{similarity:.2f}) {text_truncated}")

    return "\n".join(lines)


def format_recall_results(results: list, max_per_result: int = 200) -> str:
    """
    Format recall results for display to user.
    """
    if not results:
        return "No observations found."

    lines = []
    for i, r in enumerate(results, 1):
        obs_id = r.get("id", "?")
        text = r.get("observation", r.get("text", ""))
        domain = r.get("domain", "")
        confidence = r.get("confidence", 0)

        text_truncated = text[:max_per_result] + "..." if len(text) > max_per_result else text
        domain_str = f" [{domain}]" if domain else ""
        lines.append(f"{i}. [obs:{obs_id}]{domain_str} (conf:{confidence:.1f}) {text_truncated}")

    return "\n".join(lines)
