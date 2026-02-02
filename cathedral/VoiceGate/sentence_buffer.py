"""
Sentence buffer for VoiceGate.

Accumulates tokens until a sentence boundary is detected,
enabling smoother TTS synthesis without mid-sentence cuts.
"""

import re
from typing import Optional


class SentenceBuffer:
    """
    Accumulate tokens until a sentence boundary is reached.

    This enables synthesizing complete sentences rather than
    partial fragments, resulting in more natural speech output.
    """

    # Sentence-ending punctuation followed by whitespace or end
    SENTENCE_END = re.compile(r'[.!?]["\')\]]?\s*$')

    # Minimum length before we'll flush on punctuation
    MIN_SENTENCE_LENGTH = 10

    # Maximum buffer before force flush (prevent memory issues)
    MAX_BUFFER_LENGTH = 500

    def __init__(self):
        self.buffer = ""
        self._sentence_count = 0

    def add_token(self, token: str) -> Optional[str]:
        """
        Add a token to the buffer.

        Args:
            token: The token to add (from LLM streaming)

        Returns:
            Complete sentence if boundary reached, None otherwise
        """
        self.buffer += token

        # Check for sentence boundary
        if self._is_sentence_end():
            sentence = self.buffer.strip()
            self.buffer = ""
            self._sentence_count += 1
            return sentence

        # Force flush if buffer too long
        if len(self.buffer) > self.MAX_BUFFER_LENGTH:
            return self._force_flush()

        return None

    def _is_sentence_end(self) -> bool:
        """Check if buffer ends at a sentence boundary."""
        if len(self.buffer.strip()) < self.MIN_SENTENCE_LENGTH:
            return False

        return bool(self.SENTENCE_END.search(self.buffer))

    def _force_flush(self) -> Optional[str]:
        """
        Force flush the buffer at the best available break point.

        Tries to find a reasonable break point (comma, semicolon, etc.)
        rather than cutting mid-word.
        """
        # Look for clause breaks
        for pattern in [r'[,;:]\s+', r'\s+']:
            match = re.search(pattern, self.buffer[50:])  # Skip first 50 chars
            if match:
                break_point = 50 + match.end()
                sentence = self.buffer[:break_point].strip()
                self.buffer = self.buffer[break_point:]
                return sentence

        # No good break point, flush everything
        sentence = self.buffer.strip()
        self.buffer = ""
        return sentence if sentence else None

    def flush(self) -> Optional[str]:
        """
        Flush any remaining text in the buffer.

        Called at end of stream to get final content.

        Returns:
            Remaining text or None if empty
        """
        if self.buffer.strip():
            sentence = self.buffer.strip()
            self.buffer = ""
            return sentence
        return None

    def reset(self):
        """Clear the buffer."""
        self.buffer = ""
        self._sentence_count = 0

    @property
    def sentence_count(self) -> int:
        """Number of complete sentences yielded."""
        return self._sentence_count

    @property
    def pending_length(self) -> int:
        """Current buffer length."""
        return len(self.buffer)

    def __len__(self) -> int:
        """Return current buffer length."""
        return len(self.buffer)

    def __bool__(self) -> bool:
        """Return True if buffer has content."""
        return bool(self.buffer.strip())


__all__ = ["SentenceBuffer"]
