"""
Context Gate - Heuristic gating for memory context injection.

Decides whether a message needs memory/document context injected,
saving tokens and latency for messages that don't benefit from context.
"""

from __future__ import annotations

from typing import Tuple


class ContextGate:
    """
    Heuristic gate that decides if a message needs memory context injection.

    Uses fast string-based heuristics (zero latency) to filter out messages
    that clearly don't need external context, like acknowledgments,
    continuations, and clarifications about the immediate conversation.

    Usage:
        should_inject, score, reason = ContextGate.decide("What is Cathedral?")
        if should_inject:
            context = search_memory(query)
    """

    # === DEFINITE SKIP (exact match, normalized) ===
    SKIP_EXACT = frozenset({
        # Acknowledgments
        'ok', 'okay', 'k', 'yes', 'no', 'yeah', 'yep', 'nope', 'nah',
        'sure', 'thanks', 'thank you', 'ty', 'thx', 'got it', 'gotcha',
        'understood', 'i see', 'makes sense', 'fair enough', 'fair',
        # Affirmations
        'cool', 'nice', 'great', 'awesome', 'perfect', 'good', 'fine',
        'alright', 'right', 'correct', 'exactly', 'indeed', 'agreed', 'true',
        # Reactions
        'lol', 'lmao', 'haha', 'heh', 'wow', 'whoa', 'ah', 'oh', 'hmm', 'hm',
        'interesting', 'neat', 'sick', 'dope', 'nice one',
        # Continuations
        'continue', 'go on', 'go ahead', 'proceed', 'next', 'more',
        'keep going', 'and', 'and then', 'so', 'then what', 'what else',
        # Simple responses
        'done', 'noted', 'roger', 'copy', 'ack', 'np', 'no problem',
    })

    # === MEMORY TRIGGERS (always inject if present) ===
    MEMORY_TRIGGERS = (
        'remember when', 'remember that', 'you mentioned', 'we talked about',
        'we discussed', 'you said', 'you told me', 'last time', 'previously',
        'earlier you', 'before you', 'recall when', 'what do you know about',
        'what have we', 'have we ever', 'did we', 'did you ever',
        'as we discussed', 'as you mentioned', 'like you said',
    )

    # === KNOWLEDGE QUESTION PATTERNS ===
    KNOWLEDGE_STARTS = (
        'what is', 'what are', 'what was', 'what were', "what's",
        'who is', 'who are', 'who was', "who's",
        'how do', 'how does', 'how did', 'how can', 'how would', 'how is',
        'why is', 'why are', 'why did', 'why does', "why's",
        'when did', 'when was', 'when is', 'when does',
        'where is', 'where are', 'where did', "where's",
        'tell me about', 'explain', 'describe', 'define', 'summarize',
        'can you tell me', 'do you know', 'do you remember',
        'what do you think about', 'what are your thoughts on',
    )

    # === ANAPHORIC STARTS (refers to immediate context) ===
    ANAPHORIC = frozenset({
        'it', 'its', "it's", 'that', "that's", 'this', 'these', 'those',
        'he', 'she', 'they', 'them', 'his', 'her', 'their',
    })

    # === TOPIC SHIFT MARKERS (new topic = might need context) ===
    TOPIC_SHIFTS = (
        'anyway', 'by the way', 'btw', 'speaking of', 'unrelated',
        'different topic', 'switching gears', 'on another note', 'also',
        'oh and', 'one more thing', 'separately',
    )

    # === CLARIFICATION PATTERNS (about current response, not memory) ===
    CLARIFICATION = (
        'what do you mean', 'can you explain', 'could you explain',
        'say that again', 'come again', "don't understand", "didn't understand",
        'clarify', 'elaborate', 'be more specific', 'what does that mean',
        'huh', 'sorry what', 'wait what', "i don't get", "i didn't get",
        'can you clarify', 'what did you mean', 'expand on that',
    )

    @classmethod
    def score(cls, message: str) -> Tuple[float, str]:
        """
        Score a message for context injection need.

        Args:
            message: The user's message

        Returns:
            Tuple of (score, reason)
            Score: 0.0 = definitely skip, 1.0 = definitely inject
            Reason: Human-readable explanation for logging/debugging
        """
        msg = message.strip()
        msg_lower = msg.lower().strip()
        words = msg_lower.split()
        word_count = len(words)

        # === FAST EXITS ===

        # Commands handle themselves
        if msg.startswith('/'):
            return 0.0, "command"

        # Empty
        if not words:
            return 0.0, "empty"

        # Exact skip match
        if msg_lower in cls.SKIP_EXACT:
            return 0.0, "skip_exact"

        # Two-word combinations that are also skips
        if word_count == 2:
            two_word = ' '.join(words)
            if two_word in cls.SKIP_EXACT:
                return 0.0, "skip_exact_2word"

        # === STRONG SIGNALS ===

        # Explicit memory trigger → always inject
        if any(t in msg_lower for t in cls.MEMORY_TRIGGERS):
            return 1.0, "memory_trigger"

        # Clarification about current response → skip (needs conversation, not memory)
        if any(c in msg_lower for c in cls.CLARIFICATION):
            return 0.1, "clarification"

        # Knowledge question pattern → likely needs context
        if any(msg_lower.startswith(k) for k in cls.KNOWLEDGE_STARTS):
            return 0.9, "knowledge_question"

        # Topic shift → might need context for new topic
        if any(t in msg_lower for t in cls.TOPIC_SHIFTS):
            return 0.7, "topic_shift"

        # === LENGTH-BASED HEURISTICS ===

        # Very short without question mark
        if word_count <= 2 and '?' not in msg:
            return 0.1, "very_short"

        # Short anaphoric reference (about immediate context)
        if word_count <= 5 and words[0] in cls.ANAPHORIC:
            return 0.2, "anaphoric"

        # Short with question mark → might need context
        if word_count <= 5 and '?' in msg:
            return 0.5, "short_question"

        # Medium length (4-10 words)
        if 4 <= word_count <= 10:
            if '?' in msg:
                return 0.7, "medium_question"
            return 0.5, "medium_statement"

        # Long messages (>10 words) → probably substantive
        if word_count > 10:
            if '?' in msg:
                return 0.85, "long_question"
            return 0.7, "long_statement"

        # Default uncertain
        return 0.5, "default"

    @classmethod
    def should_inject(cls, message: str, threshold: float = 0.5) -> bool:
        """
        Simple boolean check for whether to inject context.

        Args:
            message: The user's message
            threshold: Score threshold (default 0.5)

        Returns:
            True if context should be injected
        """
        score, _ = cls.score(message)
        return score >= threshold

    @classmethod
    def decide(cls, message: str, threshold: float = 0.5) -> Tuple[bool, float, str]:
        """
        Full decision with debugging info.

        Args:
            message: The user's message
            threshold: Score threshold (default 0.5)

        Returns:
            Tuple of (should_inject, score, reason)
        """
        score, reason = cls.score(message)
        return score >= threshold, score, reason


__all__ = ["ContextGate"]
