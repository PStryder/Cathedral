"""
Tests for ContextGate heuristic gating.
"""

import pytest
from cathedral.MemoryGate.context_gate import ContextGate


class TestContextGateSkips:
    """Test messages that should be skipped."""

    @pytest.mark.parametrize("message", [
        "ok",
        "okay",
        "thanks",
        "thank you",
        "got it",
        "yes",
        "no",
        "cool",
        "nice",
        "lol",
        "haha",
        "continue",
        "go on",
        "go ahead",
    ])
    def test_acknowledgments_skipped(self, message):
        """Acknowledgments should be skipped."""
        should_inject, score, reason = ContextGate.decide(message)
        assert not should_inject, f"'{message}' should be skipped, got reason: {reason}"
        assert score < 0.5

    def test_commands_skipped(self):
        """Commands starting with / should be skipped."""
        should_inject, score, reason = ContextGate.decide("/help")
        assert not should_inject
        assert reason == "command"

    def test_empty_skipped(self):
        """Empty messages should be skipped."""
        should_inject, score, reason = ContextGate.decide("")
        assert not should_inject
        assert reason == "empty"

    def test_very_short_no_question(self):
        """Very short messages without questions should be skipped."""
        should_inject, score, reason = ContextGate.decide("hi there")
        assert not should_inject
        assert score < 0.5

    def test_clarification_skipped(self):
        """Clarification requests should be skipped (need conversation, not memory)."""
        should_inject, score, reason = ContextGate.decide("what do you mean by that?")
        assert not should_inject
        assert reason == "clarification"


class TestContextGateInjects:
    """Test messages that should trigger context injection."""

    def test_memory_trigger(self):
        """Explicit memory requests should always inject."""
        should_inject, score, reason = ContextGate.decide("remember when we talked about Cathedral?")
        assert should_inject
        assert score == 1.0
        assert reason == "memory_trigger"

    def test_memory_trigger_you_mentioned(self):
        """'You mentioned' should trigger memory."""
        should_inject, score, reason = ContextGate.decide("you mentioned something about that earlier")
        assert should_inject
        assert reason == "memory_trigger"

    @pytest.mark.parametrize("message", [
        "What is Cathedral?",
        "Who is the author?",
        "How does the memory system work?",
        "Why is pgvector used?",
        "Tell me about the architecture",
        "Explain the context injection",
    ])
    def test_knowledge_questions(self, message):
        """Knowledge questions should inject."""
        should_inject, score, reason = ContextGate.decide(message)
        assert should_inject, f"'{message}' should inject, got reason: {reason}"
        assert reason == "knowledge_question"

    def test_topic_shift(self):
        """Topic shifts should inject (new topic might need context)."""
        should_inject, score, reason = ContextGate.decide("By the way, what about the MCP client?")
        assert should_inject
        assert reason == "topic_shift"

    def test_long_question(self):
        """Long questions should inject."""
        message = "Can you help me understand how the hybrid search combines full-text search with vector similarity?"
        should_inject, score, reason = ContextGate.decide(message)
        assert should_inject
        assert score >= 0.7


class TestContextGateEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_anaphoric_reference(self):
        """Short anaphoric references should have low score."""
        should_inject, score, reason = ContextGate.decide("That sounds good")
        assert not should_inject
        assert reason == "anaphoric"

    def test_medium_question(self):
        """Medium-length questions should inject."""
        should_inject, score, reason = ContextGate.decide("How does this work?")
        assert should_inject
        # Could be short_question or medium_question depending on word count
        assert "question" in reason

    def test_long_statement(self):
        """Long statements should inject (probably substantive)."""
        message = "I'm working on integrating the memory system with the new MCP client we just built"
        should_inject, score, reason = ContextGate.decide(message)
        assert should_inject

    def test_threshold_customization(self):
        """Custom threshold should affect decision."""
        # This message might be borderline
        message = "Can you help?"

        # With default threshold
        result_default = ContextGate.decide(message, threshold=0.5)

        # With high threshold
        result_high = ContextGate.decide(message, threshold=0.9)

        # Score should be the same, decision may differ
        assert result_default[1] == result_high[1]  # Same score

    def test_should_inject_simple(self):
        """Test simple boolean interface."""
        assert not ContextGate.should_inject("ok")
        assert ContextGate.should_inject("What is Cathedral?")

    def test_score_returns_tuple(self):
        """Score method should return (score, reason) tuple."""
        result = ContextGate.score("test message")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)

    def test_case_insensitive(self):
        """Matching should be case insensitive."""
        assert not ContextGate.should_inject("OK")
        assert not ContextGate.should_inject("Thanks")
        assert not ContextGate.should_inject("THANKS")


class TestContextGateScoring:
    """Test the scoring logic."""

    def test_score_range(self):
        """Scores should be between 0 and 1."""
        test_messages = [
            "ok",
            "What is this?",
            "Remember when we discussed the architecture?",
            "By the way, about that thing",
            "This is a very long message that contains many words and should probably trigger context injection",
        ]

        for msg in test_messages:
            score, reason = ContextGate.score(msg)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for '{msg}'"

    def test_memory_trigger_max_score(self):
        """Memory triggers should get maximum score."""
        score, reason = ContextGate.score("what do you know about Cathedral?")
        assert score == 1.0

    def test_skip_exact_min_score(self):
        """Exact skip matches should get minimum score."""
        score, reason = ContextGate.score("ok")
        assert score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
