import os

from llama_cpp import Llama

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("LoomMirror")


class LoomMirror:
    """
    LoomMirror is the utility LLM interface for memory compression, tagging,
    summarization, and internal cognition tasks. It runs a small GGUF model locally
    using llama-cpp-python.
    """

    def __init__(self, model_path: str, model_name: str = "LoomMirror", n_ctx: int = 2048):
        self.model_path = model_path
        self.model_name = model_name
        self.n_ctx = n_ctx

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        _log.info(f"Initializing memory model '{model_name}' from: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=-1,
            verbose=False
        )

    def run(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        Executes a prompt against the utility memory LLM and returns the response.
        This method is suitable for summarization, scoring, and tagging tasks.
        """
        _log.debug(f"Running prompt (truncated): {prompt[:60]}...")
        output = self.llm(prompt, max_tokens=max_tokens, temperature=temperature)
        return output["choices"][0]["text"].strip()

    def summarize_pair(self, user_text: str, assistant_text: str) -> str:
        """Summarizes a message pair into a short summary."""
        prompt = (
            "Summarize the following conversation in 1-2 sentences, capturing the core idea and tone.\n"
            f"USER: {user_text}\nASSISTANT: {assistant_text}\nSUMMARY:"
        )
        return self.run(prompt, max_tokens=64)

    def extract_facts(self, text: str) -> list:
        """Extracts factual statements line-by-line from a text block."""
        prompt = (
            "Extract all standalone factual statements from the following text.\n"
            "Return each fact on a new line.\n\nTEXT:\n"
            f"{text}\n\nFACTS:"
        )
        result = self.run(prompt, max_tokens=128)
        return [line.strip() for line in result.splitlines() if line.strip()]

    def tag_pair(self, user_text: str, assistant_text: str) -> list:
        """Tags a message pair with relevant topics and tones."""
        prompt = (
            "Generate a list of relevant topic tags and tone descriptors for the following exchange.\n"
            "Return them as a comma-separated list.\n\n"
            f"USER: {user_text}\nASSISTANT: {assistant_text}\nTAGS:"
        )
        result = self.run(prompt, max_tokens=64)
        return [tag.strip() for tag in result.split(",") if tag.strip()]

    def score_salience(self, text: str) -> float:
        """
        Scores the importance/salience of a message or fact.

        Returns a float between 0 and 1 where:
        - 0.0-0.3: Low importance (small talk, acknowledgments)
        - 0.4-0.6: Medium importance (general information)
        - 0.7-1.0: High importance (key facts, decisions, identity info)
        """
        prompt = (
            "Rate the importance of the following text on a scale of 0 to 10, "
            "where 0 is trivial small talk and 10 is critically important information.\n"
            "Consider: personal identity, key decisions, factual claims, and emotional significance.\n"
            "Respond with ONLY a single number.\n\n"
            f"TEXT: {text[:500]}\n\nSCORE:"
        )
        try:
            result = self.run(prompt, max_tokens=8, temperature=0.3)
            # Extract first number from response
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', result)
            if numbers:
                score = float(numbers[0])
                return min(1.0, max(0.0, score / 10.0))
        except Exception as e:
            _log.debug(f"Salience scoring failed: {e}")
        return 0.5  # Default to medium if scoring fails

    def summarize_thread(self, messages: list) -> str:
        """Summarizes a list of message strings into a fixed-size thread summary block."""
        combined = "\n".join([f"{m}" for m in messages])
        prompt = (
            "Summarize the following conversation thread into a compressed abstract no longer than 5 sentences.\n"
            "Focus on extracting persistent themes and useful identity or topic markers.\n\nTHREAD:\n"
            f"{combined}\n\nSUMMARY:"
        )
        return self.run(prompt, max_tokens=256)

    def propose_codex_entries(self, text: str) -> list:
        """
        Proposes persistent memory facts for Codex insertion.

        Extracts statements that should be remembered long-term, such as:
        - Personal identity facts (name, preferences, relationships)
        - Important decisions or commitments
        - Recurring topics or interests

        Returns:
            List of dicts with 'fact' and 'confidence' keys
        """
        prompt = (
            "Extract facts from the following text that should be remembered permanently.\n"
            "Focus on: identity information, preferences, important decisions, and key relationships.\n"
            "Return each fact on a new line.\n\n"
            f"TEXT: {text[:1000]}\n\nPERMANENT FACTS:"
        )
        try:
            result = self.run(prompt, max_tokens=256)
            facts = []
            for line in result.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove bullet points or numbering
                    line = line.lstrip('•-*0123456789. ')
                    if len(line) > 5:
                        facts.append({
                            "fact": line,
                            "confidence": 0.7
                        })
            return facts
        except Exception as e:
            _log.debug(f"Codex proposal failed: {e}")
            return []

    def generate_reverie(self, text: str) -> dict:
        """
        Generates a reverie entry - an internal reflection or insight.

        Reveries are meta-cognitive observations about patterns, connections,
        or implications that emerge from processing information.

        Returns:
            Dict with 'key', 'value', and 'weight' keys
        """
        prompt = (
            "Based on the following text, generate a brief meta-observation or insight.\n"
            "This should be a reflective thought about patterns, implications, or connections.\n"
            "Be concise (one sentence) and insightful.\n\n"
            f"TEXT: {text[:500]}\n\nREFLECTION:"
        )
        try:
            result = self.run(prompt, max_tokens=64, temperature=0.8)
            result = result.strip()
            if result:
                # Generate a short key from the reflection
                key_prompt = f"Generate a 2-3 word topic key for: {result}\nKEY:"
                key = self.run(key_prompt, max_tokens=8, temperature=0.3).strip()
                key = key.lower().replace(' ', '_')[:30]
                return {
                    "key": key or "reflection",
                    "value": result,
                    "weight": self.score_salience(result)
                }
        except Exception as e:
            _log.debug(f"Reverie generation failed: {e}")
        return {
            "key": "unprocessed",
            "value": text[:100] if text else "",
            "weight": 0.3
        }

    def clean_text_for_embedding(self, text: str) -> str:
        """
        Preprocesses text for cleaner embedding generation.

        Removes:
        - Excessive whitespace and newlines
        - Common boilerplate phrases
        - Code blocks (keeps description only)
        - Markdown formatting artifacts
        """
        import re

        if not text:
            return ""

        # Normalize whitespace
        text = ' '.join(text.split())

        # Remove markdown code blocks but keep a marker
        text = re.sub(r'```[\w]*\n?.*?```', '[code block]', text, flags=re.DOTALL)

        # Remove inline code backticks
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Remove markdown links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Remove common boilerplate phrases
        boilerplate = [
            r"^(Sure|Of course|Certainly|Absolutely)[,!.]?\s*",
            r"^(I'd be happy to|I can help|Let me)\s+",
            r"^(Here's|Here is)\s+",
        ]
        for pattern in boilerplate:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.strip()

    def extract_facts_and_tags(self, user_text: str, assistant_text: str) -> dict:
        """
        Extracts both factual statements and topic tags from a conversation exchange.

        Args:
            user_text: The user's message
            assistant_text: The assistant's response

        Returns:
            Dict with 'facts' (list of strings) and 'tags' (list of strings)
        """
        # Extract facts from both sides of the conversation
        combined = f"USER: {user_text}\nASSISTANT: {assistant_text}"
        facts = self.extract_facts(combined)

        # Get tags for the exchange
        tags = self.tag_pair(user_text, assistant_text)

        return {
            "facts": facts,
            "tags": tags
        }

    def __repr__(self):
        return f"<LoomMirror model='{self.model_name}'>"
