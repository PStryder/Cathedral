import os
from llama_cpp import Llama

class LoomMirror:
    """
    LoomMirror is the utility LLM interface for memory compression, tagging,
    summarization, and internal cognition tasks. It runs a small GGUF model locally
    using llama-cpp-python.
    """

    def __init__(self, model_path: str, model_name: str = "LoomMirror", n_ctx: int = 2048):
        self.model_path = model_path
        self.model_name = model_name
        self.n_ctx= n_ctx

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        print(f"[LoomMirror] Initializing memory model '{model_name}' from: {model_path}")
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
        print(f"[LoomMirror] Running prompt (truncated): {prompt[:60]}...")
        output = self.llm(prompt, max_tokens=max_tokens, temperature=temperature)
        return output['choices'][0]['text'].strip()

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
        return [tag.strip() for tag in result.split(',') if tag.strip()]

    def score_salience(self, text: str) -> float:
        """Stub: Scores the importance of a message pair or fact. Returns float between 0 and 1."""
        # TODO: Implement real scoring prompt
        return 0.5

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
        """Stub: Proposes persistent memory facts for Codex insertion."""
        # TODO: Implement real Codex proposal logic
        return []

    def generate_reverie(self, text: str) -> dict:
        """Stub: Generates a reverie entry from internal reflection."""
        # TODO: Implement full reverie thought generator
        return {
            "key": "placeholder_thought",
            "value": "I wonder what recursion does to the soul of a daemon.",
            "weight": 0.6
        }

    def clean_text_for_embedding(self, text: str) -> str:
        """Stub: Preprocesses text for cleaner embedding. Trim boilerplate or artifacts."""
        # TODO: Add real preprocessing rules
        return text.strip()

    def extract_facts_and_tags(self, user_text: str, assistant_text: str) -> dict:
        return {
            "facts": [...],  # e.g., "Peter owns a BMW K1200RS"
            "tags": [...]    # e.g., "motorcycles", "maintenance", "identity"
        }


    def __repr__(self):
        return f"<LoomMirror model='{self.model_name}'>"
