"""
PersonalityGate Pydantic models.

Defines the structure for personality configurations.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration for a personality."""
    model: str = Field(default="openai/gpt-4o-2024-11-20", description="Model identifier")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4000, ge=100, le=128000)
    system_prompt: str = Field(default="You are a helpful assistant.")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)


class BehaviorConfig(BaseModel):
    """Behavioral settings for a personality."""
    style_tags: List[str] = Field(default_factory=list)
    communication_style: str = Field(default="conversational")
    response_format: str = Field(default="standard")


class MemoryConfig(BaseModel):
    """Memory/context settings for a personality."""
    domains: List[str] = Field(default_factory=list)
    auto_extract: bool = Field(default=True)
    context_priority: str = Field(default="recent_plus_relevant")
    max_context_messages: int = Field(default=20)


class ToolsConfig(BaseModel):
    """Tool availability settings."""
    enabled: List[str] = Field(default_factory=lambda: ["web_search", "scripture_search", "memory_search"])
    disabled: List[str] = Field(default_factory=list)


class ExampleExchange(BaseModel):
    """Example conversation exchange for few-shot prompting."""
    user: str
    assistant: str


class PersonalityMetadata(BaseModel):
    """Metadata about the personality."""
    category: str = Field(default="general")
    author: str = Field(default="user")
    version: str = Field(default="1.0")
    is_default: bool = Field(default=False)
    is_builtin: bool = Field(default=False)
    usage_count: int = Field(default=0)


class Personality(BaseModel):
    """Complete personality configuration."""
    id: str
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    examples: List[ExampleExchange] = Field(default_factory=list)
    metadata: PersonalityMetadata = Field(default_factory=PersonalityMetadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Personality":
        """Create from dict."""
        return cls.model_validate(data)

    def get_system_prompt(self) -> str:
        """Get the full system prompt including examples."""
        prompt = self.llm_config.system_prompt

        if self.examples:
            prompt += "\n\nHere are some example exchanges that demonstrate my style:\n"
            for ex in self.examples[:3]:  # Limit to 3 examples
                prompt += f"\nUser: {ex.user}\nAssistant: {ex.assistant}\n"

        return prompt

    def increment_usage(self):
        """Increment usage counter."""
        self.metadata.usage_count += 1
        self.updated_at = datetime.utcnow()


class PersonalitySnapshot(BaseModel):
    """Minimal snapshot of personality for thread history."""
    id: str
    name: str
    system_prompt: str
    model: str
    temperature: float
