"""
MetadataChannel - Structured metadata sidechannel for Cathedral.

Provides on-demand access to conversation metadata without polluting prompts.
Designed for agent consumption, not human readability.
"""

from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass, field
import time


class MetadataProvider(Protocol):
    """Interface for metadata providers."""
    name: str
    fields: List[str]

    def get(self, target: str, fields: Optional[List[str]], context: dict) -> dict:
        """Get metadata for target. Returns dict with requested fields."""
        ...


@dataclass
class MetadataResponse:
    """Compact metadata response."""
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to minimal dict for agent consumption."""
        if not self.ok:
            return {"ok": False, "err": self.error}
        return {"ok": True, **self.data}


class MetadataChannel:
    """
    Central metadata sidechannel manager.

    Providers register to supply specific metadata fields.
    Queries are routed to appropriate providers.
    """

    def __init__(self):
        self._providers: Dict[str, MetadataProvider] = {}
        self._field_map: Dict[str, str] = {}  # field -> provider name

    def register_provider(self, provider: MetadataProvider) -> None:
        """Register a metadata provider."""
        self._providers[provider.name] = provider
        for f in provider.fields:
            self._field_map[f] = provider.name

    def list_providers(self) -> List[str]:
        """List registered provider names."""
        return list(self._providers.keys())

    def list_fields(self) -> List[str]:
        """List all available metadata fields."""
        return list(self._field_map.keys())

    def check_policy(self, query: str, context: dict) -> bool:
        """
        Policy enforcement for metadata queries.

        Implements basic access control:
        - Rate limiting (configurable)
        - Field restrictions (configurable)
        - Context requirements validation

        Args:
            query: The target query string
            context: Query context with caller info

        Returns:
            True if query is allowed, False otherwise
        """
        # Check if policy is disabled (for testing/development)
        if getattr(self, '_policy_disabled', False):
            return True

        # Validate required context for most queries
        if query not in ("current", "thread"):
            # Non-basic queries require thread context
            if not context.get("thread_uid"):
                return False

        # Check field restrictions if configured
        restricted_fields = getattr(self, '_restricted_fields', set())
        if restricted_fields:
            requested_fields = context.get("fields", [])
            if any(f in restricted_fields for f in (requested_fields or [])):
                return False

        # Rate limiting check (simple token bucket)
        # Note: Not thread-safe - in multi-worker FastAPI, each worker has its own
        # rate bucket. Within a single worker with async concurrency, races are
        # possible but non-critical (may slightly over/under count).
        rate_limit = getattr(self, '_rate_limit', None)
        if rate_limit:
            caller = context.get("caller", "default")
            now = time.time()
            if not hasattr(self, '_rate_tokens'):
                self._rate_tokens = {}
            tokens = self._rate_tokens.get(caller, {"count": 0, "reset_at": now})
            if now > tokens["reset_at"]:
                tokens = {"count": 0, "reset_at": now + 60}  # Reset every minute
            if tokens["count"] >= rate_limit:
                return False
            tokens["count"] += 1
            self._rate_tokens[caller] = tokens

        return True

    def configure_policy(
        self,
        disabled: bool = False,
        restricted_fields: list = None,
        rate_limit: int = None
    ) -> None:
        """
        Configure policy enforcement.

        Args:
            disabled: If True, bypass all policy checks
            restricted_fields: List of field names to block
            rate_limit: Max queries per minute per caller (None = unlimited)
        """
        self._policy_disabled = disabled
        self._restricted_fields = set(restricted_fields or [])
        self._rate_limit = rate_limit

    def query(
        self,
        target: str,
        fields: Optional[List[str]] = None,
        context: Optional[dict] = None
    ) -> MetadataResponse:
        """
        Query metadata for a target.

        Args:
            target: Query target - "current", "msg:<uid>", "thread", "range:<n>", "turn:<n>"
            fields: Optional list of specific fields to return
            context: Query context (thread_uid, loom reference, etc.)

        Returns:
            MetadataResponse with requested data
        """
        context = context or {}

        # Policy check
        if not self.check_policy(target, context):
            return MetadataResponse(ok=False, error="policy_denied")

        # Parse target
        target_type, target_ref = self._parse_target(target)
        if target_type is None:
            return MetadataResponse(ok=False, error="invalid_target")

        # Determine which providers to query
        if fields:
            provider_names = set(self._field_map.get(f) for f in fields if f in self._field_map)
            provider_names.discard(None)
        else:
            provider_names = set(self._providers.keys())

        # Collect metadata from providers
        result = {"t": target_type}
        if target_ref:
            result["ref"] = target_ref

        query_context = {
            **context,
            "target_type": target_type,
            "target_ref": target_ref,
            "query_ts": int(time.time())
        }

        for pname in provider_names:
            provider = self._providers.get(pname)
            if provider:
                try:
                    provider_data = provider.get(target, fields, query_context)
                    result.update(provider_data)
                except Exception as e:
                    # Don't fail entire query for one provider error
                    result[f"_{pname}_err"] = str(e)[:50]

        return MetadataResponse(ok=True, data=result)

    def _parse_target(self, target: str) -> tuple:
        """Parse target string into (type, ref)."""
        target = target.strip().lower()

        if target == "current":
            return ("current", None)

        if target == "thread":
            return ("thread", None)

        if target.startswith("msg:"):
            ref = target[4:].strip()
            return ("msg", ref) if ref else (None, None)

        if target.startswith("range:"):
            try:
                n = int(target[6:].strip())
                return ("range", n)
            except ValueError:
                return (None, None)

        if target.startswith("turn:"):
            try:
                n = int(target[5:].strip())
                return ("turn", n)
            except ValueError:
                return (None, None)

        return (None, None)


# Global channel instance
_channel: Optional[MetadataChannel] = None


def get_channel() -> MetadataChannel:
    """Get or create the global MetadataChannel instance."""
    global _channel
    if _channel is None:
        _channel = MetadataChannel()
        # Register built-in providers
        from cathedral.MetadataChannel.providers import (
            TemporalProvider,
            PositionalProvider,
            TokenProvider,
            ThreadProvider
        )
        _channel.register_provider(TemporalProvider())
        _channel.register_provider(PositionalProvider())
        _channel.register_provider(TokenProvider())
        _channel.register_provider(ThreadProvider())
    return _channel


def query(
    target: str,
    fields: Optional[List[str]] = None,
    context: Optional[dict] = None
) -> dict:
    """
    Query metadata (convenience function).

    Args:
        target: "current", "msg:<uid>", "thread", "range:<n>", "turn:<n>"
        fields: Optional specific fields
        context: Must include 'loom' and 'thread_uid' for most queries

    Returns:
        Compact dict with metadata
    """
    channel = get_channel()
    response = channel.query(target, fields, context)
    return response.to_dict()


def register_provider(provider: MetadataProvider) -> None:
    """Register a custom metadata provider."""
    get_channel().register_provider(provider)
