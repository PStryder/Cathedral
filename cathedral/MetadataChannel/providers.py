"""
Built-in metadata providers for MetadataChannel.

Each provider supplies specific metadata fields.
Responses use compact keys to minimize token usage.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import time


class TemporalProvider:
    """Provides timestamp-related metadata."""

    name = "temporal"
    fields = ["ts", "elapsed", "gap", "created"]

    def get(self, target: str, fields: Optional[List[str]], context: dict) -> dict:
        """Get temporal metadata."""
        result = {}
        target_type = context.get("target_type")
        target_ref = context.get("target_ref")
        loom = context.get("loom")
        thread_uid = context.get("thread_uid")

        if target_type == "current":
            result["ts"] = context.get("query_ts", int(time.time()))
            if loom and thread_uid:
                messages = loom.recall(thread_uid)
                if messages:
                    last = messages[-1]
                    last_ts = self._parse_ts(last.get("timestamp"))
                    if last_ts:
                        result["elapsed"] = context.get("query_ts", int(time.time())) - last_ts

        elif target_type == "msg" and loom and thread_uid:
            messages = loom.recall(thread_uid)
            for i, m in enumerate(messages):
                if m.get("message_uid") == target_ref or self._match_partial(m, target_ref):
                    ts = self._parse_ts(m.get("timestamp"))
                    if ts:
                        result["ts"] = ts
                    # Gap from previous message
                    if i > 0:
                        prev_ts = self._parse_ts(messages[i-1].get("timestamp"))
                        if prev_ts and ts:
                            result["gap"] = ts - prev_ts
                    break

        elif target_type == "thread" and loom and thread_uid:
            threads = loom.list_all_threads()
            for t in threads:
                if t.get("thread_uid") == thread_uid:
                    created = self._parse_ts(t.get("created_at"))
                    if created:
                        result["created"] = created
                    break

        elif target_type == "range" and loom and thread_uid:
            n = target_ref
            messages = loom.recall(thread_uid)
            if messages:
                recent = messages[-n:] if n <= len(messages) else messages
                result["range_ts"] = [
                    self._parse_ts(m.get("timestamp")) for m in recent
                ]

        return result

    def _parse_ts(self, ts_str: Optional[str]) -> Optional[int]:
        """Parse ISO timestamp to unix epoch."""
        if not ts_str:
            return None
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except (ValueError, AttributeError):
            return None

    def _match_partial(self, message: dict, ref: str) -> bool:
        """Match partial message UID."""
        uid = message.get("message_uid", "")
        return uid.startswith(ref) or uid.endswith(ref)


class PositionalProvider:
    """Provides position-related metadata."""

    name = "positional"
    fields = ["turn", "pos", "total", "role"]

    def get(self, target: str, fields: Optional[List[str]], context: dict) -> dict:
        """Get positional metadata."""
        result = {}
        target_type = context.get("target_type")
        target_ref = context.get("target_ref")
        loom = context.get("loom")
        thread_uid = context.get("thread_uid")

        if not loom or not thread_uid:
            return result

        messages = loom.recall(thread_uid)
        total = len(messages)

        if target_type == "current":
            result["total"] = total
            if messages:
                # Current turn = number of complete exchanges + 1
                user_msgs = sum(1 for m in messages if m.get("role") == "user")
                result["turn"] = user_msgs
                result["pos"] = f"{total}/{total}"

        elif target_type == "msg":
            for i, m in enumerate(messages):
                if m.get("message_uid") == target_ref or self._match_partial(m, target_ref):
                    result["pos"] = f"{i+1}/{total}"
                    result["role"] = self._short_role(m.get("role"))
                    # Turn number (count user messages up to this point)
                    user_count = sum(1 for msg in messages[:i+1] if msg.get("role") == "user")
                    result["turn"] = user_count
                    break

        elif target_type == "turn":
            turn_num = target_ref
            # Find messages at this turn
            user_count = 0
            for i, m in enumerate(messages):
                if m.get("role") == "user":
                    user_count += 1
                if user_count == turn_num:
                    result["pos"] = f"{i+1}/{total}"
                    result["role"] = self._short_role(m.get("role"))
                    # Include next message if it's assistant response
                    if i + 1 < len(messages) and messages[i+1].get("role") == "assistant":
                        result["pair"] = f"{i+1}-{i+2}/{total}"
                    break

        elif target_type == "range":
            n = target_ref
            recent = messages[-n:] if n <= len(messages) else messages
            start_idx = max(0, total - n)
            result["range_pos"] = [
                {"pos": f"{start_idx+i+1}/{total}", "role": self._short_role(m.get("role"))}
                for i, m in enumerate(recent)
            ]

        elif target_type == "thread":
            result["total"] = total
            user_msgs = sum(1 for m in messages if m.get("role") == "user")
            result["turns"] = user_msgs

        return result

    def _short_role(self, role: Optional[str]) -> str:
        """Abbreviate role to single char."""
        if not role:
            return "?"
        return {"user": "u", "assistant": "a", "system": "s"}.get(role, role[0])

    def _match_partial(self, message: dict, ref: str) -> bool:
        """Match partial message UID."""
        uid = message.get("message_uid", "")
        return uid.startswith(ref) or uid.endswith(ref)


class TokenProvider:
    """Provides token estimation metadata."""

    name = "token"
    fields = ["tok", "tok_total", "tok_avg"]

    def get(self, target: str, fields: Optional[List[str]], context: dict) -> dict:
        """Get token metadata."""
        result = {}
        target_type = context.get("target_type")
        target_ref = context.get("target_ref")
        loom = context.get("loom")
        thread_uid = context.get("thread_uid")

        if not loom or not thread_uid:
            return result

        messages = loom.recall(thread_uid)

        if target_type == "current":
            total_tok = sum(self._estimate_tokens(m.get("content", "")) for m in messages)
            result["tok_total"] = total_tok
            if messages:
                result["tok_avg"] = total_tok // len(messages)

        elif target_type == "msg":
            for m in messages:
                if m.get("message_uid") == target_ref or self._match_partial(m, target_ref):
                    result["tok"] = self._estimate_tokens(m.get("content", ""))
                    break

        elif target_type == "range":
            n = target_ref
            recent = messages[-n:] if n <= len(messages) else messages
            result["range_tok"] = [
                self._estimate_tokens(m.get("content", "")) for m in recent
            ]
            result["range_tok_sum"] = sum(result["range_tok"])

        elif target_type == "thread":
            total_tok = sum(self._estimate_tokens(m.get("content", "")) for m in messages)
            result["tok_total"] = total_tok
            if messages:
                result["tok_avg"] = total_tok // len(messages)

        return result

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (words * 1.3)."""
        if not text:
            return 0
        # Simple heuristic: ~1.3 tokens per word for English
        words = len(text.split())
        return int(words * 1.3)

    def _match_partial(self, message: dict, ref: str) -> bool:
        """Match partial message UID."""
        uid = message.get("message_uid", "")
        return uid.startswith(ref) or uid.endswith(ref)


class ThreadProvider:
    """Provides thread-level metadata."""

    name = "thread"
    fields = ["thread_name", "thread_uid", "msg_count", "active"]

    def get(self, target: str, fields: Optional[List[str]], context: dict) -> dict:
        """Get thread metadata."""
        result = {}
        target_type = context.get("target_type")
        loom = context.get("loom")
        thread_uid = context.get("thread_uid")

        if not loom:
            return result

        if target_type in ("current", "thread"):
            threads = loom.list_all_threads()
            for t in threads:
                if t.get("thread_uid") == thread_uid:
                    result["thread_name"] = t.get("thread_name")
                    result["thread_uid"] = thread_uid[:8]  # Short ref
                    result["active"] = t.get("is_active", False)
                    break

            if thread_uid:
                messages = loom.recall(thread_uid)
                result["msg_count"] = len(messages)

        return result
