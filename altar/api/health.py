"""
Health check API endpoint.

Aggregates health status from all Cathedral Gates.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    async def api_health() -> Dict[str, Any]:
        """
        Get aggregated health status from all Gates.

        Returns:
            Dict with overall health and per-gate status
        """
        from cathedral import (
            BrowserGate,
            FileSystemGate,
            PersonalityGate,
            ShellGate,
        )
        from cathedral import MemoryGate
        from cathedral import ScriptureGate
        from cathedral import SecurityManager

        gates = {}
        all_healthy = True

        # FileSystemGate
        try:
            gates["FileSystemGate"] = FileSystemGate.get_health_status()
            if not gates["FileSystemGate"].get("healthy", False):
                all_healthy = False
        except Exception as e:
            gates["FileSystemGate"] = {"healthy": False, "error": str(e)}
            all_healthy = False

        # ShellGate
        try:
            gates["ShellGate"] = ShellGate.get_health_status()
            if not gates["ShellGate"].get("healthy", False):
                all_healthy = False
        except Exception as e:
            gates["ShellGate"] = {"healthy": False, "error": str(e)}
            all_healthy = False

        # PersonalityGate
        try:
            from cathedral.PersonalityGate import PersonalityManager
            gates["PersonalityGate"] = PersonalityManager.get_health_status()
            if not gates["PersonalityGate"].get("healthy", False):
                all_healthy = False
        except Exception as e:
            gates["PersonalityGate"] = {"healthy": False, "error": str(e)}
            all_healthy = False

        # BrowserGate
        try:
            gates["BrowserGate"] = BrowserGate.get_health_status()
            if not gates["BrowserGate"].get("healthy", False):
                all_healthy = False
        except Exception as e:
            gates["BrowserGate"] = {"healthy": False, "error": str(e)}
            all_healthy = False

        # MemoryGate
        try:
            gates["MemoryGate"] = MemoryGate.get_health_status()
            if not gates["MemoryGate"].get("healthy", False):
                all_healthy = False
        except Exception as e:
            gates["MemoryGate"] = {"healthy": False, "error": str(e)}
            all_healthy = False

        # ScriptureGate
        try:
            gates["ScriptureGate"] = ScriptureGate.get_health_status()
            if not gates["ScriptureGate"].get("healthy", False):
                all_healthy = False
        except Exception as e:
            gates["ScriptureGate"] = {"healthy": False, "error": str(e)}
            all_healthy = False

        # SecurityManager
        try:
            gates["SecurityManager"] = {
                "gate": "SecurityManager",
                "healthy": True,
                "initialized": SecurityManager.is_encryption_enabled() or True,
                "encryption_enabled": SecurityManager.is_encryption_enabled(),
                "unlocked": SecurityManager.is_unlocked() if SecurityManager.is_encryption_enabled() else None,
            }
        except Exception as e:
            gates["SecurityManager"] = {"healthy": False, "error": str(e)}
            all_healthy = False

        return {
            "healthy": all_healthy,
            "gates": gates,
        }

    @router.get("/api/health/gate/{gate_name}")
    async def api_health_gate(gate_name: str) -> Dict[str, Any]:
        """
        Get health status for a specific gate.

        Args:
            gate_name: Name of the gate (e.g., FileSystemGate, ShellGate)

        Returns:
            Health status for the specified gate
        """
        gate_map = {
            "filesystemgate": ("cathedral", "FileSystemGate"),
            "shellgate": ("cathedral", "ShellGate"),
            "personalitygate": ("cathedral.PersonalityGate", "PersonalityManager"),
            "browsergate": ("cathedral", "BrowserGate"),
            "memorygate": ("cathedral", "MemoryGate"),
            "scripturegate": ("cathedral", "ScriptureGate"),
        }

        key = gate_name.lower()
        if key not in gate_map:
            return {
                "error": f"Unknown gate: {gate_name}",
                "available": list(gate_map.keys()),
            }

        module_name, class_name = gate_map[key]

        try:
            import importlib
            module = importlib.import_module(module_name)
            gate_class = getattr(module, class_name)

            if hasattr(gate_class, "get_health_status"):
                return gate_class.get_health_status()
            else:
                return {"error": f"{gate_name} does not have health check methods"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    @router.get("/api/health/summary")
    async def api_health_summary() -> Dict[str, Any]:
        """
        Get a quick health summary (just healthy/unhealthy per gate).
        """
        full_health = await api_health()

        summary = {}
        for gate_name, status in full_health.get("gates", {}).items():
            summary[gate_name] = status.get("healthy", False)

        return {
            "healthy": full_health.get("healthy", False),
            "gates": summary,
        }

    return router


__all__ = ["create_router"]
