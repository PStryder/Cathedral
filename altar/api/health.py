"""
Health check API endpoint.

Aggregates health status from all Cathedral Gates.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Response


# Gate registry: name -> (module_path, class_name, health_method)
# health_method is optional - defaults to "get_health_status"
GATE_REGISTRY: Dict[str, Tuple[str, str, Optional[str]]] = {
    "FileSystemGate": ("cathedral", "FileSystemGate", None),
    "ShellGate": ("cathedral", "ShellGate", None),
    "PersonalityGate": ("cathedral.PersonalityGate", "PersonalityManager", None),
    "BrowserGate": ("cathedral", "BrowserGate", None),
    "MemoryGate": ("cathedral", "MemoryGate", None),
    "ScriptureGate": ("cathedral", "ScriptureGate", None),
    "ToolGate": ("cathedral", "ToolGate", None),
}


def _get_gate_health(
    module_path: str,
    class_name: str,
    health_method: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get health status from a gate module.

    Args:
        module_path: Python module path (e.g., "cathedral" or "cathedral.PersonalityGate")
        class_name: Class or module attribute name
        health_method: Method name to call (defaults to "get_health_status")

    Returns:
        Health status dict
    """
    import importlib

    method_name = health_method or "get_health_status"

    module = importlib.import_module(module_path)
    gate = getattr(module, class_name)

    if hasattr(gate, method_name):
        return getattr(gate, method_name)()
    else:
        return {"healthy": False, "error": f"No {method_name} method"}


def _get_security_manager_health() -> Dict[str, Any]:
    """
    Get SecurityManager health status.

    SecurityManager is special - it's always "available" if crypto deps are present,
    but encryption may or may not be enabled/unlocked.
    """
    from cathedral import SecurityManager

    # Check if crypto dependencies are available
    try:
        crypto_available = SecurityManager.is_available()
    except AttributeError:
        # Fallback: if is_available doesn't exist, check if we can import crypto
        try:
            import argon2
            import cryptography
            crypto_available = True
        except ImportError:
            crypto_available = False

    encryption_enabled = False
    unlocked = None

    try:
        encryption_enabled = SecurityManager.is_encryption_enabled()
        if encryption_enabled:
            unlocked = SecurityManager.is_unlocked()
    except Exception:
        pass

    return {
        "gate": "SecurityManager",
        "healthy": crypto_available,  # Healthy if crypto deps available
        "initialized": True,  # Module is always initialized
        "details": {
            "crypto_available": crypto_available,
            "encryption_enabled": encryption_enabled,
            "unlocked": unlocked,
        },
    }


def _collect_health_data() -> Tuple[bool, Dict[str, Any]]:
    """
    Collect health data from all gates.

    Returns:
        Tuple of (all_healthy, gates_dict)
    """
    gates = {}
    all_healthy = True

    # Check all registered gates
    for gate_name, (module_path, class_name, health_method) in GATE_REGISTRY.items():
        try:
            gates[gate_name] = _get_gate_health(module_path, class_name, health_method)
            if not gates[gate_name].get("healthy", False):
                all_healthy = False
        except Exception as e:
            gates[gate_name] = {"healthy": False, "error": str(e)}
            all_healthy = False

    # SecurityManager (special handling)
    try:
        gates["SecurityManager"] = _get_security_manager_health()
        if not gates["SecurityManager"].get("healthy", False):
            all_healthy = False
    except Exception as e:
        gates["SecurityManager"] = {"healthy": False, "error": str(e)}
        all_healthy = False

    return all_healthy, gates


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    async def api_health(response: Response) -> Dict[str, Any]:
        """
        Get aggregated health status from all Gates.

        Returns 200 when healthy, 503 when unhealthy.

        Health semantics:
        - available: dependencies exist, can be imported, external services reachable
        - initialized: gate has been set up in this process
        - healthy: initialized + all checks pass

        Returns:
            Dict with overall health and per-gate status
        """
        all_healthy, gates = _collect_health_data()

        if not all_healthy:
            response.status_code = 503

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
        # Normalize gate name for lookup
        normalized = gate_name.lower()

        # Check registered gates
        for name, (module_path, class_name, health_method) in GATE_REGISTRY.items():
            if name.lower() == normalized:
                try:
                    return _get_gate_health(module_path, class_name, health_method)
                except Exception as e:
                    return {"healthy": False, "error": str(e)}

        # Check SecurityManager
        if normalized == "securitymanager":
            try:
                return _get_security_manager_health()
            except Exception as e:
                return {"healthy": False, "error": str(e)}

        return {
            "error": f"Unknown gate: {gate_name}",
            "available": list(GATE_REGISTRY.keys()) + ["SecurityManager"],
        }

    @router.get("/api/health/summary")
    async def api_health_summary(response: Response) -> Dict[str, Any]:
        """
        Get a quick health summary (just healthy/unhealthy per gate).

        Returns 200 when healthy, 503 when unhealthy.
        """
        all_healthy, gates = _collect_health_data()

        summary = {name: status.get("healthy", False) for name, status in gates.items()}

        if not all_healthy:
            response.status_code = 503

        return {
            "healthy": all_healthy,
            "gates": summary,
        }

    return router


__all__ = ["create_router", "GATE_REGISTRY"]
