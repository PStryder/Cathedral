"""
SubAgentGate - Spawn and manage async sub-agent processes.

Allows the main agent to delegate tasks to worker agents that run
independently and report back when complete.
"""

import os
import sys
import json
import uuid
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class AgentStatus(Enum):
    """Status of a sub-agent."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubAgent:
    """Represents a spawned sub-agent task."""
    id: str
    task: str
    status: AgentStatus = AgentStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    # Runtime state (not serialized)
    process: Optional[subprocess.Popen] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize to dict (excludes process handle)."""
        return {
            "id": self.id,
            "task": self.task[:100] + "..." if len(self.task) > 100 else self.task,
            "status": self.status.value,
            "result": self.result[:200] + "..." if self.result and len(self.result) > 200 else self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def to_summary(self) -> str:
        """One-line summary."""
        status_icon = {
            AgentStatus.PENDING: "[.]",
            AgentStatus.RUNNING: "[~]",
            AgentStatus.COMPLETED: "[+]",
            AgentStatus.FAILED: "[!]",
            AgentStatus.CANCELLED: "[x]",
        }.get(self.status, "[?]")
        task_short = self.task[:50] + "..." if len(self.task) > 50 else self.task
        return f"{status_icon} {self.id[:8]} | {task_short}"


# Storage directory for agent results
AGENT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "agents"
AGENT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _result_path(agent_id: str) -> Path:
    """Get path to agent result file."""
    return AGENT_DATA_DIR / f"{agent_id}.json"


def _save_agent_result(agent_id: str, result: dict) -> None:
    """Save agent result to file."""
    with open(_result_path(agent_id), "w") as f:
        json.dump(result, f, indent=2)


def _load_agent_result(agent_id: str) -> Optional[dict]:
    """Load agent result from file."""
    path = _result_path(agent_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


class SubAgentManager:
    """
    Manages spawned sub-agents.

    Tracks running agents, polls for completion, and provides
    an interface for the main agent to interact with sub-agents.
    """

    def __init__(self):
        self.agents: Dict[str, SubAgent] = {}
        self._worker_script = Path(__file__).parent / "worker.py"

    def spawn(
        self,
        task: str,
        context: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        personality: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Spawn a new sub-agent to handle a task.

        Args:
            task: The task description/instructions for the agent
            context: Optional context dict to pass to agent
            system_prompt: Optional system prompt override
            max_tokens: Max response tokens
            temperature: Sampling temperature
            personality: Personality ID to use (overrides other settings)
            model: Model override (personality takes precedence)

        Returns:
            Agent ID
        """
        agent_id = str(uuid.uuid4())[:8]

        agent = SubAgent(
            id=agent_id,
            task=task,
            context=context or {},
            status=AgentStatus.PENDING
        )

        # Prepare task file (subprocess reads this)
        task_data = {
            "id": agent_id,
            "task": task,
            "context": context or {},
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "personality": personality,
            "model": model,
        }

        task_file = AGENT_DATA_DIR / f"{agent_id}.task.json"
        with open(task_file, "w") as f:
            json.dump(task_data, f)

        # Spawn subprocess
        # Note: Use DEVNULL to avoid pipe deadlock. Results are written to files.
        # If stderr is needed for debugging, the worker should write to a log file.
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

            process = subprocess.Popen(
                [sys.executable, str(self._worker_script), agent_id],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parents[2])
            )

            agent.process = process
            agent.status = AgentStatus.RUNNING
            agent.started_at = datetime.utcnow().isoformat()

        except Exception as e:
            agent.status = AgentStatus.FAILED
            agent.error = str(e)

        self.agents[agent_id] = agent
        return agent_id

    def get_agent(self, agent_id: str) -> Optional[SubAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    def get_status(self, agent_id: str) -> Optional[dict]:
        """Get agent status as dict."""
        agent = self.agents.get(agent_id)
        if not agent:
            return None
        self._check_agent(agent)
        return agent.to_dict()

    def list_agents(self, include_completed: bool = True) -> List[dict]:
        """List all agents."""
        self._check_all()
        agents = list(self.agents.values())
        if not include_completed:
            agents = [a for a in agents if a.status == AgentStatus.RUNNING]
        return [a.to_dict() for a in agents]

    def get_result(self, agent_id: str) -> Optional[str]:
        """Get the result of a completed agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return None
        self._check_agent(agent)
        return agent.result

    def cancel(self, agent_id: str) -> bool:
        """Cancel a running agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return False

        if agent.status != AgentStatus.RUNNING:
            return False

        if agent.process:
            try:
                agent.process.terminate()
                agent.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                agent.process.kill()
            except Exception as e:
                # Log but don't fail - process may already be dead
                import logging
                logging.getLogger(__name__).warning(
                    f"Error terminating agent {agent_id} process: {e}"
                )

        agent.status = AgentStatus.CANCELLED
        agent.completed_at = datetime.utcnow().isoformat()
        return True

    def check_completed(self) -> List[str]:
        """
        Check for newly completed agents.
        Returns list of agent IDs that just completed.
        """
        newly_completed = []

        for agent_id, agent in self.agents.items():
            if agent.status == AgentStatus.RUNNING:
                old_status = agent.status
                self._check_agent(agent)
                if agent.status in (AgentStatus.COMPLETED, AgentStatus.FAILED):
                    newly_completed.append(agent_id)

        return newly_completed

    def _check_agent(self, agent: SubAgent) -> None:
        """Check if agent process has completed."""
        if agent.status != AgentStatus.RUNNING:
            return

        if agent.process is None:
            return

        # Check if process finished
        retcode = agent.process.poll()
        if retcode is None:
            return  # Still running

        # Process completed
        agent.completed_at = datetime.utcnow().isoformat()

        # Load result from file
        result = _load_agent_result(agent.id)

        if result:
            if result.get("status") == "completed":
                agent.status = AgentStatus.COMPLETED
                agent.result = result.get("result", "")
            else:
                agent.status = AgentStatus.FAILED
                agent.error = result.get("error", "Unknown error")
        else:
            # No result file - process failed before writing result
            agent.status = AgentStatus.FAILED
            agent.error = f"Process exited with code {retcode} (no result file)"

    def _check_all(self) -> None:
        """Check all running agents."""
        for agent in self.agents.values():
            self._check_agent(agent)

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Remove old completed agents. Returns count removed."""
        now = datetime.utcnow()
        to_remove = []

        for agent_id, agent in self.agents.items():
            if agent.status in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED):
                if agent.completed_at:
                    completed = datetime.fromisoformat(agent.completed_at)
                    age = (now - completed).total_seconds() / 3600
                    if age > max_age_hours:
                        to_remove.append(agent_id)

        for agent_id in to_remove:
            del self.agents[agent_id]
            # Clean up files
            result_path = _result_path(agent_id)
            task_path = AGENT_DATA_DIR / f"{agent_id}.task.json"
            for p in [result_path, task_path]:
                if p.exists():
                    p.unlink()

        return len(to_remove)


# Global manager instance
_manager: Optional[SubAgentManager] = None


def get_manager() -> SubAgentManager:
    """Get or create the global SubAgentManager."""
    global _manager
    if _manager is None:
        _manager = SubAgentManager()
    return _manager


# Convenience functions
def spawn(task: str, **kwargs) -> str:
    """Spawn a sub-agent. Returns agent ID."""
    return get_manager().spawn(task, **kwargs)


def status(agent_id: str) -> Optional[dict]:
    """Get agent status."""
    return get_manager().get_status(agent_id)


def result(agent_id: str) -> Optional[str]:
    """Get agent result."""
    return get_manager().get_result(agent_id)


def list_agents(include_completed: bool = True) -> List[dict]:
    """List all agents."""
    return get_manager().list_agents(include_completed)


def cancel(agent_id: str) -> bool:
    """Cancel agent."""
    return get_manager().cancel(agent_id)


def check_completed() -> List[str]:
    """Check for newly completed agents."""
    return get_manager().check_completed()


__all__ = [
    # Class
    "SubAgentManager",
    # Lifecycle
    "get_manager",
    # Agent operations
    "spawn",
    "status",
    "result",
    "list_agents",
    "cancel",
    "check_completed",
    # Models
    "SubAgent",
    "AgentStatus",
]
