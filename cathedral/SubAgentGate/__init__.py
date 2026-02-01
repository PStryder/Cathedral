"""
SubAgentGate - Spawn and manage async sub-agent processes.

Allows the main agent to delegate tasks to worker agents that run
independently and report back when complete.
"""

import os
import sys
import json
import time
import uuid
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class AgentStatus(Enum):
    """Status of a sub-agent."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(Enum):
    """Type of sub-agent execution backend."""
    LLM = "llm"              # Simple LLM completion via StarMirror
    CLAUDE_CODE = "claude_code"  # Claude Code CLI (agentic with file/tool access)
    CODEX = "codex"          # Codex CLI (agentic)


@dataclass
class SubAgent:
    """Represents a spawned sub-agent task."""
    id: str
    task: str
    agent_type: AgentType = AgentType.LLM
    status: AgentStatus = AgentStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    working_dir: Optional[str] = None  # For CLI agents
    pid: Optional[int] = None

    # Runtime state (not serialized)
    process: Optional[subprocess.Popen] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize to dict (excludes process handle)."""
        return {
            "id": self.id,
            "task": self.task[:100] + "..." if len(self.task) > 100 else self.task,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "result": self.result[:200] + "..." if self.result and len(self.result) > 200 else self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "working_dir": self.working_dir,
            "pid": self.pid,
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
        type_badge = f"[{self.agent_type.value}]" if self.agent_type != AgentType.LLM else ""
        task_short = self.task[:50] + "..." if len(self.task) > 50 else self.task
        return f"{status_icon} {self.id[:8]} {type_badge} | {task_short}"


# Storage directory for agent results
AGENT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "agents"
AGENT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _task_path(agent_id: str) -> Path:
    """Get path to agent task file."""
    return AGENT_DATA_DIR / f"{agent_id}.task.json"


def _result_path(agent_id: str) -> Path:
    """Get path to agent result file."""
    return AGENT_DATA_DIR / f"{agent_id}.json"


def _save_agent_result(agent_id: str, result: dict) -> None:
    """Save agent result to file."""
    with open(_result_path(agent_id), "w") as f:
        json.dump(result, f, indent=2)


def _load_agent_result(agent_id: str, retries: int = 5, delay: float = 0.1) -> Optional[dict]:
    """
    Load agent result from file with retry for race condition.

    The worker might still be writing when we first check, so we retry
    a few times with a short delay.
    """
    path = _result_path(agent_id)

    for attempt in range(retries):
        if not path.exists():
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            return None

        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # File might be partially written, retry
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            return None

    return None


def _load_task_data(agent_id: str) -> Optional[dict]:
    """Load task data from file."""
    path = _task_path(agent_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _is_process_alive(pid: int) -> bool:
    """Check if a process with given PID is still running."""
    if pid is None:
        return False
    try:
        # This works on both Windows and Unix
        # os.kill with signal 0 doesn't kill, just checks existence
        if sys.platform == "win32":
            # Windows: use tasklist or ctypes
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False
    except Exception:
        # If we can't determine, assume not alive
        return False


def _generate_unique_id() -> str:
    """Generate a unique agent ID, ensuring no file collision."""
    for _ in range(100):  # Limit attempts to prevent infinite loop
        agent_id = str(uuid.uuid4())[:8]
        if not _task_path(agent_id).exists() and not _result_path(agent_id).exists():
            return agent_id
    # Fallback to full UUID if short IDs keep colliding
    return str(uuid.uuid4())


class SubAgentManager:
    """
    Manages spawned sub-agents.

    Tracks running agents, polls for completion, and provides
    an interface for the main agent to interact with sub-agents.

    Persists agent state to files and reconstructs on restart.
    """

    def __init__(self):
        self.agents: Dict[str, SubAgent] = {}
        self._worker_script = Path(__file__).parent / "worker.py"
        self._cli_worker_script = Path(__file__).parent / "cli_worker.py"
        self._reconstruct_from_files()

    def _reconstruct_from_files(self) -> None:
        """
        Reconstruct agent state from files on startup.

        This handles the case where the server restarts while agents
        were running or completed.
        """
        # Find all task files
        for task_file in AGENT_DATA_DIR.glob("*.task.json"):
            agent_id = task_file.stem.replace(".task", "")
            if agent_id in self.agents:
                continue  # Already tracked

            task_data = _load_task_data(agent_id)
            if not task_data:
                continue

            # Check if result exists
            result_data = _load_agent_result(agent_id, retries=1)

            # Parse agent type
            agent_type_str = task_data.get("agent_type", "llm")
            try:
                agent_type = AgentType(agent_type_str)
            except ValueError:
                agent_type = AgentType.LLM

            agent = SubAgent(
                id=agent_id,
                task=task_data.get("task", ""),
                agent_type=agent_type,
                context=task_data.get("context", {}),
                working_dir=task_data.get("working_dir"),
                created_at=task_data.get("created_at", datetime.utcnow().isoformat()),
                started_at=task_data.get("started_at"),
                pid=task_data.get("pid"),
            )

            if result_data:
                # Agent completed
                if result_data.get("status") == "completed":
                    agent.status = AgentStatus.COMPLETED
                    agent.result = result_data.get("result", "")
                else:
                    agent.status = AgentStatus.FAILED
                    agent.error = result_data.get("error", "Unknown error")
                agent.completed_at = result_data.get("completed_at", datetime.utcnow().isoformat())
            elif agent.pid and _is_process_alive(agent.pid):
                # Process still running
                agent.status = AgentStatus.RUNNING
            else:
                # No result, process not alive - orphaned/crashed
                agent.status = AgentStatus.FAILED
                agent.error = "Worker process died without producing result (recovered after restart)"
                agent.completed_at = datetime.utcnow().isoformat()

            self.agents[agent_id] = agent

    def spawn(
        self,
        task: str,
        context: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        personality: Optional[str] = None,
        model: Optional[str] = None,
        agent_type: str = "llm",
        working_dir: Optional[str] = None,
    ) -> str:
        """
        Spawn a new sub-agent to handle a task.

        Args:
            task: The task description/instructions for the agent
            context: Optional context dict to pass to agent
            system_prompt: Optional system prompt override
            max_tokens: Max response tokens (for LLM type)
            temperature: Sampling temperature (for LLM type)
            personality: Personality ID to use (for LLM type)
            model: Model override (for LLM type)
            agent_type: "llm" (default), "claude_code", or "codex"
            working_dir: Working directory for CLI agents (defaults to project root)

        Returns:
            Agent ID
        """
        agent_id = _generate_unique_id()

        # Parse agent type
        try:
            parsed_type = AgentType(agent_type.lower())
        except ValueError:
            parsed_type = AgentType.LLM

        # Resolve working directory for CLI agents
        if parsed_type in (AgentType.CLAUDE_CODE, AgentType.CODEX):
            if working_dir:
                working_dir = str(Path(working_dir).resolve())
            else:
                # Default to project root
                working_dir = str(Path(__file__).resolve().parents[2])

        agent = SubAgent(
            id=agent_id,
            task=task,
            agent_type=parsed_type,
            context=context or {},
            working_dir=working_dir,
            status=AgentStatus.PENDING
        )

        # Save task file first (worker reads it)
        task_data = {
            "id": agent_id,
            "task": task,
            "agent_type": parsed_type.value,
            "context": context or {},
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "personality": personality,
            "model": model,
            "working_dir": working_dir,
            "created_at": agent.created_at,
        }

        task_file = _task_path(agent_id)
        with open(task_file, "w") as f:
            json.dump(task_data, f)

        # Spawn subprocess
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

            # Choose worker based on agent type
            if parsed_type in (AgentType.CLAUDE_CODE, AgentType.CODEX):
                worker_script = self._cli_worker_script
                cwd = working_dir
            else:
                worker_script = self._worker_script
                cwd = str(Path(__file__).resolve().parents[2])

            process = subprocess.Popen(
                [sys.executable, str(worker_script), agent_id],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=cwd
            )

            agent.process = process
            agent.pid = process.pid
            agent.status = AgentStatus.RUNNING
            agent.started_at = datetime.utcnow().isoformat()

            # Update task file with PID
            task_data["pid"] = agent.pid
            task_data["started_at"] = agent.started_at
            with open(task_file, "w") as f:
                json.dump(task_data, f)

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

        terminated = False

        # Try process handle first (if we spawned it this session)
        if agent.process:
            try:
                agent.process.terminate()
                agent.process.wait(timeout=5)
                terminated = True
            except subprocess.TimeoutExpired:
                agent.process.kill()
                terminated = True
            except Exception:
                pass

        # If no process handle but we have PID, try to kill by PID
        if not terminated and agent.pid:
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/PID", str(agent.pid)],
                                   capture_output=True, timeout=5)
                else:
                    os.kill(agent.pid, 9)  # SIGKILL
                terminated = True
            except Exception:
                pass

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

        process_done = False

        # Check via process handle if available
        if agent.process is not None:
            retcode = agent.process.poll()
            if retcode is not None:
                process_done = True
        # Otherwise check via PID
        elif agent.pid:
            if not _is_process_alive(agent.pid):
                process_done = True

        if not process_done:
            return  # Still running

        # Process completed - try to load result with retry
        agent.completed_at = datetime.utcnow().isoformat()
        result = _load_agent_result(agent.id, retries=5, delay=0.1)

        if result:
            if result.get("status") == "completed":
                agent.status = AgentStatus.COMPLETED
                agent.result = result.get("result", "")
            else:
                agent.status = AgentStatus.FAILED
                agent.error = result.get("error", "Unknown error")
        else:
            # No result file after retries
            agent.status = AgentStatus.FAILED
            agent.error = "Worker process exited without producing result file"

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
                    try:
                        completed = datetime.fromisoformat(agent.completed_at)
                        age = (now - completed).total_seconds() / 3600
                        if age > max_age_hours:
                            to_remove.append(agent_id)
                    except ValueError:
                        # Invalid timestamp, mark for cleanup
                        to_remove.append(agent_id)

        for agent_id in to_remove:
            del self.agents[agent_id]
            # Clean up files
            for p in [_result_path(agent_id), _task_path(agent_id)]:
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass

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
def spawn(
    task: str,
    context: Optional[Dict] = None,
    agent_type: str = "llm",
    working_dir: Optional[str] = None,
    **kwargs
) -> str:
    """
    Spawn a sub-agent. Returns agent ID.

    Args:
        task: Task instructions for the agent
        context: Optional context dict
        agent_type: "llm" (default), "claude_code", or "codex"
        working_dir: Working directory for CLI agents
        **kwargs: Additional args (system_prompt, max_tokens, temperature, personality, model)

    Returns:
        Agent ID string
    """
    return get_manager().spawn(
        task,
        context=context,
        agent_type=agent_type,
        working_dir=working_dir,
        **kwargs
    )


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


def is_claude_code_available() -> bool:
    """Check if Claude Code CLI is available."""
    cmd = os.getenv("CLAUDE_CLI_CMD", "claude")
    return shutil.which(cmd) is not None


def is_codex_available() -> bool:
    """Check if Codex CLI is available."""
    cmd = os.getenv("CODEX_CLI_CMD", "codex")
    return shutil.which(cmd) is not None


def get_available_agent_types() -> List[str]:
    """Get list of available agent types."""
    types = ["llm"]  # Always available
    if is_claude_code_available():
        types.append("claude_code")
    if is_codex_available():
        types.append("codex")
    return types


def get_health_status() -> dict:
    """Get SubAgentGate health status."""
    manager = get_manager()
    running = [a for a in manager.agents.values() if a.status == AgentStatus.RUNNING]
    completed = [a for a in manager.agents.values() if a.status == AgentStatus.COMPLETED]
    failed = [a for a in manager.agents.values() if a.status == AgentStatus.FAILED]

    return {
        "healthy": True,
        "agent_types": {
            "llm": True,
            "claude_code": is_claude_code_available(),
            "codex": is_codex_available(),
        },
        "agents": {
            "running": len(running),
            "completed": len(completed),
            "failed": len(failed),
            "total": len(manager.agents),
        },
    }


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
    # Health/availability
    "is_claude_code_available",
    "is_codex_available",
    "get_available_agent_types",
    "get_health_status",
    # Models
    "SubAgent",
    "AgentStatus",
    "AgentType",
]
