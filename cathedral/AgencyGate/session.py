"""Persistent shell session management.

Ported from Arbitrium — maintains shell state (env vars, cwd, aliases)
across tool calls. Uses pipes + sentinel pattern for reliable output capture.
"""

import asyncio
import logging
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("AgencyGate")

# Sentinel used to detect command completion
_SENTINEL_PREFIX = "__AGENCY_DONE_"

# Allowed shell binaries (basename, case-insensitive)
_ALLOWED_SHELLS = {
    "bash", "bash.exe",
    "sh", "sh.exe",
    "zsh", "zsh.exe",
    "fish", "fish.exe",
    "cmd.exe",
    "powershell", "powershell.exe",
    "pwsh", "pwsh.exe",
}


def validate_shell(shell: str) -> str:
    """Validate shell path against whitelist. Returns resolved path or raises ValueError."""
    from pathlib import PurePath
    basename = PurePath(shell).name.lower()
    if basename not in _ALLOWED_SHELLS:
        raise ValueError(
            f"Shell '{shell}' not in allowed list. "
            f"Allowed: {', '.join(sorted(_ALLOWED_SHELLS))}"
        )
    # Ensure the binary actually exists
    resolved = shutil.which(shell)
    if resolved is None and not os.path.isfile(shell):
        raise ValueError(f"Shell binary not found: {shell}")
    return resolved or shell


def validate_cwd(cwd: str) -> str:
    """Validate working directory exists and is a directory."""
    p = Path(cwd).resolve()
    if not p.exists():
        raise ValueError(f"Working directory does not exist: {cwd}")
    if not p.is_dir():
        raise ValueError(f"Path is not a directory: {cwd}")
    return str(p)


def detect_shell() -> str:
    """Auto-detect the best available shell for the current platform.

    On Windows, probes in order: Git Bash > PowerShell 7+ > PowerShell 5 > cmd.exe
    On Unix, uses $SHELL or falls back to /bin/sh.
    """
    if sys.platform != "win32":
        return os.environ.get("SHELL", "/bin/sh")

    program_dirs = [
        os.environ.get("PROGRAMFILES", r"C:\Program Files"),
        os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
    ]
    for pdir in program_dirs:
        for subpath in [r"Git\bin\bash.exe", r"Git\usr\bin\bash.exe"]:
            candidate = os.path.join(pdir, subpath)
            if os.path.isfile(candidate):
                _log.info(f"Detected shell: {candidate}")
                return candidate

    for name in ["bash.exe", "bash"]:
        bash_path = shutil.which(name)
        if bash_path:
            _log.info(f"Detected shell: {bash_path}")
            return bash_path

    pwsh_path = shutil.which("pwsh")
    if pwsh_path:
        _log.info(f"Detected shell: {pwsh_path}")
        return pwsh_path

    ps_path = shutil.which("powershell")
    if ps_path:
        _log.info(f"Detected shell: {ps_path}")
        return ps_path

    _log.info("Detected shell: cmd.exe (fallback)")
    return "cmd.exe"


class ShellSession:
    """A persistent shell subprocess with stdin/stdout pipes."""

    def __init__(
        self,
        session_id: str,
        shell: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        log_dir: Path | None = None,
    ):
        self.session_id = session_id
        raw_shell = shell or detect_shell()
        self.shell = validate_shell(raw_shell)
        self.cwd = validate_cwd(cwd) if cwd else os.getcwd()
        self.env = env
        self.process: asyncio.subprocess.Process | None = None
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.command_count: int = 0
        self.last_command: str | None = None
        self._log_file = None
        self._log_dir = log_dir or Path("data/shell_history")
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Spawn the shell subprocess."""
        shell_env = os.environ.copy()
        if self.env:
            shell_env.update(self.env)
        shell_env.pop("CLAUDECODE", None)

        try:
            self.process = await asyncio.create_subprocess_exec(
                self.shell,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.cwd,
                env=shell_env,
            )
        except Exception:
            # Clean up any partial state before re-raising
            self.process = None
            raise

        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = self._log_dir / f"{self.session_id}_{timestamp}.log"
        self._log_file = open(log_path, "a", encoding="utf-8")
        self._write_log(f"Session started: shell={self.shell}, cwd={self.cwd}")

        _log.info(f"Shell session '{self.session_id}' started (PID {self.process.pid})")

    @staticmethod
    def _fix_powershell_quoting(command: str) -> str:
        """Auto-fix PowerShell commands to use single quotes around -Command."""
        pattern = r'((?:powershell(?:\.exe)?|pwsh(?:\.exe)?)\s+(?:-\w+\s+)*-[Cc]ommand\s+)"((?:[^"\\]|\\.)*\$(?:[^"\\]|\\.)*)"'

        def replace_quotes(m):
            prefix = m.group(1)
            inner = m.group(2)
            inner = inner.replace("'", "'\\''")
            return f"{prefix}'{inner}'"

        return re.sub(pattern, replace_quotes, command)

    async def execute(self, command: str, timeout_ms: int = 30000) -> dict[str, Any]:
        """Execute a command and return its full output."""
        if not self.process or self.process.returncode is not None:
            return {"status": "error", "error": "Shell session is not running"}

        command = self._fix_powershell_quoting(command)

        async with self._lock:
            sentinel = f"{_SENTINEL_PREFIX}{uuid.uuid4().hex[:8]}"
            cmd_line = f": _\n{command}\necho $?:{sentinel}\n: _\n"
            self.process.stdin.write(cmd_line.encode())
            await self.process.stdin.drain()

            self._write_log(f"$ {command}")

            output_lines: list[str] = []
            exit_code: int | None = None
            timeout_sec = timeout_ms / 1000.0

            try:
                while True:
                    line_bytes = await asyncio.wait_for(
                        self.process.stdout.readline(),
                        timeout=timeout_sec,
                    )
                    if not line_bytes:
                        break

                    line = line_bytes.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")

                    if sentinel in line:
                        prefix = line.split(sentinel)[0].rstrip(":")
                        try:
                            exit_code = int(prefix)
                        except ValueError:
                            exit_code = None
                        break

                    output_lines.append(line)

            except asyncio.TimeoutError:
                output = "\n".join(output_lines)
                self._write_log(f"[TIMEOUT after {timeout_ms}ms]\n{output}")
                self.command_count += 1
                self.last_command = command
                return {
                    "status": "timeout",
                    "output": output,
                    "timeout_ms": timeout_ms,
                    "command": command,
                }

            output = "\n".join(output_lines)
            self._write_log(f"{output}\n[exit: {exit_code}]")

            self.command_count += 1
            self.last_command = command

            return {
                "status": "ok",
                "output": output,
                "exit_code": exit_code,
                "command": command,
            }

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def close(self) -> None:
        """Terminate the shell session."""
        if self.process and self.process.returncode is None:
            self.process.stdin.write(b"exit\n")
            await self.process.stdin.drain()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

        self._write_log("Session closed")
        if self._log_file:
            self._log_file.close()
            self._log_file = None

        _log.info(f"Shell session '{self.session_id}' closed")

    def _write_log(self, message: str) -> None:
        """Write to the session log file."""
        if self._log_file:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
            self._log_file.write(f"[{ts}] {message}\n")
            self._log_file.flush()

    def info(self) -> dict[str, Any]:
        """Return session metadata."""
        return {
            "session_id": self.session_id,
            "shell": self.shell,
            "cwd": self.cwd,
            "alive": self.alive,
            "pid": self.process.pid if self.process else None,
            "command_count": self.command_count,
            "last_command": self.last_command,
            "created_at": self.created_at,
        }
