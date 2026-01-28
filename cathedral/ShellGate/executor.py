"""
ShellGate command executor.

Provides synchronous, asynchronous, background, and streaming command execution.
"""

import os
import sys
import asyncio
import subprocess
import signal
import uuid
from datetime import datetime
from typing import Optional, Dict, AsyncGenerator, Callable
import threading

from .models import (
    CommandConfig,
    CommandExecution,
    CommandResult,
    CommandStatus,
)
from .security import validate_command, sanitize_environment, validate_working_directory


# Track running background processes
_background_processes: Dict[str, Dict] = {}
_process_lock = threading.Lock()


def execute_sync(
    command: str,
    config: CommandConfig,
    working_dir: Optional[str] = None,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None
) -> CommandResult:
    """
    Execute a command synchronously.

    Args:
        command: Command string to execute
        config: Command configuration
        working_dir: Working directory (default: config.default_working_dir or cwd)
        timeout: Timeout in seconds (default: config.default_timeout_seconds)
        env: Environment variables (will be sanitized)

    Returns:
        CommandResult
    """
    # Validate command
    is_valid, error = validate_command(command, config)
    if not is_valid:
        return CommandResult(
            success=False,
            command=command,
            exit_code=-1,
            error=error,
            status=CommandStatus.FAILED
        )

    # Set working directory
    if working_dir is None:
        working_dir = config.default_working_dir or os.getcwd()

    is_valid_dir, resolved_dir, dir_error = validate_working_directory(working_dir)
    if not is_valid_dir:
        return CommandResult(
            success=False,
            command=command,
            exit_code=-1,
            error=dir_error,
            status=CommandStatus.FAILED
        )

    # Set timeout
    if timeout is None:
        timeout = config.default_timeout_seconds
    timeout = min(timeout, config.max_timeout_seconds)

    # Sanitize environment
    exec_env = sanitize_environment(env, config.env_blocklist)

    # Execute
    start_time = datetime.utcnow()
    execution_id = str(uuid.uuid4())[:8]

    try:
        # Determine shell based on platform
        if sys.platform == "win32":
            # Windows: use cmd.exe
            result = subprocess.run(
                command,
                shell=True,
                cwd=resolved_dir,
                env=exec_env,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        else:
            # Unix: use bash
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=resolved_dir,
                env=exec_env,
                capture_output=True,
                text=True,
                timeout=timeout
            )

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        # Truncate output if needed
        stdout = result.stdout
        stderr = result.stderr
        if len(stdout) > config.max_output_bytes:
            stdout = stdout[:config.max_output_bytes] + "\n... (output truncated)"
        if len(stderr) > config.max_output_bytes:
            stderr = stderr[:config.max_output_bytes] + "\n... (output truncated)"

        return CommandResult(
            success=result.returncode == 0,
            command=command,
            exit_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            status=CommandStatus.COMPLETED,
            execution_id=execution_id
        )

    except subprocess.TimeoutExpired as e:
        return CommandResult(
            success=False,
            command=command,
            exit_code=-1,
            stdout=e.stdout.decode() if e.stdout else "",
            stderr=e.stderr.decode() if e.stderr else "",
            duration_seconds=timeout,
            status=CommandStatus.TIMEOUT,
            error=f"Command timed out after {timeout} seconds",
            execution_id=execution_id
        )

    except Exception as e:
        return CommandResult(
            success=False,
            command=command,
            exit_code=-1,
            error=str(e),
            status=CommandStatus.FAILED,
            execution_id=execution_id
        )


async def execute_async(
    command: str,
    config: CommandConfig,
    working_dir: Optional[str] = None,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None
) -> CommandResult:
    """
    Execute a command asynchronously.

    Args:
        command: Command string to execute
        config: Command configuration
        working_dir: Working directory
        timeout: Timeout in seconds
        env: Environment variables

    Returns:
        CommandResult
    """
    # Validate command
    is_valid, error = validate_command(command, config)
    if not is_valid:
        return CommandResult(
            success=False,
            command=command,
            exit_code=-1,
            error=error,
            status=CommandStatus.FAILED
        )

    # Set working directory
    if working_dir is None:
        working_dir = config.default_working_dir or os.getcwd()

    is_valid_dir, resolved_dir, dir_error = validate_working_directory(working_dir)
    if not is_valid_dir:
        return CommandResult(
            success=False,
            command=command,
            exit_code=-1,
            error=dir_error,
            status=CommandStatus.FAILED
        )

    # Set timeout
    if timeout is None:
        timeout = config.default_timeout_seconds
    timeout = min(timeout, config.max_timeout_seconds)

    # Sanitize environment
    exec_env = sanitize_environment(env, config.env_blocklist)

    # Execute
    start_time = datetime.utcnow()
    execution_id = str(uuid.uuid4())[:8]

    try:
        # Create subprocess
        if sys.platform == "win32":
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=resolved_dir,
                env=exec_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=resolved_dir,
                env=exec_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                executable="/bin/bash"
            )

        # Wait with timeout
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return CommandResult(
                success=False,
                command=command,
                exit_code=-1,
                duration_seconds=timeout,
                status=CommandStatus.TIMEOUT,
                error=f"Command timed out after {timeout} seconds",
                execution_id=execution_id
            )

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        # Truncate if needed
        if len(stdout) > config.max_output_bytes:
            stdout = stdout[:config.max_output_bytes] + "\n... (output truncated)"
        if len(stderr) > config.max_output_bytes:
            stderr = stderr[:config.max_output_bytes] + "\n... (output truncated)"

        return CommandResult(
            success=process.returncode == 0,
            command=command,
            exit_code=process.returncode or 0,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            status=CommandStatus.COMPLETED,
            execution_id=execution_id
        )

    except Exception as e:
        return CommandResult(
            success=False,
            command=command,
            exit_code=-1,
            error=str(e),
            status=CommandStatus.FAILED,
            execution_id=execution_id
        )


async def execute_streaming(
    command: str,
    config: CommandConfig,
    working_dir: Optional[str] = None,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None
) -> AsyncGenerator[str, None]:
    """
    Execute a command with streaming output.

    Args:
        command: Command string to execute
        config: Command configuration
        working_dir: Working directory
        timeout: Timeout in seconds
        env: Environment variables

    Yields:
        Output lines as they become available
    """
    # Validate command
    is_valid, error = validate_command(command, config)
    if not is_valid:
        yield f"ERROR: {error}"
        return

    # Set working directory
    if working_dir is None:
        working_dir = config.default_working_dir or os.getcwd()

    is_valid_dir, resolved_dir, dir_error = validate_working_directory(working_dir)
    if not is_valid_dir:
        yield f"ERROR: {dir_error}"
        return

    # Set timeout
    if timeout is None:
        timeout = config.default_timeout_seconds
    timeout = min(timeout, config.max_timeout_seconds)

    # Sanitize environment
    exec_env = sanitize_environment(env, config.env_blocklist)

    try:
        # Create subprocess
        if sys.platform == "win32":
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=resolved_dir,
                env=exec_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=resolved_dir,
                env=exec_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                executable="/bin/bash"
            )

        # Stream output
        start_time = datetime.utcnow()
        total_output = 0

        async def read_with_timeout():
            return await asyncio.wait_for(
                process.stdout.readline(),
                timeout=1.0
            )

        while True:
            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > timeout:
                process.kill()
                yield f"\nERROR: Command timed out after {timeout} seconds"
                break

            try:
                line = await read_with_timeout()
                if not line:
                    break

                decoded = line.decode("utf-8", errors="replace")
                total_output += len(decoded)

                if total_output > config.max_output_bytes:
                    yield "\n... (output truncated)"
                    process.kill()
                    break

                yield decoded

            except asyncio.TimeoutError:
                # Check if process is still running
                if process.returncode is not None:
                    break
                continue

        await process.wait()

        if process.returncode != 0:
            yield f"\nProcess exited with code {process.returncode}"

    except Exception as e:
        yield f"\nERROR: {e}"


def start_background(
    command: str,
    config: CommandConfig,
    working_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    on_complete: Optional[Callable[[CommandExecution], None]] = None
) -> CommandExecution:
    """
    Start a command in the background.

    Args:
        command: Command string to execute
        config: Command configuration
        working_dir: Working directory
        env: Environment variables
        on_complete: Callback when command completes

    Returns:
        CommandExecution record
    """
    execution_id = str(uuid.uuid4())[:8]

    # Validate command
    is_valid, error = validate_command(command, config)
    if not is_valid:
        return CommandExecution(
            id=execution_id,
            command=command,
            working_dir=working_dir or os.getcwd(),
            status=CommandStatus.FAILED,
            stderr=error or "Command validation failed",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )

    # Set working directory
    if working_dir is None:
        working_dir = config.default_working_dir or os.getcwd()

    is_valid_dir, resolved_dir, dir_error = validate_working_directory(working_dir)
    if not is_valid_dir:
        return CommandExecution(
            id=execution_id,
            command=command,
            working_dir=working_dir,
            status=CommandStatus.FAILED,
            stderr=dir_error or "Invalid working directory",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )

    # Check concurrent limit
    with _process_lock:
        running_count = sum(
            1 for p in _background_processes.values()
            if p.get("execution") and p["execution"].status == CommandStatus.RUNNING
        )
        if running_count >= config.max_concurrent_background:
            return CommandExecution(
                id=execution_id,
                command=command,
                working_dir=resolved_dir,
                status=CommandStatus.FAILED,
                stderr=f"Maximum concurrent background commands ({config.max_concurrent_background}) reached",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )

    # Create execution record
    execution = CommandExecution(
        id=execution_id,
        command=command,
        working_dir=resolved_dir,
        status=CommandStatus.RUNNING,
        started_at=datetime.utcnow(),
        is_background=True,
        timeout_seconds=config.max_timeout_seconds
    )

    # Sanitize environment
    exec_env = sanitize_environment(env, config.env_blocklist)

    # Start process in thread
    def run_process():
        nonlocal execution
        try:
            if sys.platform == "win32":
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=resolved_dir,
                    env=exec_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            else:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    executable="/bin/bash",
                    cwd=resolved_dir,
                    env=exec_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=os.setsid
                )

            with _process_lock:
                _background_processes[execution_id]["process"] = process

            stdout, stderr = process.communicate(timeout=config.max_timeout_seconds)

            execution.stdout = stdout[:config.max_output_bytes] if stdout else ""
            execution.stderr = stderr[:config.max_output_bytes] if stderr else ""
            execution.exit_code = process.returncode
            execution.status = CommandStatus.COMPLETED if process.returncode == 0 else CommandStatus.FAILED
            execution.completed_at = datetime.utcnow()

        except subprocess.TimeoutExpired:
            process.kill()
            execution.status = CommandStatus.TIMEOUT
            execution.completed_at = datetime.utcnow()

        except Exception as e:
            execution.status = CommandStatus.FAILED
            execution.stderr = str(e)
            execution.completed_at = datetime.utcnow()

        finally:
            with _process_lock:
                if execution_id in _background_processes:
                    _background_processes[execution_id]["execution"] = execution

            if on_complete:
                on_complete(execution)

    # Register and start
    with _process_lock:
        _background_processes[execution_id] = {
            "execution": execution,
            "process": None,
            "thread": None
        }

    thread = threading.Thread(target=run_process, daemon=True)
    with _process_lock:
        _background_processes[execution_id]["thread"] = thread
    thread.start()

    return execution


def get_background_status(execution_id: str) -> Optional[CommandExecution]:
    """Get status of a background command."""
    with _process_lock:
        info = _background_processes.get(execution_id)
        if info:
            return info.get("execution")
    return None


def cancel_background(execution_id: str) -> bool:
    """
    Cancel a background command.

    Args:
        execution_id: Execution ID to cancel

    Returns:
        True if cancelled, False if not found or already complete
    """
    with _process_lock:
        info = _background_processes.get(execution_id)
        if not info:
            return False

        execution = info.get("execution")
        process = info.get("process")

        if execution and execution.is_done:
            return False

        if process and process.poll() is None:
            try:
                if sys.platform == "win32":
                    process.terminate()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

                execution.status = CommandStatus.CANCELLED
                execution.completed_at = datetime.utcnow()
                return True
            except Exception:
                return False

    return False


def list_background() -> list[CommandExecution]:
    """List all background command executions."""
    with _process_lock:
        return [
            info["execution"]
            for info in _background_processes.values()
            if info.get("execution")
        ]


def cleanup_completed(max_age_seconds: int = 3600) -> int:
    """
    Clean up completed background commands older than max_age.

    Args:
        max_age_seconds: Remove completed commands older than this

    Returns:
        Number of cleaned up entries
    """
    now = datetime.utcnow()
    to_remove = []

    with _process_lock:
        for exec_id, info in _background_processes.items():
            execution = info.get("execution")
            if execution and execution.is_done and execution.completed_at:
                age = (now - execution.completed_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(exec_id)

        for exec_id in to_remove:
            del _background_processes[exec_id]

    return len(to_remove)
