"""
MCPClient Transport Layer.

Handles stdio-based communication with MCP servers via subprocess.
Implements JSON-RPC 2.0 protocol for MCP message exchange.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, List, Optional

from cathedral.shared.gate import GateLogger
from cathedral.MCPClient.models import JSONRPCRequest, JSONRPCResponse

_log = GateLogger.get("MCPClient.Transport")


class StdioTransport:
    """
    Stdio transport for MCP server communication.

    Spawns a subprocess and communicates via stdin/stdout using JSON-RPC 2.0.
    """

    def __init__(
        self,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize transport.

        Args:
            command: Executable command (e.g., 'npx', 'python')
            args: Command arguments
            env: Additional environment variables
        """
        self.command = command
        self.args = args
        self.env = env or {}

        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def pid(self) -> Optional[int]:
        """Get subprocess PID."""
        return self._process.pid if self._process else None

    def _resolve_windows_cmd(
        self, cmd: List[str], env: Dict[str, str]
    ) -> List[str]:
        """
        Resolve Windows .cmd files to their full paths.

        On Windows, npm/npx/node are typically .cmd batch files.
        We need to find and run them directly to avoid shell=True.

        Args:
            cmd: Original command list
            env: Environment variables

        Returns:
            Command list with resolved executable path
        """
        import shutil

        executable = cmd[0]
        args = cmd[1:]

        # Try to find the .cmd file in PATH
        # shutil.which handles Windows .cmd/.bat extensions automatically
        resolved = shutil.which(executable, path=env.get("PATH", os.environ.get("PATH", "")))

        if resolved:
            _log.debug(f"Resolved {executable} to {resolved}")
            return [resolved] + args

        # If not found, try common extensions
        for ext in [".cmd", ".bat", ".exe"]:
            resolved = shutil.which(executable + ext, path=env.get("PATH", os.environ.get("PATH", "")))
            if resolved:
                _log.debug(f"Resolved {executable}{ext} to {resolved}")
                return [resolved] + args

        # Fall back to original (will likely fail with FileNotFoundError)
        _log.warning(f"Could not resolve {executable} on Windows PATH")
        return cmd

    async def start(self) -> bool:
        """
        Start the subprocess and begin reading responses.

        Returns:
            True if subprocess started successfully
        """
        if self._process is not None:
            _log.warning("Transport already started")
            return True

        # Build environment
        process_env = os.environ.copy()
        process_env.update(self.env)

        # Build command
        cmd = [self.command] + self.args
        _log.info(f"Starting MCP server: {' '.join(cmd)}")

        try:
            # On Windows, npm/npx/node are .cmd files that need special handling
            # We avoid shell=True for security - instead find the actual executable
            actual_cmd = cmd
            if sys.platform == "win32" and self.command in ("npx", "npm", "node"):
                actual_cmd = self._resolve_windows_cmd(cmd, process_env)

            self._process = subprocess.Popen(
                actual_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=process_env,
                shell=False,  # Never use shell=True for security
                bufsize=0,  # Unbuffered
            )

            self._closed = False

            # Start async reader for stdout
            self._reader_task = asyncio.create_task(self._read_loop())

            _log.info(f"MCP server started with PID {self._process.pid}")
            return True

        except FileNotFoundError as e:
            _log.error(f"Command not found: {self.command} - {e}")
            return False
        except Exception as e:
            _log.error(f"Failed to start MCP server: {e}")
            return False

    async def stop(self) -> None:
        """Stop the subprocess and cleanup."""
        self._closed = True

        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Cancel reader task
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        # Terminate process
        if self._process:
            try:
                self._process.terminate()
                # Give it a moment to terminate gracefully
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=1)
            except Exception as e:
                _log.warning(f"Error stopping process: {e}")
            finally:
                self._process = None

        _log.info("Transport stopped")

    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> JSONRPCResponse:
        """
        Send a JSON-RPC request and wait for response.

        Args:
            method: RPC method name
            params: Method parameters
            timeout: Request timeout in seconds

        Returns:
            JSONRPCResponse with result or error
        """
        if self._process is None or self._closed:
            return JSONRPCResponse(
                error={"code": -1, "message": "Transport not connected"}
            )

        async with self._lock:
            self._request_id += 1
            request_id = self._request_id

        request = JSONRPCRequest(
            id=request_id,
            method=method,
            params=params,
        )

        # Create future for response
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_requests[request_id] = future

        try:
            # Send request
            request_json = request.model_dump_json(exclude_none=True)
            request_bytes = (request_json + "\n").encode("utf-8")

            _log.debug(f"Sending request: {request_json}")

            # Write to stdin in thread to avoid blocking
            await asyncio.to_thread(self._write_stdin, request_bytes)

            # Wait for response with timeout
            response_data = await asyncio.wait_for(future, timeout=timeout)
            return JSONRPCResponse(**response_data)

        except asyncio.TimeoutError:
            _log.warning(f"Request {request_id} timed out after {timeout}s")
            return JSONRPCResponse(
                id=request_id,
                error={"code": -2, "message": f"Request timed out after {timeout}s"},
            )
        except Exception as e:
            _log.error(f"Request {request_id} failed: {e}")
            return JSONRPCResponse(
                id=request_id,
                error={"code": -3, "message": str(e)},
            )
        finally:
            self._pending_requests.pop(request_id, None)

    def _write_stdin(self, data: bytes) -> None:
        """Write data to subprocess stdin (blocking, run in thread)."""
        if self._process and self._process.stdin:
            try:
                self._process.stdin.write(data)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                _log.warning(f"Failed to write to stdin: {e}")

    async def _read_loop(self) -> None:
        """Async loop to read responses from stdout."""
        if not self._process or not self._process.stdout:
            return

        buffer = b""

        try:
            while not self._closed and self._process.poll() is None:
                # Read in thread to avoid blocking event loop
                try:
                    chunk = await asyncio.to_thread(
                        self._read_chunk, self._process.stdout
                    )
                    if not chunk:
                        # EOF or empty read
                        await asyncio.sleep(0.01)
                        continue

                    buffer += chunk

                    # Process complete lines
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line.strip():
                            await self._handle_message(line.decode("utf-8", errors="replace"))

                except Exception as e:
                    if not self._closed:
                        _log.warning(f"Read error: {e}")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            _log.error(f"Reader loop error: {e}")

    def _read_chunk(self, stdout, size: int = 4096) -> bytes:
        """Read a chunk from stdout (blocking, run in thread)."""
        try:
            if sys.platform == "win32":
                # Windows doesn't support select on pipes
                # Use read1 if available (BufferedReader), otherwise readline for safety
                if hasattr(stdout, 'read1'):
                    return stdout.read1(size)
                else:
                    # readline is safer than read(size) which could block forever
                    line = stdout.readline()
                    return line
            else:
                import select
                ready, _, _ = select.select([stdout], [], [], 0.1)
                if ready:
                    return stdout.read(size)
                return b""
        except Exception:
            return b""

    async def _handle_message(self, message: str) -> None:
        """Handle a received JSON-RPC message."""
        try:
            data = json.loads(message)
            _log.debug(f"Received message: {message[:200]}")

            # Check if it's a response (has id)
            if "id" in data and data["id"] is not None:
                request_id = data["id"]
                if request_id in self._pending_requests:
                    future = self._pending_requests[request_id]
                    if not future.done():
                        future.set_result(data)
                else:
                    _log.debug(f"No pending request for id {request_id}")

            # Notifications (no id) are logged but not processed
            elif "method" in data:
                _log.debug(f"Received notification: {data.get('method')}")

        except json.JSONDecodeError as e:
            _log.warning(f"Invalid JSON message: {e}")
        except Exception as e:
            _log.error(f"Error handling message: {e}")

    def is_running(self) -> bool:
        """Check if subprocess is still running."""
        return self._process is not None and self._process.poll() is None

    def get_stderr(self) -> str:
        """Get any stderr output from the process."""
        if self._process and self._process.stderr:
            try:
                # Non-blocking read of stderr
                import select
                if sys.platform != "win32":
                    ready, _, _ = select.select([self._process.stderr], [], [], 0)
                    if ready:
                        return self._process.stderr.read().decode("utf-8", errors="replace")
                return ""
            except Exception:
                return ""
        return ""


__all__ = ["StdioTransport"]
