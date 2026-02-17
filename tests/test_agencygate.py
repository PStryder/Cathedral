"""Tests for AgencyGate persistent shell sessions."""

import asyncio
import pytest
from unittest.mock import patch

from cathedral.AgencyGate import (
    initialize,
    is_initialized,
    spawn,
    exec,
    list_sessions,
    close,
    close_all,
    get_health_status,
    get_info,
    MAX_SESSIONS,
)
from cathedral.AgencyGate.session import (
    detect_shell,
    ShellSession,
    validate_shell,
    validate_cwd,
)


class TestAgencyGateInitialization:
    """Tests for AgencyGate lifecycle."""

    def test_initialize(self):
        result = initialize()
        assert result is True
        assert is_initialized() is True

    def test_health_status(self):
        initialize()
        status = get_health_status()
        assert status["gate"] == "AgencyGate"
        assert status["initialized"] is True
        assert "active_sessions" in status["details"]

    def test_get_info(self):
        info = get_info()
        assert info["gate"] == "AgencyGate"
        assert "spawn" in info["tools"]
        assert "exec" in info["tools"]


class TestShellDetection:
    """Tests for shell auto-detection."""

    def test_detect_shell_returns_string(self):
        shell = detect_shell()
        assert isinstance(shell, str)
        assert len(shell) > 0


class TestSessionLifecycle:
    """Tests for session spawn/exec/close."""

    @pytest.mark.asyncio
    async def test_spawn_session(self):
        initialize()
        info = await spawn("test-session-1")
        assert info["session_id"] == "test-session-1"
        assert info["alive"] is True
        assert info["pid"] is not None

        # Cleanup
        await close("test-session-1")

    @pytest.mark.asyncio
    async def test_spawn_auto_id(self):
        initialize()
        info = await spawn()
        assert info["session_id"].startswith("ses_")
        assert info["alive"] is True

        await close(info["session_id"])

    @pytest.mark.asyncio
    async def test_exec_command(self):
        initialize()
        await spawn("test-exec")

        result = await exec("test-exec", "echo hello")
        assert result["status"] == "ok"
        assert "hello" in result["output"]

        await close("test-exec")

    @pytest.mark.asyncio
    async def test_exec_state_persists(self):
        initialize()
        await spawn("test-persist")

        await exec("test-persist", "export MY_TEST_VAR=cathedral42")
        result = await exec("test-persist", "echo $MY_TEST_VAR")
        assert result["status"] == "ok"
        assert "cathedral42" in result["output"]

        await close("test-persist")

    @pytest.mark.asyncio
    async def test_exec_nonexistent_session(self):
        initialize()
        result = await exec("nonexistent", "echo test")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        initialize()
        await spawn("test-list-1")
        await spawn("test-list-2")

        sessions = list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert "test-list-1" in ids
        assert "test-list-2" in ids

        await close("test-list-1")
        await close("test-list-2")

    @pytest.mark.asyncio
    async def test_close_session(self):
        initialize()
        await spawn("test-close")

        result = await close("test-close")
        assert result["status"] == "closed"

    @pytest.mark.asyncio
    async def test_close_nonexistent(self):
        initialize()
        result = await close("nonexistent")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_close_all(self):
        initialize()
        await spawn("test-all-1")
        await spawn("test-all-2")

        count = await close_all()
        assert count >= 2
        assert len(list_sessions()) == 0


class TestShellValidation:
    """Tests for shell whitelist validation."""

    def test_validate_shell_rejects_arbitrary_binary(self):
        with pytest.raises(ValueError, match="not in allowed list"):
            validate_shell("python")

    def test_validate_shell_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="not in allowed list"):
            validate_shell("/tmp/evil_script.sh")

    def test_validate_shell_accepts_known_shell(self):
        # detect_shell returns a known shell on any platform
        detected = detect_shell()
        result = validate_shell(detected)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_validate_cwd_rejects_nonexistent(self):
        with pytest.raises(ValueError, match="does not exist"):
            validate_cwd("/nonexistent/path/that/should/not/exist")

    def test_validate_cwd_accepts_real_dir(self, tmp_path):
        result = validate_cwd(str(tmp_path))
        assert result == str(tmp_path.resolve())

    def test_validate_cwd_rejects_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="not a directory"):
            validate_cwd(str(f))


class TestSpawnValidation:
    """Tests for spawn-time validation."""

    @pytest.mark.asyncio
    async def test_spawn_rejects_bad_shell(self):
        initialize()
        result = await spawn("test-bad-shell", shell="python3")
        assert result["status"] == "error"
        assert "not in allowed list" in result["error"]
        await close_all()

    @pytest.mark.asyncio
    async def test_spawn_rejects_bad_cwd(self):
        initialize()
        result = await spawn("test-bad-cwd", cwd="/nonexistent/xyz")
        assert result["status"] == "error"
        assert "does not exist" in result["error"]
        await close_all()


class TestFailClosedSecurity:
    """Tests for fail-closed security in exec()."""

    @pytest.mark.asyncio
    async def test_exec_blocks_when_shellgate_import_fails(self):
        initialize()
        await spawn("test-failclosed")

        # Patch the import mechanism so `from cathedral import ShellGate` raises
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cathedral" or name == "cathedral.ShellGate":
                raise ImportError("mocked ShellGate unavailable")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = await exec("test-failclosed", "echo pwned")

        assert result["status"] == "error"
        assert "Security validation unavailable" in result.get("error", "") or \
               "Security validation failed" in result.get("error", "")

        await close("test-failclosed")
