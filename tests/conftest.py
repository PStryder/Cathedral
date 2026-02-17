"""
Pytest configuration and fixtures for Cathedral tests.
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Check for database dependencies
try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

# Check for pytest-asyncio
try:
    import pytest_asyncio
    HAS_PYTEST_ASYNCIO = True
except ImportError:
    HAS_PYTEST_ASYNCIO = False


def pytest_collection_modifyitems(config, items):
    """Skip async tests if pytest-asyncio is not installed."""
    if HAS_PYTEST_ASYNCIO:
        return

    import asyncio
    skip_asyncio = pytest.mark.skip(
        reason="pytest-asyncio not installed - async tests require pytest-asyncio"
    )
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(skip_asyncio)


@pytest.fixture(scope="session", autouse=True)
def init_test_db(tmp_path_factory):
    """Initialize a SQLite DB for tests (if aiosqlite available)."""
    if not HAS_AIOSQLITE:
        # Skip DB initialization if aiosqlite not available
        # Tests requiring DB will be skipped by their own markers
        yield
        return

    db_path = tmp_path_factory.mktemp("db") / "cathedral_test.sqlite"
    database_url = f"sqlite+aiosqlite:///{db_path.as_posix()}"

    from cathedral.shared.db_service import init_db
    from cathedral.MemoryGate.conversation import db as conversation_db
    from cathedral.ScriptureGate import init_scripture_db

    init_db(database_url)
    conversation_db.init_conversation_db()
    init_scripture_db()

    yield


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_config_dir(temp_dir: Path) -> Path:
    """Create a temporary config directory."""
    config_dir = temp_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def temp_data_dir(temp_dir: Path) -> Path:
    """Create a temporary data directory structure."""
    data_dir = temp_dir / "data"
    (data_dir / "config").mkdir(parents=True, exist_ok=True)
    (data_dir / "backups").mkdir(parents=True, exist_ok=True)
    (data_dir / "personalities").mkdir(parents=True, exist_ok=True)
    (data_dir / "shell_history").mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def sample_folder(temp_dir: Path) -> Path:
    """Create a sample folder with test files."""
    folder = temp_dir / "sample_folder"
    folder.mkdir(parents=True, exist_ok=True)

    # Create some test files
    (folder / "readme.txt").write_text("Hello World")
    (folder / "data.json").write_text('{"key": "value"}')

    subfolder = folder / "subfolder"
    subfolder.mkdir()
    (subfolder / "nested.txt").write_text("Nested content")

    return folder


@pytest.fixture
def mock_personality_data() -> dict:
    """Sample personality data for testing."""
    return {
        "id": "test_personality",
        "name": "Test Personality",
        "description": "A test personality",
        "llm_config": {
            "model": "openai/gpt-4o",
            "temperature": 0.7,
            "max_tokens": 2000,
            "system_prompt": "You are a test assistant."
        },
        "behavior": {
            "style_tags": ["helpful", "concise"],
            "response_length": "medium"
        },
        "memory": {
            "enabled": True,
            "domains": ["general"]
        },
        "metadata": {
            "category": "test",
            "author": "test",
            "is_default": False,
            "is_builtin": False,
            "usage_count": 0
        }
    }


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state between tests."""
    yield
    # Reset any module state after each test
    # This is important for modules that use global state

    # Reset ShellGate
    try:
        import cathedral.ShellGate as shell_gate
        shell_gate._config = None
        shell_gate._history = []
        shell_gate._initialized = False
    except (ImportError, AttributeError):
        pass

    # Reset FileSystemGate
    try:
        import cathedral.FileSystemGate as fs_gate
        fs_gate._config = None
        fs_gate._backup_manager = None
        fs_gate._initialized = False
    except (ImportError, AttributeError):
        pass

    # Reset PersonalityGate
    try:
        from cathedral.PersonalityGate import PersonalityManager
        PersonalityManager._cache = {}
        PersonalityManager._initialized = False
    except (ImportError, AttributeError):
        pass

    # Reset BrowserGate
    try:
        import cathedral.BrowserGate as browser_gate
        browser_gate._browser = None
    except (ImportError, AttributeError):
        pass

    # Reset SubAgentGate
    try:
        import cathedral.SubAgentGate as subagent_gate
        subagent_gate._manager = None
    except (ImportError, AttributeError):
        pass

    # Reset MemoryGate
    try:
        import cathedral.MemoryGate as memory_gate
        memory_gate._initialized = False
        memory_gate._context = None
    except (ImportError, AttributeError):
        pass

    # Reset Config
    try:
        import cathedral.Config as config_module
        config_module._manager = None
    except (ImportError, AttributeError):
        pass

    # Reset AgencyGate
    try:
        import cathedral.AgencyGate as agency_gate
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            loop.create_task(agency_gate._reset())
        else:
            asyncio.run(agency_gate._reset())
    except (ImportError, AttributeError, RuntimeError):
        pass

    # Reset VolitionGate
    try:
        import cathedral.VolitionGate as volition_gate
        volition_gate._reset()
    except (ImportError, AttributeError):
        pass

    # Reset PerceptionGate
    try:
        import cathedral.PerceptionGate as perception_gate
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            loop.create_task(perception_gate._reset())
        else:
            asyncio.run(perception_gate._reset())
    except (ImportError, AttributeError, RuntimeError):
        pass


# Skip markers for tests requiring external services
requires_network = pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_TESTS", "0") == "1",
    reason="Network tests disabled"
)

requires_api_key = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="API key not configured"
)

requires_database = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="Database not configured"
)
