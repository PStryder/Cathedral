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


@pytest.fixture(scope="session", autouse=True)
def init_test_db(tmp_path_factory):
    """Initialize a SQLite DB for tests."""
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
        from cathedral.ShellGate import __init__ as shell_init
        shell_init._config = None
        shell_init._history = []
        shell_init._initialized = False
    except (ImportError, AttributeError):
        pass

    # Reset FileSystemGate
    try:
        from cathedral.FileSystemGate import __init__ as fs_init
        fs_init._config = None
        fs_init._backup_manager = None
        fs_init._initialized = False
    except (ImportError, AttributeError):
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
