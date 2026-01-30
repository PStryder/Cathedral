"""
Cathedral Configuration Manager.

Centralized configuration with:
- Schema-driven validation
- Environment variable fallback
- Persistent storage
- Web UI for editing
- Auto-discovery of missing config
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv, set_key

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("Config")

from cathedral.Config.schema import (
    CONFIG_SCHEMA,
    ConfigField,
    ConfigType,
    ConfigCategory,
    get_schema_by_key,
    get_required_fields,
    schema_to_dict,
)


# Config file paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_JSON = PROJECT_ROOT / "data" / "config.json"
CONFIG_MD = PROJECT_ROOT / "CONFIG.md"

# Ensure data directory exists
CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """
    Manages Cathedral configuration.

    Priority order:
    1. Environment variables
    2. config.json
    3. Schema defaults
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._loaded = False
        self._load()

    def _load(self):
        """Load configuration from all sources."""
        # Load .env file
        load_dotenv(ENV_FILE)

        # Load config.json if exists
        json_config = {}
        if CONFIG_JSON.exists():
            try:
                with open(CONFIG_JSON) as f:
                    json_config = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Build config from schema
        for field in CONFIG_SCHEMA:
            # Priority: env var > json config > default
            value = os.environ.get(field.env_var)

            if value is None and field.key in json_config:
                value = json_config[field.key]

            if value is None:
                value = field.default

            # Type conversion
            value = self._convert_type(value, field.config_type)

            self._cache[field.key] = value

        self._loaded = True

    def _convert_type(self, value: Any, config_type: ConfigType) -> Any:
        """Convert value to appropriate type."""
        if value is None:
            return None

        try:
            if config_type == ConfigType.INTEGER:
                return int(value)
            elif config_type == ConfigType.FLOAT:
                return float(value)
            elif config_type == ConfigType.BOOLEAN:
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ("true", "1", "yes", "on")
            elif config_type == ConfigType.LIST:
                if isinstance(value, list):
                    return value
                return [v.strip() for v in str(value).split(",") if v.strip()]
            else:
                return str(value) if value else None
        except (ValueError, TypeError):
            return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if not self._loaded:
            self._load()
        return self._cache.get(key, default)

    def set(self, key: str, value: Any, persist: bool = True) -> bool:
        """
        Set a configuration value.

        Args:
            key: Config key
            value: New value
            persist: Whether to save to config.json

        Returns:
            True if successful
        """
        field = get_schema_by_key(key)
        if not field:
            return False

        # Convert type
        value = self._convert_type(value, field.config_type)

        # Update cache
        self._cache[key] = value

        # Persist if requested
        if persist:
            self._save_json()

            # Also update .env for env vars
            if field.env_var:
                self._update_env(field.env_var, value, field.config_type)

        return True

    def _save_json(self):
        """Save current config to JSON."""
        # Only save non-sensitive, non-default values
        to_save = {}
        for field in CONFIG_SCHEMA:
            value = self._cache.get(field.key)
            # Skip secrets (they should stay in .env)
            if field.sensitive:
                continue
            # Skip if same as default
            if value == field.default:
                continue
            # Skip None values
            if value is None:
                continue
            to_save[field.key] = value

        with open(CONFIG_JSON, "w") as f:
            json.dump(to_save, f, indent=2)

    def _update_env(self, key: str, value: Any, config_type: ConfigType):
        """Update .env file."""
        # Convert value to string
        if config_type == ConfigType.BOOLEAN:
            str_value = "true" if value else "false"
        elif config_type == ConfigType.LIST:
            str_value = ",".join(value) if isinstance(value, list) else str(value)
        else:
            str_value = str(value) if value is not None else ""

        # Update .env file
        try:
            set_key(str(ENV_FILE), key, str_value)
            # Also update environment
            os.environ[key] = str_value
        except Exception as e:
            _log.warning(f"Could not update .env: {e}")

    def get_all(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Get all configuration values."""
        result = {}
        for field in CONFIG_SCHEMA:
            value = self._cache.get(field.key)
            if field.sensitive and not include_secrets:
                # Mask secrets - show last 4 chars only if value is long enough
                if value:
                    str_value = str(value)
                    if len(str_value) > 12:
                        result[field.key] = "***" + str_value[-4:]
                    else:
                        result[field.key] = "****"  # Don't expose any part of short secrets
                else:
                    result[field.key] = None
            else:
                result[field.key] = value
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get configuration status with missing/invalid checks."""
        missing = []
        invalid = []
        configured = []

        for field in CONFIG_SCHEMA:
            value = self._cache.get(field.key)

            if value is None or value == "":
                if field.required:
                    missing.append({
                        "key": field.key,
                        "description": field.description,
                        "category": field.category.value,
                    })
            else:
                # Validate if pattern provided
                if field.validation:
                    import re
                    if not re.match(field.validation, str(value)):
                        invalid.append({
                            "key": field.key,
                            "value": "***" if field.sensitive else value,
                            "pattern": field.validation,
                        })
                    else:
                        configured.append(field.key)
                else:
                    configured.append(field.key)

        return {
            "status": "ok" if not missing else "incomplete",
            "missing": missing,
            "invalid": invalid,
            "configured_count": len(configured),
            "total_count": len(CONFIG_SCHEMA),
            "required_missing": len([m for m in missing if get_schema_by_key(m["key"]).required]),
        }

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration.

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        for field in CONFIG_SCHEMA:
            value = self._cache.get(field.key)

            # Check required
            if field.required and (value is None or value == ""):
                errors.append(f"Required config missing: {field.key}")
                continue

            # Check validation pattern
            if value and field.validation:
                import re
                if not re.match(field.validation, str(value)):
                    errors.append(f"Invalid format for {field.key}")

            # Check options
            if value and field.options and value not in field.options:
                errors.append(f"Invalid option for {field.key}: {value}")

        return len(errors) == 0, errors

    def generate_config_md(self) -> str:
        """Generate CONFIG.md documentation."""
        lines = [
            "# Cathedral Configuration",
            "",
            f"*Auto-generated on {datetime.utcnow().isoformat()}*",
            "",
            "## Configuration Status",
            "",
        ]

        status = self.get_status()
        lines.append(f"- **Status**: {status['status']}")
        lines.append(f"- **Configured**: {status['configured_count']}/{status['total_count']}")
        if status['missing']:
            lines.append(f"- **Missing Required**: {status['required_missing']}")
        lines.append("")

        # Group by category
        for category in ConfigCategory:
            fields = [f for f in CONFIG_SCHEMA if f.category == category]
            if not fields:
                continue

            lines.append(f"## {category.value.replace('_', ' ').title()}")
            lines.append("")
            lines.append("| Variable | Description | Required | Default |")
            lines.append("|----------|-------------|----------|---------|")

            for field in fields:
                default = field.default if not field.sensitive else "(secret)"
                default = str(default)[:30] if default else "-"
                req = "Yes" if field.required else "No"
                lines.append(f"| `{field.key}` | {field.description} | {req} | {default} |")

            lines.append("")

        # Add quick start
        lines.extend([
            "## Quick Start",
            "",
            "1. Copy `.env.example` to `.env`",
            "2. Edit `.env` with your API keys:",
            "   ```",
            "   OPENROUTER_API_KEY=sk-or-your-key-here",
            "   OPENAI_API_KEY=sk-your-openai-key",
            "   DATABASE_URL=postgresql://user:pass@localhost:5432/cathedral",
            "   ```",
            "3. Or use the web config editor at `/config`",
            "",
            "## Environment Variables",
            "",
            "All configuration can be set via environment variables.",
            "The `.env` file is automatically loaded on startup.",
            "",
        ])

        return "\n".join(lines)

    def save_config_md(self):
        """Save CONFIG.md file."""
        content = self.generate_config_md()
        with open(CONFIG_MD, "w") as f:
            f.write(content)

    def create_env_template(self) -> str:
        """Generate .env.example template."""
        lines = [
            "# Cathedral Configuration",
            "# Copy this file to .env and fill in your values",
            "",
        ]

        current_category = None
        for field in CONFIG_SCHEMA:
            if field.category != current_category:
                current_category = field.category
                lines.append(f"# === {current_category.value.replace('_', ' ').title()} ===")
                lines.append("")

            # Comment with description
            lines.append(f"# {field.description}")
            if field.required:
                lines.append("# (REQUIRED)")
            if field.options:
                lines.append(f"# Options: {', '.join(field.options)}")

            # Value line
            default = "" if field.sensitive else (field.default or "")
            lines.append(f"{field.env_var}={default}")
            lines.append("")

        return "\n".join(lines)

    def save_env_template(self):
        """Save .env.example file."""
        content = self.create_env_template()
        with open(PROJECT_ROOT / ".env.example", "w") as f:
            f.write(content)


# Global instance
_manager: Optional[ConfigManager] = None


def get_manager() -> ConfigManager:
    """Get or create the global ConfigManager."""
    global _manager
    if _manager is None:
        _manager = ConfigManager()
    return _manager


def reload():
    """Reload configuration from files."""
    global _manager
    _manager = ConfigManager()


# Convenience functions
def get(key: str, default: Any = None) -> Any:
    """Get a config value."""
    return get_manager().get(key, default)


def set(key: str, value: Any, persist: bool = True) -> bool:
    """Set a config value."""
    return get_manager().set(key, value, persist)


def get_all(include_secrets: bool = False) -> Dict:
    """Get all config values."""
    return get_manager().get_all(include_secrets)


def get_status() -> Dict:
    """Get config status."""
    return get_manager().get_status()


def validate() -> Tuple[bool, List[str]]:
    """Validate configuration."""
    return get_manager().validate()


def get_schema() -> Dict:
    """Get schema as dict for API."""
    return schema_to_dict()
