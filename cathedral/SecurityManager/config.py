"""
Security Configuration Manager.

Handles the security.json configuration file which stores
encryption settings and metadata (but NOT the encryption key).
"""

import os
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .crypto import generate_salt, derive_key, encrypt_bytes, decrypt_bytes, DecryptionError


# Config file path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECURITY_CONFIG_PATH = PROJECT_ROOT / "data" / "config" / "security.json"
KEY_VERIFICATION_PATH = PROJECT_ROOT / "data" / "config" / ".keycheck"

# Ensure directory exists
SECURITY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Default security configuration
DEFAULT_CONFIG = {
    "encryption_enabled": False,
    "encryption_version": "1.0",
    "created_at": None,
    "kdf": {
        "algorithm": "argon2id",
        "time_cost": 3,
        "memory_cost": 65536,
        "parallelism": 4,
        "salt": None  # Base64 encoded
    },
    "cipher": {
        "algorithm": "AES-256-GCM",
        "key_size": 256
    },
    "encrypted_components": {
        "messages": True,
        "scripture_files": True,
        "scripture_metadata": True,
        "agent_files": True,
        "api_keys": True,
        "embeddings": False,
        "personalities": False
    },
    "session": {
        "timeout_minutes": 30,
        "lock_on_idle": True,
        "require_reauth_on_resume": True
    },
    "security_tier": "basic"
}

# Security tier presets
SECURITY_TIERS = {
    "basic": {
        "encrypted_components": {
            "messages": True,
            "scripture_files": True,
            "scripture_metadata": True,
            "agent_files": True,
            "api_keys": True,
            "embeddings": False,
            "personalities": False
        },
        "session": {
            "timeout_minutes": 30,
            "lock_on_idle": True
        }
    },
    "professional": {
        "encrypted_components": {
            "messages": True,
            "scripture_files": True,
            "scripture_metadata": True,
            "agent_files": True,
            "api_keys": True,
            "embeddings": True,
            "personalities": True
        },
        "session": {
            "timeout_minutes": 15,
            "lock_on_idle": True
        }
    },
    "maximum": {
        "encrypted_components": {
            "messages": True,
            "scripture_files": True,
            "scripture_metadata": True,
            "agent_files": True,
            "api_keys": True,
            "embeddings": True,
            "personalities": True
        },
        "session": {
            "timeout_minutes": 5,
            "lock_on_idle": True
        }
    }
}


class SecurityConfig:
    """Manages security configuration."""

    def __init__(self):
        self._config: Dict[str, Any] = None
        self._loaded = False

    def _load(self):
        """Load configuration from file."""
        if SECURITY_CONFIG_PATH.exists():
            try:
                with open(SECURITY_CONFIG_PATH, 'r') as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._config = DEFAULT_CONFIG.copy()
        else:
            self._config = DEFAULT_CONFIG.copy()
        self._loaded = True

    def _save(self):
        """Save configuration to file."""
        with open(SECURITY_CONFIG_PATH, 'w') as f:
            json.dump(self._config, f, indent=2)

    def _ensure_loaded(self):
        if not self._loaded:
            self._load()

    @property
    def config(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return self._config

    def is_encryption_enabled(self) -> bool:
        """Check if encryption is enabled."""
        self._ensure_loaded()
        return self._config.get("encryption_enabled", False)

    def get_salt(self) -> Optional[bytes]:
        """Get the stored salt (decoded from base64)."""
        self._ensure_loaded()
        salt_b64 = self._config.get("kdf", {}).get("salt")
        if salt_b64:
            return base64.b64decode(salt_b64)
        return None

    def get_session_timeout(self) -> int:
        """Get session timeout in minutes."""
        self._ensure_loaded()
        return self._config.get("session", {}).get("timeout_minutes", 30)

    def get_lock_on_idle(self) -> bool:
        """Check if lock on idle is enabled."""
        self._ensure_loaded()
        return self._config.get("session", {}).get("lock_on_idle", True)

    def should_encrypt(self, component: str) -> bool:
        """Check if a specific component should be encrypted."""
        if not self.is_encryption_enabled():
            return False
        self._ensure_loaded()
        return self._config.get("encrypted_components", {}).get(component, False)

    def get_security_tier(self) -> str:
        """Get current security tier."""
        self._ensure_loaded()
        return self._config.get("security_tier", "basic")

    def enable_encryption(self, password: str, tier: str = "basic") -> bytes:
        """
        Enable encryption with a new master password.

        Args:
            password: Master password
            tier: Security tier (basic, professional, maximum)

        Returns:
            The derived master key (caller should store in session)
        """
        self._ensure_loaded()

        # Generate salt
        salt = generate_salt()

        # Derive key
        key, _ = derive_key(password, salt)

        # Update config
        self._config["encryption_enabled"] = True
        self._config["created_at"] = datetime.utcnow().isoformat()
        self._config["kdf"]["salt"] = base64.b64encode(salt).decode('ascii')
        self._config["security_tier"] = tier

        # Apply tier settings
        if tier in SECURITY_TIERS:
            tier_config = SECURITY_TIERS[tier]
            self._config["encrypted_components"] = tier_config["encrypted_components"].copy()
            self._config["session"].update(tier_config["session"])

        # Save config
        self._save()

        # Create key verification file
        self._create_key_verification(key)

        return key

    def _create_key_verification(self, key: bytes):
        """
        Create a file to verify the key is correct on unlock.

        Contains encrypted known plaintext.
        """
        verification_data = b"CATHEDRAL_KEY_VERIFICATION_v1"
        encrypted = encrypt_bytes(verification_data, key)
        with open(KEY_VERIFICATION_PATH, 'wb') as f:
            f.write(encrypted)

    def verify_key(self, key: bytes) -> bool:
        """
        Verify that a key is correct.

        Args:
            key: Key to verify

        Returns:
            True if key is correct
        """
        if not KEY_VERIFICATION_PATH.exists():
            return True  # No verification file, assume correct

        try:
            with open(KEY_VERIFICATION_PATH, 'rb') as f:
                encrypted = f.read()
            plaintext = decrypt_bytes(encrypted, key)
            return plaintext == b"CATHEDRAL_KEY_VERIFICATION_v1"
        except DecryptionError:
            return False
        except Exception:
            return False

    def change_password(self, old_password: str, new_password: str) -> bool:
        """
        Change the master password.

        This requires re-encrypting all data with the new key.
        Returns True if successful.
        """
        if not self.is_encryption_enabled():
            return False

        # Verify old password
        old_salt = self.get_salt()
        old_key, _ = derive_key(old_password, old_salt)
        if not self.verify_key(old_key):
            return False

        # Generate new salt and key
        new_salt = generate_salt()
        new_key, _ = derive_key(new_password, new_salt)

        # Update config
        self._config["kdf"]["salt"] = base64.b64encode(new_salt).decode('ascii')
        self._save()

        # Update verification file
        self._create_key_verification(new_key)

        # Note: Caller must re-encrypt all data with new key
        return True

    def disable_encryption(self, password: str) -> bool:
        """
        Disable encryption (requires password verification).

        Note: This should only be done after decrypting all data.
        """
        if not self.is_encryption_enabled():
            return True

        # Verify password
        salt = self.get_salt()
        key, _ = derive_key(password, salt)
        if not self.verify_key(key):
            return False

        # Disable
        self._config["encryption_enabled"] = False
        self._config["kdf"]["salt"] = None
        self._save()

        # Remove verification file
        if KEY_VERIFICATION_PATH.exists():
            KEY_VERIFICATION_PATH.unlink()

        return True

    def update_component_encryption(self, component: str, enabled: bool):
        """Update encryption setting for a specific component."""
        self._ensure_loaded()
        if "encrypted_components" not in self._config:
            self._config["encrypted_components"] = {}
        self._config["encrypted_components"][component] = enabled
        self._save()

    def update_session_settings(self, timeout_minutes: int = None, lock_on_idle: bool = None):
        """Update session settings."""
        self._ensure_loaded()
        if "session" not in self._config:
            self._config["session"] = {}
        if timeout_minutes is not None:
            self._config["session"]["timeout_minutes"] = timeout_minutes
        if lock_on_idle is not None:
            self._config["session"]["lock_on_idle"] = lock_on_idle
        self._save()

    def set_security_tier(self, tier: str):
        """Apply a security tier preset."""
        if tier not in SECURITY_TIERS:
            raise ValueError(f"Unknown security tier: {tier}")

        self._ensure_loaded()
        tier_config = SECURITY_TIERS[tier]
        self._config["encrypted_components"] = tier_config["encrypted_components"].copy()
        self._config["session"].update(tier_config["session"])
        self._config["security_tier"] = tier
        self._save()

    def get_full_config(self) -> Dict[str, Any]:
        """Get full configuration (for API/UI)."""
        self._ensure_loaded()
        # Return copy without sensitive data
        config = self._config.copy()
        # Don't expose raw salt
        if "kdf" in config and "salt" in config["kdf"]:
            config["kdf"] = config["kdf"].copy()
            config["kdf"]["salt"] = "[REDACTED]" if config["kdf"]["salt"] else None
        return config

    def reset(self):
        """
        Reset security configuration (DANGEROUS - loses all encrypted data).
        """
        self._config = DEFAULT_CONFIG.copy()
        self._save()

        if KEY_VERIFICATION_PATH.exists():
            KEY_VERIFICATION_PATH.unlink()


# Global config instance
_config: Optional[SecurityConfig] = None


def get_config() -> SecurityConfig:
    """Get or create the global security config."""
    global _config
    if _config is None:
        _config = SecurityConfig()
    return _config
