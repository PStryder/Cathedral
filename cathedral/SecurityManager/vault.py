"""
API Key Vault - Secure storage for API keys and secrets.

When encryption is enabled, API keys are stored encrypted.
When disabled, they're stored in plaintext.
"""

import os
import json
import base64
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from .session import get_session, SecurityLockError
from .config import get_config


# Storage paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEYS_DIR = PROJECT_ROOT / "data" / "keys"
KEYS_DIR.mkdir(parents=True, exist_ok=True)

VAULT_FILE = KEYS_DIR / "vault.json"


class APIKeyVault:
    """
    Secure storage for API keys.

    Keys are stored in a JSON vault file, encrypted when encryption is enabled.
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._loaded = False

    def _get_vault_data(self) -> dict:
        """Load vault data from file."""
        if not VAULT_FILE.exists():
            return {"keys": {}, "metadata": {"created_at": datetime.utcnow().isoformat()}}

        try:
            with open(VAULT_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"keys": {}, "metadata": {"created_at": datetime.utcnow().isoformat()}}

    def _save_vault_data(self, data: dict):
        """Save vault data to file."""
        with open(VAULT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def store_key(self, service: str, api_key: str, description: str = None):
        """
        Store an API key.

        Args:
            service: Service identifier (e.g., "openrouter", "openai")
            api_key: The API key value
            description: Optional description
        """
        config = get_config()
        vault = self._get_vault_data()

        if config.should_encrypt("api_keys"):
            # Encrypt the key
            session = get_session()
            if session.is_locked():
                raise SecurityLockError("Unlock to store API keys")

            from .crypto import encrypt_string
            key = session.get_master_key()
            encrypted = encrypt_string(api_key, key)
            stored_value = base64.b64encode(encrypted).decode('ascii')
            is_encrypted = True
        else:
            stored_value = api_key
            is_encrypted = False

        vault["keys"][service] = {
            "value": stored_value,
            "encrypted": is_encrypted,
            "description": description,
            "updated_at": datetime.utcnow().isoformat()
        }

        self._save_vault_data(vault)

        # Update cache
        self._cache[service] = api_key

    def get_key(self, service: str) -> Optional[str]:
        """
        Retrieve an API key.

        Args:
            service: Service identifier

        Returns:
            The API key or None if not found
        """
        # Check cache first
        if service in self._cache:
            return self._cache[service]

        vault = self._get_vault_data()
        key_data = vault.get("keys", {}).get(service)

        if not key_data:
            return None

        stored_value = key_data.get("value")
        is_encrypted = key_data.get("encrypted", False)

        if is_encrypted:
            session = get_session()
            if session.is_locked():
                raise SecurityLockError("Unlock to access API keys")

            from .crypto import decrypt_string
            key = session.get_master_key()
            encrypted = base64.b64decode(stored_value)
            api_key = decrypt_string(encrypted, key)
        else:
            api_key = stored_value

        # Cache the result
        self._cache[service] = api_key

        return api_key

    def delete_key(self, service: str) -> bool:
        """
        Delete an API key.

        Args:
            service: Service identifier

        Returns:
            True if deleted, False if not found
        """
        vault = self._get_vault_data()

        if service not in vault.get("keys", {}):
            return False

        del vault["keys"][service]
        self._save_vault_data(vault)

        # Clear from cache
        self._cache.pop(service, None)

        return True

    def list_keys(self) -> List[dict]:
        """
        List all stored API keys (without values).

        Returns:
            List of key metadata
        """
        vault = self._get_vault_data()
        result = []

        for service, data in vault.get("keys", {}).items():
            result.append({
                "service": service,
                "description": data.get("description"),
                "encrypted": data.get("encrypted", False),
                "updated_at": data.get("updated_at"),
                "has_value": bool(data.get("value"))
            })

        return sorted(result, key=lambda x: x["service"])

    def clear_cache(self):
        """Clear the in-memory cache (call on lock)."""
        self._cache.clear()

    def re_encrypt_all(self, old_key: bytes, new_key: bytes):
        """
        Re-encrypt all API keys with a new master key.

        Called during password change.
        """
        from .crypto import decrypt_string, encrypt_string

        vault = self._get_vault_data()

        for service, data in vault.get("keys", {}).items():
            if data.get("encrypted"):
                # Decrypt with old key
                encrypted = base64.b64decode(data["value"])
                plaintext = decrypt_string(encrypted, old_key)

                # Re-encrypt with new key
                new_encrypted = encrypt_string(plaintext, new_key)
                data["value"] = base64.b64encode(new_encrypted).decode('ascii')
                data["updated_at"] = datetime.utcnow().isoformat()

        self._save_vault_data(vault)

    def migrate_to_encrypted(self):
        """
        Migrate plaintext keys to encrypted format.

        Called when encryption is enabled.
        """
        session = get_session()
        if session.is_locked():
            raise SecurityLockError("Unlock to migrate keys")

        from .crypto import encrypt_string
        key = session.get_master_key()

        vault = self._get_vault_data()

        for service, data in vault.get("keys", {}).items():
            if not data.get("encrypted"):
                # Encrypt the plaintext value
                plaintext = data["value"]
                encrypted = encrypt_string(plaintext, key)
                data["value"] = base64.b64encode(encrypted).decode('ascii')
                data["encrypted"] = True
                data["updated_at"] = datetime.utcnow().isoformat()

        self._save_vault_data(vault)

    def migrate_to_plaintext(self):
        """
        Migrate encrypted keys to plaintext format.

        Called when encryption is disabled.
        """
        session = get_session()
        if session.is_locked():
            raise SecurityLockError("Unlock to migrate keys")

        from .crypto import decrypt_string
        key = session.get_master_key()

        vault = self._get_vault_data()

        for service, data in vault.get("keys", {}).items():
            if data.get("encrypted"):
                # Decrypt the value
                encrypted = base64.b64decode(data["value"])
                plaintext = decrypt_string(encrypted, key)
                data["value"] = plaintext
                data["encrypted"] = False
                data["updated_at"] = datetime.utcnow().isoformat()

        self._save_vault_data(vault)


# Global vault instance
_vault: Optional[APIKeyVault] = None


def get_vault() -> APIKeyVault:
    """Get or create the global API key vault."""
    global _vault
    if _vault is None:
        _vault = APIKeyVault()
    return _vault


# Convenience functions
def store_api_key(service: str, api_key: str, description: str = None):
    """Store an API key."""
    get_vault().store_key(service, api_key, description)


def get_api_key(service: str) -> Optional[str]:
    """Get an API key."""
    return get_vault().get_key(service)


def delete_api_key(service: str) -> bool:
    """Delete an API key."""
    return get_vault().delete_key(service)


def list_api_keys() -> List[dict]:
    """List all API keys."""
    return get_vault().list_keys()
