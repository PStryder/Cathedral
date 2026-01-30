"""
SecurityManager - Encryption and password protection for Cathedral.

Provides:
- AES-256-GCM encryption for data at rest
- Argon2id key derivation from master password
- Session-based key management
- Component-level encryption controls

Usage:
    from cathedral import SecurityManager

    # Check if encryption is enabled
    if SecurityManager.is_encryption_enabled():
        # Unlock with password
        if not SecurityManager.unlock("master_password"):
            raise Exception("Wrong password")

        # Encrypt/decrypt data
        encrypted = SecurityManager.encrypt_field("sensitive data")
        decrypted = SecurityManager.decrypt_field(encrypted)

        # Lock when done
        SecurityManager.lock()
"""

import base64
from typing import Any, Dict, Optional, Tuple

from cathedral.shared.gate import GateLogger

_log = GateLogger.get("SecurityManager")

from .crypto import (
    derive_key,
    encrypt_bytes,
    decrypt_bytes,
    encrypt_string,
    decrypt_string,
    encrypt_to_base64,
    decrypt_from_base64,
    generate_salt,
    CryptoError,
    CryptoNotAvailable,
    DecryptionError,
    check_crypto_available,
)

from .session import (
    SessionManager,
    get_session,
    SecurityLockError,
    require_unlocked,
)

from .config import (
    SecurityConfig,
    get_config,
    SECURITY_TIERS,
)


class SecurityManager:
    """
    Main interface for Cathedral's encryption system.

    All methods are class methods for easy access throughout the application.
    """

    @classmethod
    def initialize(cls):
        """
        Initialize the security system.

        Call this on application startup to load configuration
        and set up the session manager.
        """
        config = get_config()
        session = get_session()

        if config.is_encryption_enabled():
            salt = config.get_salt()
            session.configure(
                salt=salt,
                timeout_minutes=config.get_session_timeout(),
                lock_on_idle=config.get_lock_on_idle(),
                encryption_enabled=True
            )
        else:
            session.configure(
                salt=None,
                encryption_enabled=False
            )

    @classmethod
    def is_encryption_enabled(cls) -> bool:
        """Check if encryption is enabled."""
        return get_config().is_encryption_enabled()

    @classmethod
    def is_locked(cls) -> bool:
        """Check if the session is locked."""
        return get_session().is_locked()

    @classmethod
    def is_available(cls) -> bool:
        """Check if cryptographic libraries are available."""
        try:
            check_crypto_available()
            return True
        except CryptoNotAvailable:
            return False

    @classmethod
    def unlock(cls, password: str) -> bool:
        """
        Unlock the session with the master password.

        Args:
            password: Master password

        Returns:
            True if unlock successful
        """
        config = get_config()
        session = get_session()

        if not config.is_encryption_enabled():
            return True

        # Derive key and verify
        salt = config.get_salt()
        if salt is None:
            return False

        key, _ = derive_key(password, salt)

        # Verify key is correct
        if not config.verify_key(key):
            return False

        # Configure session with verified key
        session.configure(
            salt=salt,
            timeout_minutes=config.get_session_timeout(),
            lock_on_idle=config.get_lock_on_idle(),
            encryption_enabled=True
        )

        # Unlock
        return session.unlock(password)

    @classmethod
    def lock(cls):
        """Lock the session immediately."""
        get_session().lock()

    @classmethod
    def extend_session(cls):
        """Extend the session timeout (call on user activity)."""
        get_session().extend_session()

    @classmethod
    def get_master_key(cls) -> bytes:
        """
        Get the master encryption key.

        Raises:
            SecurityLockError: If session is locked
        """
        return get_session().get_master_key()

    # ==================== Encryption Operations ====================

    @classmethod
    def encrypt_stream(cls, data: bytes) -> bytes:
        """
        Encrypt binary data.

        Args:
            data: Bytes to encrypt

        Returns:
            Encrypted bytes
        """
        key = cls.get_master_key()
        return encrypt_bytes(data, key)

    @classmethod
    def decrypt_stream(cls, encrypted: bytes) -> bytes:
        """
        Decrypt binary data.

        Args:
            encrypted: Encrypted bytes

        Returns:
            Decrypted bytes
        """
        key = cls.get_master_key()
        return decrypt_bytes(encrypted, key)

    @classmethod
    def encrypt_field(cls, plaintext: str) -> bytes:
        """
        Encrypt a string field (for database storage).

        Args:
            plaintext: String to encrypt

        Returns:
            Encrypted bytes
        """
        key = cls.get_master_key()
        return encrypt_string(plaintext, key)

    @classmethod
    def decrypt_field(cls, encrypted: bytes) -> str:
        """
        Decrypt a string field.

        Args:
            encrypted: Encrypted bytes

        Returns:
            Decrypted string
        """
        key = cls.get_master_key()
        return decrypt_string(encrypted, key)

    @classmethod
    def encrypt_field_b64(cls, plaintext: str) -> str:
        """
        Encrypt a string field and return as base64 (for JSON storage).

        Args:
            plaintext: String to encrypt

        Returns:
            Base64-encoded encrypted data
        """
        key = cls.get_master_key()
        encrypted = encrypt_string(plaintext, key)
        return base64.b64encode(encrypted).decode('ascii')

    @classmethod
    def decrypt_field_b64(cls, b64_encrypted: str) -> str:
        """
        Decrypt a base64-encoded encrypted field.

        Args:
            b64_encrypted: Base64-encoded encrypted data

        Returns:
            Decrypted string
        """
        key = cls.get_master_key()
        encrypted = base64.b64decode(b64_encrypted)
        return decrypt_string(encrypted, key)

    # ==================== Configuration ====================

    @classmethod
    def should_encrypt(cls, component: str) -> bool:
        """
        Check if a specific component should be encrypted.

        Args:
            component: Component name (messages, scripture_files, api_keys, etc.)

        Returns:
            True if component should be encrypted
        """
        return get_config().should_encrypt(component)

    @classmethod
    def setup_encryption(cls, password: str, tier: str = "basic") -> bool:
        """
        Enable encryption with a new password.

        Args:
            password: Master password
            tier: Security tier (basic, professional, maximum)

        Returns:
            True if setup successful
        """
        try:
            check_crypto_available()

            config = get_config()
            key = config.enable_encryption(password, tier)

            # Initialize session
            session = get_session()
            session.configure(
                salt=config.get_salt(),
                timeout_minutes=config.get_session_timeout(),
                lock_on_idle=config.get_lock_on_idle(),
                encryption_enabled=True
            )

            # Auto-unlock after setup
            session.unlock(password)

            return True

        except Exception as e:
            _log.error(f"Setup failed: {e}")
            return False

    @classmethod
    def change_password(cls, old_password: str, new_password: str) -> bool:
        """
        Change the master password.

        Note: This changes the key derivation but does NOT re-encrypt data.
        Caller must handle re-encryption separately.

        Args:
            old_password: Current password
            new_password: New password

        Returns:
            True if successful
        """
        config = get_config()
        if config.change_password(old_password, new_password):
            # Re-unlock with new password
            cls.lock()
            return cls.unlock(new_password)
        return False

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Get security status for UI/API."""
        config = get_config()
        session = get_session()

        return {
            "encryption_enabled": config.is_encryption_enabled(),
            "crypto_available": cls.is_available(),
            "session": session.get_status(),
            "security_tier": config.get_security_tier(),
            "encrypted_components": config.config.get("encrypted_components", {}),
        }

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get full security configuration."""
        return get_config().get_full_config()

    @classmethod
    def update_settings(
        cls,
        timeout_minutes: int = None,
        lock_on_idle: bool = None,
        tier: str = None,
        components: Dict[str, bool] = None
    ):
        """Update security settings."""
        config = get_config()

        if timeout_minutes is not None or lock_on_idle is not None:
            config.update_session_settings(timeout_minutes, lock_on_idle)

        if tier is not None:
            config.set_security_tier(tier)

        if components is not None:
            for component, enabled in components.items():
                config.update_component_encryption(component, enabled)

        # Update session
        session = get_session()
        if session.is_encryption_enabled():
            session.configure(
                salt=config.get_salt(),
                timeout_minutes=config.get_session_timeout(),
                lock_on_idle=config.get_lock_on_idle(),
                encryption_enabled=True
            )


# ==================== Convenience Functions ====================

def is_encryption_enabled() -> bool:
    """Check if encryption is enabled."""
    return SecurityManager.is_encryption_enabled()


def is_locked() -> bool:
    """Check if session is locked."""
    return SecurityManager.is_locked()


def unlock(password: str) -> bool:
    """Unlock with password."""
    return SecurityManager.unlock(password)


def lock():
    """Lock the session."""
    SecurityManager.lock()


def encrypt_field(plaintext: str) -> bytes:
    """Encrypt a string field."""
    return SecurityManager.encrypt_field(plaintext)


def decrypt_field(encrypted: bytes) -> str:
    """Decrypt a string field."""
    return SecurityManager.decrypt_field(encrypted)


def encrypt_stream(data: bytes) -> bytes:
    """Encrypt binary data."""
    return SecurityManager.encrypt_stream(data)


def decrypt_stream(encrypted: bytes) -> bytes:
    """Decrypt binary data."""
    return SecurityManager.decrypt_stream(encrypted)


def should_encrypt(component: str) -> bool:
    """Check if component should be encrypted."""
    return SecurityManager.should_encrypt(component)


def initialize():
    """Initialize security system."""
    SecurityManager.initialize()


def get_status() -> Dict[str, Any]:
    """Get security status."""
    return SecurityManager.get_status()
