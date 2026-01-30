"""
Tests for SecurityManager encryption and session management.
"""

import pytest
import base64
from pathlib import Path
from unittest.mock import patch, MagicMock

# Check if crypto libraries are available
try:
    from argon2.low_level import hash_secret_raw, Type
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CRYPTO_AVAILABLE,
    reason="Crypto libraries (argon2-cffi, cryptography) not installed"
)


class TestCryptoCore:
    """Tests for cryptographic operations."""

    def test_generate_salt(self):
        """Should generate unique 32-byte salts."""
        from cathedral.SecurityManager.crypto import generate_salt, SALT_SIZE

        salt1 = generate_salt()
        salt2 = generate_salt()

        assert len(salt1) == SALT_SIZE
        assert len(salt2) == SALT_SIZE
        assert salt1 != salt2  # Should be unique

    def test_derive_key_consistency(self):
        """Same password and salt should produce same key."""
        from cathedral.SecurityManager.crypto import derive_key, generate_salt

        salt = generate_salt()
        password = "test_password_123"

        key1, _ = derive_key(password, salt)
        key2, _ = derive_key(password, salt)

        assert key1 == key2
        assert len(key1) == 32  # 256 bits

    def test_derive_key_different_passwords(self):
        """Different passwords should produce different keys."""
        from cathedral.SecurityManager.crypto import derive_key, generate_salt

        salt = generate_salt()

        key1, _ = derive_key("password1", salt)
        key2, _ = derive_key("password2", salt)

        assert key1 != key2

    def test_derive_key_different_salts(self):
        """Different salts should produce different keys."""
        from cathedral.SecurityManager.crypto import derive_key, generate_salt

        password = "same_password"
        salt1 = generate_salt()
        salt2 = generate_salt()

        key1, _ = derive_key(password, salt1)
        key2, _ = derive_key(password, salt2)

        assert key1 != key2

    def test_encrypt_decrypt_bytes(self):
        """Should encrypt and decrypt bytes correctly."""
        from cathedral.SecurityManager.crypto import (
            encrypt_bytes, decrypt_bytes, derive_key, generate_salt
        )

        salt = generate_salt()
        key, _ = derive_key("test_password", salt)
        plaintext = b"Hello, World! This is secret data."

        encrypted = encrypt_bytes(plaintext, key)
        decrypted = decrypt_bytes(encrypted, key)

        assert decrypted == plaintext
        assert encrypted != plaintext  # Should be different from plaintext

    def test_encrypt_decrypt_string(self):
        """Should encrypt and decrypt strings correctly."""
        from cathedral.SecurityManager.crypto import (
            encrypt_string, decrypt_string, derive_key, generate_salt
        )

        salt = generate_salt()
        key, _ = derive_key("test_password", salt)
        plaintext = "This is a secret message with unicode: été 日本語"

        encrypted = encrypt_string(plaintext, key)
        decrypted = decrypt_string(encrypted, key)

        assert decrypted == plaintext

    def test_decrypt_wrong_key_fails(self):
        """Decryption with wrong key should fail."""
        from cathedral.SecurityManager.crypto import (
            encrypt_bytes, decrypt_bytes, derive_key, generate_salt, DecryptionError
        )

        salt = generate_salt()
        key1, _ = derive_key("correct_password", salt)
        key2, _ = derive_key("wrong_password", salt)

        plaintext = b"Secret data"
        encrypted = encrypt_bytes(plaintext, key1)

        with pytest.raises(DecryptionError):
            decrypt_bytes(encrypted, key2)

    def test_decrypt_corrupted_data_fails(self):
        """Decryption of corrupted data should fail."""
        from cathedral.SecurityManager.crypto import (
            encrypt_bytes, decrypt_bytes, derive_key, generate_salt, DecryptionError
        )

        salt = generate_salt()
        key, _ = derive_key("password", salt)
        plaintext = b"Secret data"

        encrypted = encrypt_bytes(plaintext, key)
        # Corrupt the data
        corrupted = encrypted[:-5] + b"xxxxx"

        with pytest.raises(DecryptionError):
            decrypt_bytes(corrupted, key)

    def test_encrypt_to_base64(self):
        """Should encrypt to base64 for JSON storage."""
        from cathedral.SecurityManager.crypto import (
            encrypt_to_base64, decrypt_from_base64, derive_key, generate_salt
        )

        salt = generate_salt()
        key, _ = derive_key("password", salt)
        plaintext = b"Data for JSON storage"

        b64_encrypted = encrypt_to_base64(plaintext, key)

        # Should be valid base64
        assert isinstance(b64_encrypted, str)
        base64.b64decode(b64_encrypted)  # Should not raise

        # Should decrypt correctly
        decrypted = decrypt_from_base64(b64_encrypted, key)
        assert decrypted == plaintext

    def test_secure_compare(self):
        """Constant-time comparison should work correctly."""
        from cathedral.SecurityManager.crypto import secure_compare

        assert secure_compare(b"hello", b"hello") is True
        assert secure_compare(b"hello", b"world") is False
        assert secure_compare(b"hello", b"helloX") is False

    def test_key_size_validation(self):
        """Should reject invalid key sizes."""
        from cathedral.SecurityManager.crypto import encrypt_bytes, CryptoError

        short_key = b"too_short"
        plaintext = b"data"

        with pytest.raises(CryptoError, match="Key must be"):
            encrypt_bytes(plaintext, short_key)


class TestSessionManager:
    """Tests for session management."""

    @pytest.fixture
    def session(self):
        """Create a fresh SessionManager for each test."""
        from cathedral.SecurityManager.session import SessionManager
        return SessionManager()

    @pytest.fixture
    def configured_session(self, session):
        """Session configured for encryption."""
        from cathedral.SecurityManager.crypto import generate_salt

        salt = generate_salt()
        session.configure(
            salt=salt,
            timeout_minutes=30,
            lock_on_idle=True,
            encryption_enabled=True
        )
        return session, salt

    def test_initial_state(self, session):
        """Session should start locked with no encryption enabled."""
        assert session.is_locked() is False  # Not enabled, so not locked
        assert session.is_encryption_enabled() is False

    def test_configure_enables_encryption(self, configured_session):
        """Configure should enable encryption mode."""
        session, _ = configured_session
        assert session.is_encryption_enabled() is True
        assert session.is_locked() is True

    def test_unlock_with_password(self, configured_session):
        """Should unlock with correct password."""
        session, salt = configured_session
        password = "test_password"

        result = session.unlock(password)

        assert result is True
        assert session.is_locked() is False

    def test_get_master_key_when_unlocked(self, configured_session):
        """Should return master key when unlocked."""
        session, _ = configured_session
        session.unlock("test_password")

        key = session.get_master_key()

        assert isinstance(key, bytes)
        assert len(key) == 32

    def test_get_master_key_when_locked_fails(self, configured_session):
        """Should raise error when getting key from locked session."""
        from cathedral.SecurityManager.session import SecurityLockError

        session, _ = configured_session
        assert session.is_locked() is True

        with pytest.raises(SecurityLockError):
            session.get_master_key()

    def test_lock_clears_key(self, configured_session):
        """Lock should clear the master key from memory."""
        from cathedral.SecurityManager.session import SecurityLockError

        session, _ = configured_session
        session.unlock("test_password")
        assert session.is_locked() is False

        session.lock()

        assert session.is_locked() is True
        with pytest.raises(SecurityLockError):
            session.get_master_key()

    def test_extend_session(self, configured_session):
        """Extend session should update last activity."""
        session, _ = configured_session
        session.unlock("test_password")

        initial_activity = session._last_activity
        session.extend_session()

        assert session._last_activity >= initial_activity

    def test_callbacks_on_lock_unlock(self, configured_session):
        """Should fire callbacks on lock/unlock."""
        session, _ = configured_session
        lock_called = []
        unlock_called = []

        session.on_lock(lambda: lock_called.append(True))
        session.on_unlock(lambda: unlock_called.append(True))

        session.unlock("test_password")
        assert len(unlock_called) == 1

        session.lock()
        assert len(lock_called) == 1

    def test_get_status(self, configured_session):
        """Should return proper status dict."""
        session, _ = configured_session
        status = session.get_status()

        assert "encryption_enabled" in status
        assert "is_locked" in status
        assert "timeout_minutes" in status
        assert "lock_on_idle" in status

    def test_no_encryption_mode(self, session):
        """When encryption disabled, should never be locked."""
        session.configure(
            salt=None,
            encryption_enabled=False
        )

        assert session.is_locked() is False
        assert session.is_encryption_enabled() is False


class TestSecurityConfig:
    """Tests for security configuration."""

    @pytest.fixture
    def temp_config_path(self, tmp_path):
        """Use temporary path for config during tests."""
        config_path = tmp_path / "security.json"
        keycheck_path = tmp_path / ".keycheck"

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                yield config_path, keycheck_path

    @pytest.fixture
    def config(self, temp_config_path):
        """Create fresh SecurityConfig with temp paths."""
        from cathedral.SecurityManager.config import SecurityConfig

        config_path, keycheck_path = temp_config_path

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                cfg = SecurityConfig()
                yield cfg

    def test_default_encryption_disabled(self, config):
        """Encryption should be disabled by default."""
        assert config.is_encryption_enabled() is False

    def test_enable_encryption(self, config, temp_config_path):
        """Should enable encryption and store salt."""
        config_path, keycheck_path = temp_config_path

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                key = config.enable_encryption("test_password", tier="basic")

                assert config.is_encryption_enabled() is True
                assert config.get_salt() is not None
                assert len(key) == 32
                assert keycheck_path.exists()

    def test_verify_correct_key(self, config, temp_config_path):
        """Should verify correct key."""
        config_path, keycheck_path = temp_config_path

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                password = "test_password"
                key = config.enable_encryption(password, tier="basic")

                assert config.verify_key(key) is True

    def test_verify_wrong_key(self, config, temp_config_path):
        """Should reject wrong key."""
        from cathedral.SecurityManager.crypto import derive_key

        config_path, keycheck_path = temp_config_path

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                config.enable_encryption("correct_password", tier="basic")

                # Generate key with wrong password
                salt = config.get_salt()
                wrong_key, _ = derive_key("wrong_password", salt)

                assert config.verify_key(wrong_key) is False

    def test_security_tiers(self, config, temp_config_path):
        """Security tiers should apply correct settings."""
        config_path, keycheck_path = temp_config_path

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                config.enable_encryption("password", tier="maximum")

                assert config.get_security_tier() == "maximum"
                assert config.get_session_timeout() == 5  # Maximum tier = 5 min

    def test_should_encrypt_component(self, config, temp_config_path):
        """Should check component encryption settings."""
        config_path, keycheck_path = temp_config_path

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                config.enable_encryption("password", tier="basic")

                # Basic tier encrypts messages
                assert config.should_encrypt("messages") is True
                # Basic tier doesn't encrypt embeddings
                assert config.should_encrypt("embeddings") is False

    def test_update_component_encryption(self, config, temp_config_path):
        """Should update individual component settings."""
        config_path, keycheck_path = temp_config_path

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                config.enable_encryption("password", tier="basic")

                config.update_component_encryption("embeddings", True)
                assert config.should_encrypt("embeddings") is True

    def test_change_password(self, config, temp_config_path):
        """Should change password successfully."""
        config_path, keycheck_path = temp_config_path

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                old_password = "old_password"
                new_password = "new_password"

                config.enable_encryption(old_password, tier="basic")
                old_salt = config.get_salt()

                result = config.change_password(old_password, new_password)

                assert result is True
                # Salt should have changed
                assert config.get_salt() != old_salt

    def test_change_password_wrong_old_password(self, config, temp_config_path):
        """Should reject password change with wrong old password."""
        config_path, keycheck_path = temp_config_path

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                config.enable_encryption("correct_password", tier="basic")

                result = config.change_password("wrong_password", "new_password")

                assert result is False

    def test_disable_encryption(self, config, temp_config_path):
        """Should disable encryption with correct password."""
        config_path, keycheck_path = temp_config_path

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                password = "test_password"
                config.enable_encryption(password, tier="basic")
                assert config.is_encryption_enabled() is True

                result = config.disable_encryption(password)

                assert result is True
                assert config.is_encryption_enabled() is False

    def test_get_full_config_redacts_salt(self, config, temp_config_path):
        """Full config should redact sensitive salt."""
        config_path, keycheck_path = temp_config_path

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                config.enable_encryption("password", tier="basic")

                full_config = config.get_full_config()

                assert full_config["kdf"]["salt"] == "[REDACTED]"


class TestSecurityManagerIntegration:
    """Integration tests for SecurityManager class."""

    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset global singletons between tests."""
        import cathedral.SecurityManager.session as session_mod
        import cathedral.SecurityManager.config as config_mod

        session_mod._session = None
        config_mod._config = None
        yield
        session_mod._session = None
        config_mod._config = None

    @pytest.fixture
    def temp_paths(self, tmp_path):
        """Set up temp paths for config files."""
        config_path = tmp_path / "security.json"
        keycheck_path = tmp_path / ".keycheck"

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                yield

    def test_is_available(self, temp_paths):
        """Should report crypto availability."""
        from cathedral.SecurityManager import SecurityManager

        assert SecurityManager.is_available() is True

    def test_setup_and_unlock_flow(self, temp_paths):
        """Full setup and unlock workflow."""
        from cathedral.SecurityManager import SecurityManager

        password = "master_password_123"

        # Setup encryption
        result = SecurityManager.setup_encryption(password, tier="basic")
        assert result is True
        assert SecurityManager.is_encryption_enabled() is True

        # Should be unlocked after setup
        assert SecurityManager.is_locked() is False

        # Lock and unlock
        SecurityManager.lock()
        assert SecurityManager.is_locked() is True

        result = SecurityManager.unlock(password)
        assert result is True
        assert SecurityManager.is_locked() is False

    def test_encrypt_decrypt_field(self, temp_paths):
        """Should encrypt and decrypt string fields."""
        from cathedral.SecurityManager import SecurityManager

        password = "test_password"
        SecurityManager.setup_encryption(password, tier="basic")

        plaintext = "Sensitive data"
        encrypted = SecurityManager.encrypt_field(plaintext)
        decrypted = SecurityManager.decrypt_field(encrypted)

        assert decrypted == plaintext
        assert encrypted != plaintext.encode()

    def test_encrypt_decrypt_stream(self, temp_paths):
        """Should encrypt and decrypt binary streams."""
        from cathedral.SecurityManager import SecurityManager

        password = "test_password"
        SecurityManager.setup_encryption(password, tier="basic")

        data = b"\x00\x01\x02\x03Binary data with nulls"
        encrypted = SecurityManager.encrypt_stream(data)
        decrypted = SecurityManager.decrypt_stream(encrypted)

        assert decrypted == data

    def test_encrypt_decrypt_base64(self, temp_paths):
        """Should encrypt/decrypt with base64 encoding."""
        from cathedral.SecurityManager import SecurityManager

        password = "test_password"
        SecurityManager.setup_encryption(password, tier="basic")

        plaintext = "Data for JSON storage"
        encrypted_b64 = SecurityManager.encrypt_field_b64(plaintext)
        decrypted = SecurityManager.decrypt_field_b64(encrypted_b64)

        assert decrypted == plaintext
        # Should be valid base64
        base64.b64decode(encrypted_b64)

    def test_get_status(self, temp_paths):
        """Should return comprehensive status."""
        from cathedral.SecurityManager import SecurityManager

        SecurityManager.setup_encryption("password", tier="professional")
        status = SecurityManager.get_status()

        assert "encryption_enabled" in status
        assert "crypto_available" in status
        assert "session" in status
        assert "security_tier" in status
        assert status["security_tier"] == "professional"

    def test_should_encrypt_component(self, temp_paths):
        """Should check component encryption settings."""
        from cathedral.SecurityManager import SecurityManager

        SecurityManager.setup_encryption("password", tier="basic")

        assert SecurityManager.should_encrypt("messages") is True
        assert SecurityManager.should_encrypt("embeddings") is False

    def test_update_settings(self, temp_paths):
        """Should update security settings."""
        from cathedral.SecurityManager import SecurityManager

        SecurityManager.setup_encryption("password", tier="basic")

        SecurityManager.update_settings(
            timeout_minutes=60,
            components={"embeddings": True}
        )

        assert SecurityManager.should_encrypt("embeddings") is True

    def test_change_password(self, temp_paths):
        """Should change password successfully."""
        from cathedral.SecurityManager import SecurityManager

        old_password = "old_password"
        new_password = "new_password"

        SecurityManager.setup_encryption(old_password, tier="basic")
        SecurityManager.lock()

        result = SecurityManager.change_password(old_password, new_password)
        assert result is True

        # Old password should no longer work
        SecurityManager.lock()
        assert SecurityManager.unlock(old_password) is False

        # New password should work
        assert SecurityManager.unlock(new_password) is True


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset global singletons between tests."""
        import cathedral.SecurityManager.session as session_mod
        import cathedral.SecurityManager.config as config_mod

        session_mod._session = None
        config_mod._config = None
        yield
        session_mod._session = None
        config_mod._config = None

    @pytest.fixture
    def temp_paths(self, tmp_path):
        """Set up temp paths for config files."""
        config_path = tmp_path / "security.json"
        keycheck_path = tmp_path / ".keycheck"

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                yield

    def test_convenience_functions(self, temp_paths):
        """Module-level convenience functions should work."""
        from cathedral.SecurityManager import (
            is_encryption_enabled,
            is_locked,
            unlock,
            lock,
            encrypt_field,
            decrypt_field,
            should_encrypt,
            get_status,
            SecurityManager,
        )

        # Setup
        SecurityManager.setup_encryption("password", tier="basic")

        assert is_encryption_enabled() is True
        assert is_locked() is False

        lock()
        assert is_locked() is True

        assert unlock("password") is True
        assert is_locked() is False

        encrypted = encrypt_field("secret")
        assert decrypt_field(encrypted) == "secret"

        assert should_encrypt("messages") is True
        assert isinstance(get_status(), dict)


class TestRequireUnlockedDecorator:
    """Tests for the require_unlocked decorator."""

    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset global singletons between tests."""
        import cathedral.SecurityManager.session as session_mod
        import cathedral.SecurityManager.config as config_mod

        session_mod._session = None
        config_mod._config = None
        yield
        session_mod._session = None
        config_mod._config = None

    @pytest.fixture
    def temp_paths(self, tmp_path):
        """Set up temp paths for config files."""
        config_path = tmp_path / "security.json"
        keycheck_path = tmp_path / ".keycheck"

        with patch("cathedral.SecurityManager.config.SECURITY_CONFIG_PATH", config_path):
            with patch("cathedral.SecurityManager.config.KEY_VERIFICATION_PATH", keycheck_path):
                yield

    def test_decorator_allows_when_unlocked(self, temp_paths):
        """Decorated function should work when unlocked."""
        from cathedral.SecurityManager.session import require_unlocked, get_session
        from cathedral.SecurityManager.crypto import generate_salt

        @require_unlocked
        def protected_function():
            return "success"

        # Configure and unlock
        session = get_session()
        session.configure(salt=generate_salt(), encryption_enabled=True)
        session.unlock("password")

        result = protected_function()
        assert result == "success"

    def test_decorator_raises_when_locked(self, temp_paths):
        """Decorated function should raise when locked."""
        from cathedral.SecurityManager.session import (
            require_unlocked, get_session, SecurityLockError
        )
        from cathedral.SecurityManager.crypto import generate_salt

        @require_unlocked
        def protected_function():
            return "success"

        # Configure but don't unlock
        session = get_session()
        session.configure(salt=generate_salt(), encryption_enabled=True)

        with pytest.raises(SecurityLockError):
            protected_function()
