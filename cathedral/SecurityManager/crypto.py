"""
SecurityManager Cryptographic Core.

Provides AES-256-GCM encryption and Argon2id key derivation.
"""

import os
import secrets
import base64
from typing import Tuple, Optional

# Cryptographic dependencies
try:
    from argon2.low_level import hash_secret_raw, Type
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# Constants
KEY_SIZE = 32  # 256 bits
NONCE_SIZE = 12  # 96 bits for GCM
SALT_SIZE = 32  # 256 bits

# Argon2id parameters (OWASP recommendations)
ARGON2_TIME_COST = 3
ARGON2_MEMORY_COST = 65536  # 64 MB
ARGON2_PARALLELISM = 4


class CryptoError(Exception):
    """Base exception for cryptographic errors."""
    pass


class CryptoNotAvailable(CryptoError):
    """Raised when crypto libraries are not installed."""
    pass


class DecryptionError(CryptoError):
    """Raised when decryption fails (wrong key or corrupted data)."""
    pass


def check_crypto_available():
    """Verify cryptographic libraries are available."""
    if not ARGON2_AVAILABLE:
        raise CryptoNotAvailable(
            "argon2-cffi not installed. Run: pip install argon2-cffi"
        )
    if not CRYPTO_AVAILABLE:
        raise CryptoNotAvailable(
            "cryptography not installed. Run: pip install cryptography"
        )


def generate_salt() -> bytes:
    """Generate a cryptographically secure random salt."""
    return secrets.token_bytes(SALT_SIZE)


def derive_key(password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
    """
    Derive a 256-bit key from password using Argon2id.

    Args:
        password: User's master password
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (derived_key, salt)
    """
    check_crypto_available()

    if salt is None:
        salt = generate_salt()

    if isinstance(password, str):
        password = password.encode('utf-8')

    key = hash_secret_raw(
        secret=password,
        salt=salt,
        time_cost=ARGON2_TIME_COST,
        memory_cost=ARGON2_MEMORY_COST,
        parallelism=ARGON2_PARALLELISM,
        hash_len=KEY_SIZE,
        type=Type.ID  # Argon2id
    )

    return key, salt


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    """
    Encrypt data using AES-256-GCM.

    Args:
        data: Plaintext bytes to encrypt
        key: 256-bit encryption key

    Returns:
        Encrypted bytes: nonce (12 bytes) + ciphertext + tag
    """
    check_crypto_available()

    if len(key) != KEY_SIZE:
        raise CryptoError(f"Key must be {KEY_SIZE} bytes, got {len(key)}")

    nonce = secrets.token_bytes(NONCE_SIZE)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, data, None)

    # Format: nonce + ciphertext (includes GCM tag)
    return nonce + ciphertext


def decrypt_bytes(encrypted: bytes, key: bytes) -> bytes:
    """
    Decrypt data encrypted with AES-256-GCM.

    Args:
        encrypted: Encrypted bytes (nonce + ciphertext)
        key: 256-bit encryption key

    Returns:
        Decrypted plaintext bytes

    Raises:
        DecryptionError: If decryption fails
    """
    check_crypto_available()

    if len(key) != KEY_SIZE:
        raise CryptoError(f"Key must be {KEY_SIZE} bytes, got {len(key)}")

    if len(encrypted) < NONCE_SIZE + 16:  # Minimum: nonce + tag
        raise DecryptionError("Encrypted data too short")

    nonce = encrypted[:NONCE_SIZE]
    ciphertext = encrypted[NONCE_SIZE:]

    try:
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext
    except Exception as e:
        raise DecryptionError(f"Decryption failed: {e}")


def encrypt_string(plaintext: str, key: bytes) -> bytes:
    """
    Encrypt a string field.

    Args:
        plaintext: String to encrypt
        key: Encryption key

    Returns:
        Encrypted bytes
    """
    return encrypt_bytes(plaintext.encode('utf-8'), key)


def decrypt_string(encrypted: bytes, key: bytes) -> str:
    """
    Decrypt a string field.

    Args:
        encrypted: Encrypted bytes
        key: Encryption key

    Returns:
        Decrypted string
    """
    plaintext_bytes = decrypt_bytes(encrypted, key)
    return plaintext_bytes.decode('utf-8')


def encrypt_to_base64(data: bytes, key: bytes) -> str:
    """Encrypt data and return as base64 string (for JSON storage)."""
    encrypted = encrypt_bytes(data, key)
    return base64.b64encode(encrypted).decode('ascii')


def decrypt_from_base64(b64_data: str, key: bytes) -> bytes:
    """Decrypt base64-encoded encrypted data."""
    encrypted = base64.b64decode(b64_data)
    return decrypt_bytes(encrypted, key)


def secure_compare(a: bytes, b: bytes) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0


def secure_wipe(data: bytearray):
    """
    Attempt to securely wipe sensitive data from memory.

    Note: Python's memory management makes this imperfect,
    but it's better than nothing.
    """
    if isinstance(data, bytearray):
        for i in range(len(data)):
            data[i] = 0
