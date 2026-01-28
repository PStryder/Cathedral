"""
SessionManager - Manages authentication state and session lifecycle.

Handles:
- Master key storage in memory
- Lock/unlock operations
- Session timeout management
- Auto-lock on idle
"""

import time
import threading
from typing import Optional, Callable
from datetime import datetime, timedelta


class SecurityLockError(Exception):
    """Raised when operation requires unlock but session is locked."""
    pass


class SessionManager:
    """
    Manages the encryption session state.

    The master key is only held in memory while unlocked.
    On lock, the key is cleared and must be re-derived from password.
    """

    def __init__(self):
        self._master_key: Optional[bytearray] = None
        self._salt: Optional[bytes] = None
        self._unlock_time: Optional[datetime] = None
        self._last_activity: Optional[datetime] = None
        self._timeout_minutes: int = 30
        self._lock_on_idle: bool = True
        self._is_locked: bool = True
        self._encryption_enabled: bool = False

        # Callbacks for lock/unlock events
        self._on_lock_callbacks: list = []
        self._on_unlock_callbacks: list = []

        # Auto-lock timer
        self._timer_thread: Optional[threading.Thread] = None
        self._timer_running: bool = False

    def configure(
        self,
        salt: bytes,
        timeout_minutes: int = 30,
        lock_on_idle: bool = True,
        encryption_enabled: bool = True
    ):
        """Configure session parameters."""
        self._salt = salt
        self._timeout_minutes = timeout_minutes
        self._lock_on_idle = lock_on_idle
        self._encryption_enabled = encryption_enabled

    def is_encryption_enabled(self) -> bool:
        """Check if encryption is enabled."""
        return self._encryption_enabled

    def is_locked(self) -> bool:
        """Check if session is locked."""
        if not self._encryption_enabled:
            return False
        return self._is_locked

    def get_master_key(self) -> bytes:
        """
        Get the master encryption key.

        Raises:
            SecurityLockError: If session is locked
        """
        if self.is_locked():
            raise SecurityLockError("Session is locked. Unlock to access encrypted data.")

        if self._master_key is None:
            raise SecurityLockError("No master key available.")

        return bytes(self._master_key)

    def get_salt(self) -> Optional[bytes]:
        """Get the stored salt."""
        return self._salt

    def unlock(self, password: str) -> bool:
        """
        Unlock the session with the master password.

        Args:
            password: Master password

        Returns:
            True if unlock successful, False if password incorrect
        """
        if not self._encryption_enabled:
            return True

        if self._salt is None:
            raise SecurityLockError("Encryption not configured. No salt available.")

        from .crypto import derive_key, DecryptionError

        try:
            # Derive key from password
            key, _ = derive_key(password, self._salt)

            # Store key in bytearray (slightly easier to wipe)
            self._master_key = bytearray(key)
            self._is_locked = False
            self._unlock_time = datetime.utcnow()
            self._last_activity = datetime.utcnow()

            # Start auto-lock timer
            self._start_auto_lock_timer()

            # Fire unlock callbacks
            for callback in self._on_unlock_callbacks:
                try:
                    callback()
                except Exception:
                    pass

            return True

        except Exception as e:
            # Clear any partial state
            self._master_key = None
            self._is_locked = True
            return False

    def lock(self):
        """
        Lock the session, clearing the master key from memory.
        """
        if not self._encryption_enabled:
            return

        # Clear master key
        if self._master_key is not None:
            # Attempt to wipe
            for i in range(len(self._master_key)):
                self._master_key[i] = 0
            self._master_key = None

        self._is_locked = True
        self._unlock_time = None
        self._last_activity = None

        # Stop timer
        self._stop_auto_lock_timer()

        # Fire lock callbacks
        for callback in self._on_lock_callbacks:
            try:
                callback()
            except Exception:
                pass

    def extend_session(self):
        """Record activity to extend the session timeout."""
        if not self._is_locked:
            self._last_activity = datetime.utcnow()

    def check_timeout(self) -> bool:
        """
        Check if session has timed out.

        Returns:
            True if timed out (will auto-lock)
        """
        if self._is_locked or not self._lock_on_idle:
            return False

        if self._last_activity is None:
            return False

        elapsed = datetime.utcnow() - self._last_activity
        if elapsed > timedelta(minutes=self._timeout_minutes):
            self.lock()
            return True

        return False

    def time_until_lock(self) -> Optional[int]:
        """
        Get seconds until auto-lock.

        Returns:
            Seconds until lock, or None if not applicable
        """
        if self._is_locked or not self._lock_on_idle or self._last_activity is None:
            return None

        elapsed = datetime.utcnow() - self._last_activity
        remaining = timedelta(minutes=self._timeout_minutes) - elapsed
        return max(0, int(remaining.total_seconds()))

    def on_lock(self, callback: Callable):
        """Register a callback to be called on lock."""
        self._on_lock_callbacks.append(callback)

    def on_unlock(self, callback: Callable):
        """Register a callback to be called on unlock."""
        self._on_unlock_callbacks.append(callback)

    def _start_auto_lock_timer(self):
        """Start background thread to check for timeout."""
        if self._timer_running:
            return

        self._timer_running = True

        def timer_loop():
            while self._timer_running and not self._is_locked:
                time.sleep(30)  # Check every 30 seconds
                if self._timer_running:
                    self.check_timeout()

        self._timer_thread = threading.Thread(target=timer_loop, daemon=True)
        self._timer_thread.start()

    def _stop_auto_lock_timer(self):
        """Stop the auto-lock timer."""
        self._timer_running = False
        self._timer_thread = None

    def get_status(self) -> dict:
        """Get current session status."""
        return {
            "encryption_enabled": self._encryption_enabled,
            "is_locked": self.is_locked(),
            "timeout_minutes": self._timeout_minutes,
            "lock_on_idle": self._lock_on_idle,
            "time_until_lock": self.time_until_lock(),
            "unlock_time": self._unlock_time.isoformat() if self._unlock_time else None,
        }


# Global session instance
_session: Optional[SessionManager] = None


def get_session() -> SessionManager:
    """Get or create the global session manager."""
    global _session
    if _session is None:
        _session = SessionManager()
    return _session


def require_unlocked(func):
    """Decorator to require unlocked session."""
    def wrapper(*args, **kwargs):
        session = get_session()
        if session.is_locked():
            raise SecurityLockError("Session is locked. Unlock to continue.")
        session.extend_session()
        return func(*args, **kwargs)
    return wrapper
