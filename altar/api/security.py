from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel


class PasswordRequest(BaseModel):
    """Model for password-based requests."""
    password: str


class SecuritySetupRequest(BaseModel):
    """Model for security setup."""
    password: str
    tier: str = "basic"


class ChangePasswordRequest(BaseModel):
    """Model for password change."""
    current_password: str
    new_password: str


class SecuritySettingsUpdate(BaseModel):
    """Model for security settings update."""
    timeout_minutes: int | None = None
    lock_on_idle: bool | None = None
    tier: str | None = None
    components: Dict[str, bool] | None = None


def create_router(templates, SecurityManager) -> APIRouter:
    router = APIRouter()

    @router.get("/security", response_class=HTMLResponse)
    async def security_page(request: Request):
        """Serve the security settings page."""
        return templates.TemplateResponse("security.html", {"request": request})

    @router.get("/security/setup", response_class=HTMLResponse)
    async def security_setup_page(request: Request):
        """Serve the security setup page."""
        return templates.TemplateResponse("security_setup.html", {"request": request})

    @router.get("/lock", response_class=HTMLResponse)
    async def lock_screen_page(request: Request):
        """Serve the lock screen."""
        # If not locked or encryption disabled, redirect to main
        if not SecurityManager.is_encryption_enabled() or not SecurityManager.is_locked():
            return RedirectResponse(url="/")
        return templates.TemplateResponse("lock_screen.html", {"request": request})

    @router.get("/api/security/status")
    async def api_security_status():
        """Get security status."""
        return SecurityManager.get_status()

    @router.post("/api/security/setup")
    async def api_security_setup(data: SecuritySetupRequest):
        """Enable encryption with a new password."""
        if SecurityManager.is_encryption_enabled():
            raise HTTPException(status_code=400, detail="Encryption already enabled")

        if not SecurityManager.is_available():
            raise HTTPException(status_code=400, detail="Cryptographic libraries not available")

        if len(data.password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

        if SecurityManager.setup_encryption(data.password, data.tier):
            return {"status": "enabled"}
        else:
            raise HTTPException(status_code=500, detail="Setup failed")

    @router.post("/api/security/unlock")
    async def api_security_unlock(data: PasswordRequest):
        """Unlock the session with password."""
        if not SecurityManager.is_encryption_enabled():
            return {"status": "not_enabled"}

        if not SecurityManager.is_locked():
            return {"status": "already_unlocked"}

        if SecurityManager.unlock(data.password):
            return {"status": "unlocked"}
        else:
            raise HTTPException(status_code=401, detail="Incorrect password")

    @router.post("/api/security/lock")
    async def api_security_lock():
        """Lock the session immediately."""
        SecurityManager.lock()
        return {"status": "locked"}

    @router.post("/api/security/change-password")
    async def api_change_password(data: ChangePasswordRequest):
        """Change the master password."""
        if not SecurityManager.is_encryption_enabled():
            raise HTTPException(status_code=400, detail="Encryption not enabled")

        if SecurityManager.is_locked():
            raise HTTPException(status_code=401, detail="Session is locked")

        if len(data.new_password) < 8:
            raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

        if SecurityManager.change_password(data.current_password, data.new_password):
            return {"status": "changed"}
        else:
            raise HTTPException(status_code=401, detail="Current password is incorrect")

    @router.post("/api/security/settings")
    async def api_security_settings(data: SecuritySettingsUpdate):
        """Update security settings."""
        if not SecurityManager.is_encryption_enabled():
            raise HTTPException(status_code=400, detail="Encryption not enabled")

        if SecurityManager.is_locked():
            raise HTTPException(status_code=401, detail="Session is locked")

        SecurityManager.update_settings(
            timeout_minutes=data.timeout_minutes,
            lock_on_idle=data.lock_on_idle,
            tier=data.tier,
            components=data.components
        )

        return {"status": "updated"}

    @router.post("/api/security/disable")
    async def api_security_disable(data: PasswordRequest):
        """Disable encryption."""
        if not SecurityManager.is_encryption_enabled():
            return {"status": "already_disabled"}

        from cathedral.SecurityManager.config import get_config
        config = get_config()

        if config.disable_encryption(data.password):
            SecurityManager.lock()  # Clear session state
            return {"status": "disabled"}
        else:
            raise HTTPException(status_code=401, detail="Incorrect password")

    @router.post("/api/security/reset")
    async def api_security_reset():
        """Reset all security and encrypted data (DANGEROUS)."""
        from cathedral.SecurityManager.config import get_config
        config = get_config()
        config.reset()

        # Note: This should also clear encrypted data files
        # For now, just reset the config
        return {"status": "reset"}

    return router


__all__ = ["create_router"]
