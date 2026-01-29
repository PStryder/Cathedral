from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import RedirectResponse, JSONResponse


class SecurityMiddleware(BaseHTTPMiddleware):
    """Check if session is locked and redirect/block as needed."""

    # Routes that don't require unlock
    PUBLIC_PATHS = {
        "/lock",
        "/security/setup",
        "/api/security/status",
        "/api/security/unlock",
        "/api/security/setup",
        "/api/security/reset",
        "/static",
        "/favicon.ico",
    }

    def __init__(self, app, security_manager):
        super().__init__(app)
        self._security = security_manager

    async def dispatch(self, request, call_next):
        path = request.url.path

        # Allow public paths
        for public_path in self.PUBLIC_PATHS:
            if path.startswith(public_path):
                return await call_next(request)

        # Check if encryption is enabled and session is locked
        if self._security.is_encryption_enabled() and self._security.is_locked():
            # API requests get 401
            if path.startswith("/api/"):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Session is locked", "locked": True}
                )
            # HTML requests get redirected to lock screen
            return RedirectResponse(url=f"/lock?redirect={path}")

        # Extend session on activity
        if not self._security.is_locked():
            self._security.extend_session()

        return await call_next(request)


__all__ = ["SecurityMiddleware"]
