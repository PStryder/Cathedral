from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


class FolderCreate(BaseModel):
    """Model for creating a folder configuration."""
    folder_id: str
    path: str
    name: str
    permission: str = "read_only"
    backup_policy: str = "on_modify"
    max_file_size_mb: int = 10
    allowed_extensions: list | None = None
    blocked_extensions: list | None = None
    recursive: bool = True


class FolderUpdate(BaseModel):
    """Model for updating a folder configuration."""
    name: str | None = None
    permission: str | None = None
    backup_policy: str | None = None
    max_file_size_mb: int | None = None
    allowed_extensions: list | None = None
    blocked_extensions: list | None = None
    recursive: bool | None = None


class FileWriteRequest(BaseModel):
    """Model for writing a file."""
    content: str
    encoding: str = "utf-8"
    create_dirs: bool = False


def create_router(templates, FileSystemGate, emit_event) -> APIRouter:
    router = APIRouter()

    @router.get("/files", response_class=HTMLResponse)
    async def files_page(request: Request):
        """Serve the file management UI."""
        return templates.TemplateResponse("files.html", {"request": request})

    @router.get("/api/files/folders")
    async def api_list_folders():
        """List all configured folders."""
        return {"folders": FileSystemGate.list_folders()}

    @router.post("/api/files/folders")
    async def api_add_folder(data: FolderCreate):
        """Add a new folder configuration."""
        kwargs = {}
        if data.allowed_extensions is not None:
            kwargs["allowed_extensions"] = data.allowed_extensions
        if data.blocked_extensions is not None:
            kwargs["blocked_extensions"] = data.blocked_extensions

        success, message = FileSystemGate.add_folder(
            folder_id=data.folder_id,
            path=data.path,
            name=data.name,
            permission=data.permission,
            backup_policy=data.backup_policy,
            max_file_size_mb=data.max_file_size_mb,
            recursive=data.recursive,
            **kwargs
        )

        if not success:
            raise HTTPException(status_code=400, detail=message)

        await emit_event("filesystem", message, operation="add_folder", folder_id=data.folder_id)
        return {"status": "created", "message": message}

    @router.get("/api/files/folders/{folder_id}")
    async def api_get_folder(folder_id: str):
        """Get a folder configuration."""
        folder = FileSystemGate.get_folder(folder_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        return folder.to_dict()

    @router.put("/api/files/folders/{folder_id}")
    async def api_update_folder(folder_id: str, data: FolderUpdate):
        """Update a folder configuration."""
        updates = {k: v for k, v in data.model_dump().items() if v is not None}
        success, message = FileSystemGate.update_folder(folder_id, **updates)

        if not success:
            raise HTTPException(status_code=400, detail=message)

        await emit_event("filesystem", message, operation="update_folder", folder_id=folder_id)
        return {"status": "updated", "message": message}

    @router.delete("/api/files/folders/{folder_id}")
    async def api_remove_folder(folder_id: str):
        """Remove a folder configuration."""
        success, message = FileSystemGate.remove_folder(folder_id)

        if not success:
            raise HTTPException(status_code=404, detail=message)

        await emit_event("filesystem", message, operation="remove_folder", folder_id=folder_id)
        return {"status": "deleted", "message": message}

    @router.get("/api/files/list")
    async def api_list_files(folder_id: str, path: str = "", show_hidden: bool = False):
        """List directory contents."""
        result = FileSystemGate.list_dir(folder_id, path, show_hidden)
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        return {"files": result.data, "path": result.path}

    @router.get("/api/files/read")
    async def api_read_file(folder_id: str, path: str, encoding: str = "utf-8"):
        """Read a file's contents."""
        result = FileSystemGate.read_file(folder_id, path, encoding)
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        return {"content": result.data, "path": result.path, "message": result.message}

    @router.post("/api/files/write")
    async def api_write_file(folder_id: str, path: str, data: FileWriteRequest):
        """Write content to a file."""
        result = FileSystemGate.write_file(
            folder_id, path, data.content, data.encoding, data.create_dirs
        )
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)

        await emit_event("filesystem", result.message, operation="write", path=result.path)
        return result.to_dict()

    @router.post("/api/files/mkdir")
    async def api_mkdir(folder_id: str, path: str, parents: bool = False):
        """Create a directory."""
        result = FileSystemGate.mkdir(folder_id, path, parents)
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)

        await emit_event("filesystem", result.message, operation="mkdir", path=result.path)
        return result.to_dict()

    @router.delete("/api/files/delete")
    async def api_delete_file(folder_id: str, path: str, recursive: bool = False):
        """Delete a file or directory."""
        result = FileSystemGate.delete(folder_id, path, recursive)
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)

        await emit_event("filesystem", result.message, operation="delete", path=result.path)
        return result.to_dict()

    @router.get("/api/files/info")
    async def api_file_info(folder_id: str, path: str):
        """Get file or directory information."""
        result = FileSystemGate.info(folder_id, path)
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        return result.data

    @router.get("/api/files/backups")
    async def api_list_backups(folder_id: str | None = None, limit: int = 50):
        """List backups."""
        return {"backups": FileSystemGate.list_backups(folder_id, limit)}

    @router.post("/api/files/backups/{backup_id}/restore")
    async def api_restore_backup(backup_id: str, overwrite: bool = True):
        """Restore a file from backup."""
        success, message = FileSystemGate.restore_backup(backup_id, overwrite)
        if not success:
            raise HTTPException(status_code=400, detail=message)

        await emit_event("filesystem", message, operation="restore", backup_id=backup_id)
        return {"status": "restored", "message": message}

    @router.delete("/api/files/backups/{backup_id}")
    async def api_delete_backup(backup_id: str):
        """Delete a backup."""
        success, message = FileSystemGate.delete_backup(backup_id)
        if not success:
            raise HTTPException(status_code=404, detail=message)
        return {"status": "deleted", "message": message}

    @router.get("/api/files/backups/stats")
    async def api_backup_stats():
        """Get backup statistics."""
        return FileSystemGate.get_backup_stats()

    @router.get("/api/files/config")
    async def api_files_config():
        """Get FileSystemGate configuration."""
        return FileSystemGate.get_config()

    return router


__all__ = ["create_router"]
