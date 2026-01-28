# cathedral/ScriptureGate/__init__.py
from pathlib import Path
import json

# Root for all file-based ritual ops
SCRIPTURE_PATH = Path(__file__).parent.parent.parent / "data" / "scripture"
SCRIPTURE_PATH.mkdir(parents=True, exist_ok=True)

# Standard extensions for sacred forms
EXTENSIONS = {
    "thread": ".thread.json",
    "bios": ".bios.txt",
    "glyph": ".glyph.json",
}

def _resolve_path(name: str, kind: str) -> Path:
    ext = EXTENSIONS.get(kind, ".txt")
    return SCRIPTURE_PATH / f"{name}{ext}"

def save_text(name: str, content: str, kind: str = "bios"):
    path = _resolve_path(name, kind)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def load_text(name: str, kind: str = "bios") -> str:
    path = _resolve_path(name, kind)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_json(name: str, data: dict, kind: str = "glyph"):
    path = _resolve_path(name, kind)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(name: str, kind: str = "glyph") -> dict:
    path = _resolve_path(name, kind)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_files(kind: str = ""):
    ext = EXTENSIONS.get(kind, "")
    return sorted(p.name for p in SCRIPTURE_PATH.glob(f"*{ext}"))

# Ritual shell command handlers

def export_thread(thread_data: list, name: str):
    save_json(name, {"thread": thread_data}, kind="thread")

# Placeholders for future import expansion
def import_bios(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def import_glyph(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
