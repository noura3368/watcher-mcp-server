#!/usr/bin/env python3
"""Watch raw_docs/ and keep the ColQwen page index (colrag/store.py) in sync with its PDFs."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from colrag import store  # noqa: E402

RAW_DIR = Path(os.getenv("RAW_DIR", "/data/nkhajehn/watcher-mcp-server/raw_docs"))
DB_PATH = Path(os.getenv("COLRAG_DB", "/data/nkhajehn/watcher-mcp-server/data/colqwen.lancedb"))
STATE_PATH = Path(os.getenv("STATE_PATH", "/data/nkhajehn/watcher-mcp-server/data/watcher_state.json"))


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def handle_file(path: Path, state: dict[str, Any]) -> None:
    if not path.exists() or not path.is_file():
        return

    # ColQwen embeds page images, so only PDFs can be indexed.
    if path.suffix.lower() != ".pdf":
        print(f"[skip] not a PDF: {path.name}", flush=True)
        return

    file_hash = sha256_file(path)
    # Entries from the old haiku.rag watcher (kind manual/normal) are re-indexed.
    old = state["files"].get(str(path), {})
    if old.get("kind") == "pdf" and old.get("sha256") == file_hash:
        print(f"[skip] unchanged: {path.name}", flush=True)
        return

    print(f"[index] {path.name}", flush=True)
    try:
        # index_pdf replaces any rows the file already has, so a changed PDF is not duplicated.
        pages = store.index_pdf(path, file_hash, db_path=DB_PATH)
    except Exception as e:
        print(f"[index] failed for {path.name}: {e}", flush=True)
        return
    state["files"][str(path)] = {"sha256": file_hash, "kind": "pdf", "pages": pages, "processed_at": time.time()}
    save_state(state)
    print(f"[index] {path.name}: {pages} pages indexed", flush=True)


def remove_file(path: Path, state: dict[str, Any]) -> None:
    info = state["files"].pop(str(path), None)
    save_state(state)
    if not info:
        print(f"[delete] no state for {path}", flush=True)
        return
    try:
        store.delete_source(str(path), db_path=DB_PATH)
        print(f"[delete] removed pages of {path.name}", flush=True)
    except Exception as e:
        print(f"[delete] failed for {path.name}: {e}", flush=True)


class Handler(FileSystemEventHandler):
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def on_created(self, event):
        if event.is_directory:
            return
        time.sleep(1)
        handle_file(Path(event.src_path), self.state)

    def on_modified(self, event):
        if event.is_directory:
            return
        time.sleep(1)
        handle_file(Path(event.src_path), self.state)

    def on_deleted(self, event):
        if event.is_directory:
            return
        remove_file(Path(event.src_path), self.state)


def initial_scan(state: dict[str, Any]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for path in RAW_DIR.rglob("*"):
        if path.is_file():
            handle_file(path, state)
    # Files deleted while the watcher was not running.
    for name in [p for p in state["files"] if not Path(p).exists()]:
        remove_file(Path(name), state)


def main() -> None:
    state = load_state()
    initial_scan(state)

    observer = Observer()
    observer.schedule(Handler(state), str(RAW_DIR), recursive=True)
    observer.start()

    print(f"Watching {RAW_DIR}", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()
