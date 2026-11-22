"""
Gideon - Encrypted Memory Vault

Stores memories in a local SQLite database (gideon_vault.db). Every memory is
encrypted with Fernet (AES-128-CBC + HMAC-SHA256, authenticated encryption)
from the `cryptography` library before it touches the disk.

What is encrypted:  category, text, and timestamp (all inside one encrypted blob).
What is NOT encrypted: the row id and the size of each blob (so an attacker with
only the database file can see roughly how many memories exist, nothing more).

The encryption key lives in gideon.key next to this file. It is generated
automatically on first run. Anyone who has BOTH files can read the vault, so
never commit gideon.key or gideon_vault.db to Git and never store them together
in a shared or cloud-synced folder.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from cryptography.fernet import Fernet, InvalidToken

BASE_DIR = Path(__file__).resolve().parent
KEY_PATH = BASE_DIR / "gideon.key"
DB_PATH = BASE_DIR / "gideon_vault.db"

VALID_CATEGORIES = ("goal", "project", "coursework", "context")
UNREADABLE = "unreadable"
MAX_MEMORY_CHARS = 200

# Things Gideon must never store, even if the language model suggests them:
# credentials, government/financial ID numbers, phone numbers, and email addresses.
_SENSITIVE_WORDS = re.compile(
    r"\b(password|passcode|passphrase|otp|api[ -]?key|secret key|private key|cvv|"
    r"aadhaar|aadhar|pan card|ssn|social security|credit card|debit card|"
    r"card number|bank account|account number)\b",
    re.IGNORECASE,
)
_LONG_NUMBER = re.compile(r"(?:\d[ -]?){9,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class VaultError(Exception):
    """Raised when the vault key or database cannot be used."""


@dataclass(frozen=True)
class Memory:
    id: int
    category: str
    text: str
    created_at: str


def looks_sensitive(text: str) -> bool:
    """Return True if the text looks like a credential, ID number, phone, or email."""
    return bool(_SENSITIVE_WORDS.search(text) or _LONG_NUMBER.search(text) or _EMAIL.search(text))


def _restrict_to_current_user(path: Path) -> None:
    """Best effort: make the key file readable only by the current user."""
    try:
        if os.name == "nt":
            user = os.environ.get("USERNAME", "")
            if user:
                subprocess.run(
                    ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                    check=True,
                    capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        else:
            os.chmod(path, 0o600)
    except Exception:
        # Permissions hardening is a bonus; the vault still works without it.
        pass


def _load_or_create_key(key_path: Path) -> bytes:
    """Load the Fernet key, creating a fresh one on first run."""
    if not key_path.exists():
        key = Fernet.generate_key()
        try:
            # "xb" = create exclusively, so two sessions can never overwrite each other's key.
            with open(key_path, "xb") as handle:
                handle.write(key)
            _restrict_to_current_user(key_path)
        except FileExistsError:
            pass  # another session created it first; read that one below

    key = key_path.read_bytes().strip()
    try:
        Fernet(key)
    except Exception as exc:
        raise VaultError(
            f"{key_path.name} is not a valid Fernet key. If you deleted or replaced it, "
            "the old vault can no longer be decrypted. Move gideon_vault.db aside to start fresh."
        ) from exc
    return key


class Vault:
    def __init__(self, db_path: Path = DB_PATH, key_path: Path = KEY_PATH) -> None:
        self.db_path = Path(db_path)
        self.key_path = Path(key_path)
        self._fernet = Fernet(_load_or_create_key(self.key_path))
        self._init_db()

    # ----- database plumbing -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        # Overwrite deleted content with zeros so erased memories do not linger in free pages.
        conn.execute("PRAGMA secure_delete = ON")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memories ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "payload BLOB NOT NULL)"
            )
            conn.commit()

    def _encrypt(self, category: str, text: str, created_at: str) -> bytes:
        blob = json.dumps({"c": category, "t": text, "d": created_at}, ensure_ascii=False)
        return self._fernet.encrypt(blob.encode("utf-8"))

    def _decrypt(self, row_id: int, payload: bytes) -> Memory:
        try:
            data = json.loads(self._fernet.decrypt(bytes(payload)).decode("utf-8"))
            return Memory(
                id=row_id,
                category=str(data.get("c", "context")),
                text=str(data.get("t", "")),
                created_at=str(data.get("d", "")),
            )
        except (InvalidToken, ValueError, UnicodeDecodeError):
            return Memory(
                id=row_id,
                category=UNREADABLE,
                text="This entry could not be decrypted (wrong key or damaged data). You can delete it.",
                created_at="",
            )

    # ----- public API --------------------------------------------------------

    def add(self, category: str, text: str) -> Optional[int]:
        """Encrypt and save one memory. Returns its id, or None if it was skipped."""
        text = " ".join(text.split())[:MAX_MEMORY_CHARS].strip()
        if not text or looks_sensitive(text):
            return None
        if category not in VALID_CATEGORIES:
            category = "context"

        existing = {m.text.casefold() for m in self.list_all() if m.category != UNREADABLE}
        if text.casefold() in existing:
            return None

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        payload = self._encrypt(category, text, created_at)
        with closing(self._connect()) as conn:
            cursor = conn.execute("INSERT INTO memories (payload) VALUES (?)", (payload,))
            conn.commit()
            return cursor.lastrowid

    def list_all(self) -> List[Memory]:
        """Return every memory, decrypted, newest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT id, payload FROM memories ORDER BY id DESC").fetchall()
        return [self._decrypt(row_id, payload) for row_id, payload in rows]

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def delete(self, memory_id: int) -> bool:
        """Permanently erase one memory. Returns True if a row was removed."""
        with closing(self._connect()) as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self) -> int:
        """Permanently erase every memory and compact the database file."""
        with closing(self._connect()) as conn:
            cursor = conn.execute("DELETE FROM memories")
            removed = cursor.rowcount
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'memories'")
            conn.commit()
            conn.execute("VACUUM")
        return removed

    def summary_for_prompt(self, max_items: int = 20, max_chars: int = 1400) -> str:
        """A compact plain-text digest of recent memories for Gideon's system prompt."""
        lines: List[str] = []
        used = 0
        usable = [m for m in self.list_all() if m.category in VALID_CATEGORIES]
        for memory in usable[:max_items]:
            line = f"- [{memory.category}] {memory.text[:160]}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)
