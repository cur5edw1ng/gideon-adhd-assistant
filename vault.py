"""
Gideon - Encrypted Vault (Level 3)

Everything Gideon remembers is stored in one local SQLite database
(gideon_vault.db). Every record is encrypted with Fernet (AES-128-CBC +
HMAC-SHA256, authenticated encryption) from the `cryptography` library before
it touches the disk.

Tables (each row is: id + one encrypted blob):
  memories      goals, projects, coursework, personal context
  tasks         things you said you need to do, with done/not done
  reminders     deadlines, exam dates, reminders
  security_log  voice lock events (intrusion attempts, enrollment, resets)

What is encrypted: all text, categories, dates, and flags (inside the blob).
What is NOT encrypted: row ids and blob sizes. An attacker holding only the
database file can see roughly how many records exist and nothing else.

The encryption key lives in gideon.key next to this file and is generated
automatically on first run. Anyone who has BOTH files can read the vault, so
never commit gideon.key or gideon_vault.db to Git and never keep them together
in a shared or cloud-synced folder.

The same key also encrypts voiceprint.enc (see voicelock.py).
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
from typing import Any, Dict, List, Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken

BASE_DIR = Path(__file__).resolve().parent
KEY_PATH = BASE_DIR / "gideon.key"
DB_PATH = BASE_DIR / "gideon_vault.db"

VALID_CATEGORIES = ("goal", "project", "coursework", "context")
UNREADABLE = "unreadable"
MAX_MEMORY_CHARS = 200
MAX_TASK_CHARS = 160
MAX_REMINDER_CHARS = 160
MAX_LOG_DETAIL_CHARS = 300

_TABLES = ("memories", "tasks", "reminders", "security_log")

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
    """Raised when the vault key or an encrypted file cannot be used."""


@dataclass(frozen=True)
class Memory:
    id: int
    category: str
    text: str
    created_at: str


@dataclass(frozen=True)
class Task:
    id: int
    text: str
    done: bool
    created_at: str


@dataclass(frozen=True)
class Reminder:
    id: int
    title: str
    due: str  # "YYYY-MM-DD", or "" when no date was given
    created_at: str


@dataclass(frozen=True)
class LogEntry:
    id: int
    timestamp: str
    event: str
    detail: str


def looks_sensitive(text: str) -> bool:
    """Return True if the text looks like a credential, ID number, phone, or email."""
    return bool(_SENSITIVE_WORDS.search(text) or _LONG_NUMBER.search(text) or _EMAIL.search(text))


def normalize_due(value: Any) -> str:
    """Return a valid YYYY-MM-DD string, or "" if the value is not a real date."""
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return ""


def _clean(text: Any, limit: int) -> str:
    return " ".join(str(text).split())[:limit].strip()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _now_seconds() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
        # Overwrite deleted content with zeros so erased records do not linger in free pages.
        conn.execute("PRAGMA secure_delete = ON")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            for table in _TABLES:
                conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "payload BLOB NOT NULL)"
                )
            conn.commit()

    @staticmethod
    def _check_table(table: str) -> str:
        if table not in _TABLES:
            raise ValueError(f"Unknown vault table: {table}")
        return table

    def _seal(self, record: Dict[str, Any]) -> bytes:
        return self._fernet.encrypt(json.dumps(record, ensure_ascii=False).encode("utf-8"))

    def _open(self, payload: bytes) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(self._fernet.decrypt(bytes(payload)).decode("utf-8"))
        except (InvalidToken, ValueError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _insert(self, table: str, record: Dict[str, Any]) -> int:
        table = self._check_table(table)
        with closing(self._connect()) as conn:
            cursor = conn.execute(f"INSERT INTO {table} (payload) VALUES (?)", (self._seal(record),))
            conn.commit()
            return int(cursor.lastrowid)

    def _fetch(self, table: str) -> List[Tuple[int, Optional[Dict[str, Any]]]]:
        """All rows oldest first. The record is None if it cannot be decrypted."""
        table = self._check_table(table)
        with closing(self._connect()) as conn:
            rows = conn.execute(f"SELECT id, payload FROM {table} ORDER BY id ASC").fetchall()
        return [(int(row_id), self._open(payload)) for row_id, payload in rows]

    def _update(self, table: str, row_id: int, record: Dict[str, Any]) -> bool:
        table = self._check_table(table)
        with closing(self._connect()) as conn:
            cursor = conn.execute(f"UPDATE {table} SET payload = ? WHERE id = ?", (self._seal(record), row_id))
            conn.commit()
            return cursor.rowcount > 0

    def _remove(self, table: str, row_id: int) -> bool:
        table = self._check_table(table)
        with closing(self._connect()) as conn:
            cursor = conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _wipe(self, table: str) -> int:
        table = self._check_table(table)
        with closing(self._connect()) as conn:
            cursor = conn.execute(f"DELETE FROM {table}")
            removed = cursor.rowcount
            conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table,))
            conn.commit()
            conn.execute("VACUUM")
        return removed

    # ----- raw encryption (used for voiceprint.enc) ----------------------------

    def encrypt_bytes(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def decrypt_bytes(self, token: bytes) -> bytes:
        try:
            return self._fernet.decrypt(token)
        except InvalidToken as exc:
            raise VaultError(
                "This encrypted file cannot be decrypted with the current gideon.key "
                "(wrong key or damaged data)."
            ) from exc

    # ----- memories ----------------------------------------------------------

    def add(self, category: str, text: str) -> Optional[int]:
        """Encrypt and save one memory. Returns its id, or None if it was skipped."""
        text = _clean(text, MAX_MEMORY_CHARS)
        if not text or looks_sensitive(text):
            return None
        if category not in VALID_CATEGORIES:
            category = "context"

        existing = {m.text.casefold() for m in self.list_all() if m.category != UNREADABLE}
        if text.casefold() in existing:
            return None
        return self._insert("memories", {"c": category, "t": text, "d": _now()})

    def list_all(self) -> List[Memory]:
        """Every memory, decrypted, newest first."""
        memories: List[Memory] = []
        for row_id, record in reversed(self._fetch("memories")):
            if record is None:
                memories.append(
                    Memory(
                        id=row_id,
                        category=UNREADABLE,
                        text="This entry could not be decrypted (wrong key or damaged data). You can delete it.",
                        created_at="",
                    )
                )
            else:
                memories.append(
                    Memory(
                        id=row_id,
                        category=str(record.get("c", "context")),
                        text=str(record.get("t", "")),
                        created_at=str(record.get("d", "")),
                    )
                )
        return memories

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def delete(self, memory_id: int) -> bool:
        """Permanently erase one memory."""
        return self._remove("memories", memory_id)

    def clear_all(self) -> int:
        """Permanently erase every memory and compact the database file."""
        return self._wipe("memories")

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

    # ----- tasks -------------------------------------------------------------

    def add_task(self, text: str) -> Optional[int]:
        """Save a task. Returns its id, or None if skipped (empty, sensitive, or already open)."""
        text = _clean(text, MAX_TASK_CHARS)
        if not text or looks_sensitive(text):
            return None
        open_texts = {t.text.casefold() for t in self.list_tasks() if not t.done}
        if text.casefold() in open_texts:
            return None
        return self._insert("tasks", {"t": text, "x": 0, "d": _now()})

    def list_tasks(self) -> List[Task]:
        """Open tasks first (oldest first), then completed tasks (newest first)."""
        tasks: List[Task] = []
        for row_id, record in self._fetch("tasks"):
            if record is None:
                continue
            tasks.append(
                Task(
                    id=row_id,
                    text=str(record.get("t", "")),
                    done=bool(record.get("x", 0)),
                    created_at=str(record.get("d", "")),
                )
            )
        open_tasks = [t for t in tasks if not t.done]
        done_tasks = [t for t in tasks if t.done]
        return open_tasks + list(reversed(done_tasks))

    def set_task_done(self, task_id: int, done: bool) -> bool:
        record = dict(self._fetch("tasks")).get(task_id)
        if record is None:
            return False
        record["x"] = 1 if done else 0
        return self._update("tasks", task_id, record)

    def delete_task(self, task_id: int) -> bool:
        return self._remove("tasks", task_id)

    def clear_done_tasks(self) -> int:
        removed = 0
        for task in self.list_tasks():
            if task.done and self._remove("tasks", task.id):
                removed += 1
        return removed

    # ----- reminders and deadlines ---------------------------------------------

    def add_reminder(self, title: str, due: str = "") -> Optional[int]:
        """Save a reminder or deadline. `due` is YYYY-MM-DD or empty."""
        title = _clean(title, MAX_REMINDER_CHARS)
        if not title or looks_sensitive(title):
            return None
        due = normalize_due(due)
        existing = {(r.title.casefold(), r.due) for r in self.list_reminders()}
        if (title.casefold(), due) in existing:
            return None
        return self._insert("reminders", {"t": title, "u": due, "d": _now()})

    def list_reminders(self) -> List[Reminder]:
        """Sorted by urgency: earliest date first (overdue first), undated last."""
        reminders: List[Reminder] = []
        for row_id, record in self._fetch("reminders"):
            if record is None:
                continue
            reminders.append(
                Reminder(
                    id=row_id,
                    title=str(record.get("t", "")),
                    due=normalize_due(record.get("u", "")),
                    created_at=str(record.get("d", "")),
                )
            )
        reminders.sort(key=lambda r: (r.due == "", r.due, r.id))
        return reminders

    def delete_reminder(self, reminder_id: int) -> bool:
        return self._remove("reminders", reminder_id)

    # ----- security log ---------------------------------------------------------

    def add_log(self, event: str, detail: str = "") -> int:
        return self._insert(
            "security_log",
            {"e": _clean(event, 40), "t": _clean(detail, MAX_LOG_DETAIL_CHARS), "d": _now_seconds()},
        )

    def list_log(self, limit: int = 100) -> List[LogEntry]:
        """Security events, newest first."""
        entries: List[LogEntry] = []
        for row_id, record in reversed(self._fetch("security_log")):
            if record is None:
                continue
            entries.append(
                LogEntry(
                    id=row_id,
                    timestamp=str(record.get("d", "")),
                    event=str(record.get("e", "")),
                    detail=str(record.get("t", "")),
                )
            )
            if len(entries) >= limit:
                break
        return entries

    def count_log(self, event: str) -> int:
        return sum(1 for entry in self.list_log(limit=10_000) if entry.event == event)

    def clear_log(self) -> int:
        return self._wipe("security_log")
