"""Concrete repository implementations backed by SQLite and local storage."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from uuid import UUID

from .domain import UserState
from .models import QuizCard
from .repositories import FSRSStateRepository, NoteRepository, QuizCardRepository, UserMetadataRepository


def _json_default(value):
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")


class LocalNoteRepository(NoteRepository):
    """Persists markdown notes to the local filesystem."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str, note_id: UUID) -> Path:
        user_dir = self._base_path / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / f"{note_id}.md"

    def save(self, user_id: str, note_id: UUID, markdown: str) -> None:
        path = self._path(user_id, note_id)
        path.write_text(markdown, encoding="utf-8")

    def get(self, user_id: str, note_id: UUID) -> str:
        path = self._path(user_id, note_id)
        return path.read_text(encoding="utf-8")


class SqliteStudyRepository(QuizCardRepository, FSRSStateRepository, UserMetadataRepository):
    """Stores quiz cards, FSRS state and user metadata in a SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialise_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _initialise_schema(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS quiz_cards (
                    user_id TEXT NOT NULL,
                    card_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    due_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fsrs_state (
                    user_id TEXT NOT NULL,
                    card_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    PRIMARY KEY (user_id, card_id)
                );

                CREATE TABLE IF NOT EXISTS user_metadata (
                    user_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    # QuizCardRepository -------------------------------------------------
    def save_cards(
        self, user_id: str, cards: Iterable[QuizCard], due_in_minutes: int = 60
    ) -> None:
        due_at = datetime.utcnow() + timedelta(minutes=due_in_minutes)
        payloads = [
            (
                user_id,
                str(card.id),
                json.dumps(card.dict(), default=_json_default),
                due_at.isoformat(),
            )
            for card in cards
        ]
        if not payloads:
            return
        with self._lock:
            cursor = self._conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO quiz_cards (user_id, card_id, payload_json, due_at)
                VALUES (?, ?, ?, ?)
                """,
                payloads,
            )
            self._conn.commit()

    def get_due_cards(self, user_id: str) -> List[Tuple[QuizCard, datetime]]:
        now = datetime.utcnow().isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            rows = cursor.execute(
                "SELECT payload_json, due_at FROM quiz_cards WHERE user_id = ? AND due_at <= ?",
                (user_id, now),
            ).fetchall()
        cards: List[Tuple[QuizCard, datetime]] = []
        for row in rows:
            card_payload = json.loads(row["payload_json"])
            cards.append(
                (QuizCard.parse_obj(card_payload), datetime.fromisoformat(row["due_at"]))
            )
        return cards

    def update_due(self, user_id: str, card_id: UUID, next_due_at: datetime) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                UPDATE quiz_cards
                   SET due_at = ?
                 WHERE user_id = ? AND card_id = ?
                """,
                (next_due_at.isoformat(), user_id, str(card_id)),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Card {card_id} does not exist for user {user_id}")
            self._conn.commit()

    # FSRSStateRepository ------------------------------------------------
    def load_state(self, user_id: str, card_id: UUID) -> Optional[dict]:
        with self._lock:
            cursor = self._conn.cursor()
            row = cursor.execute(
                "SELECT state_json FROM fsrs_state WHERE user_id = ? AND card_id = ?",
                (user_id, str(card_id)),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["state_json"])

    def save_state(self, user_id: str, card_id: UUID, state: dict) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO fsrs_state (user_id, card_id, state_json)
                VALUES (?, ?, ?)
                """,
                (user_id, str(card_id), json.dumps(state)),
            )
            self._conn.commit()

    # UserMetadataRepository --------------------------------------------
    def get_user_state(self, user_id: str) -> UserState:
        with self._lock:
            cursor = self._conn.cursor()
            row = cursor.execute(
                "SELECT state_json FROM user_metadata WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            state = UserState()
            self.save_user_state(user_id, state)
            return state
        payload = json.loads(row["state_json"])
        return UserState.from_dict(payload)

    def save_user_state(self, user_id: str, state: UserState) -> None:
        payload = json.dumps(state.to_dict())
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO user_metadata (user_id, state_json)
                VALUES (?, ?)
                """,
                (user_id, payload),
            )
            self._conn.commit()


__all__ = ["LocalNoteRepository", "SqliteStudyRepository"]
