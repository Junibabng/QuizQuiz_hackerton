"""Repository interfaces for QuizQuiz persistent state."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, List, Optional, Tuple
from uuid import UUID

from .domain import UserState
from .models import QuizCard


class NoteRepository(ABC):
    """Persist and retrieve learner notes stored in object storage."""

    @abstractmethod
    def save(self, user_id: str, note_id: UUID, markdown: str) -> None:
        """Persist markdown content for the learner."""

    @abstractmethod
    def get(self, user_id: str, note_id: UUID) -> str:
        """Retrieve markdown content for the learner."""


class QuizCardRepository(ABC):
    """Manage quiz card metadata and scheduling details."""

    @abstractmethod
    def save_cards(
        self, user_id: str, cards: Iterable[QuizCard], due_in_minutes: int = 60
    ) -> None:
        """Persist a batch of cards for a learner."""

    @abstractmethod
    def get_due_cards(self, user_id: str) -> List[Tuple[QuizCard, datetime]]:
        """Return all cards due at or before the current time."""

    @abstractmethod
    def update_due(self, user_id: str, card_id: UUID, next_due_at: datetime) -> None:
        """Update the due timestamp for a card."""


class FSRSStateRepository(ABC):
    """Maintain FSRS scheduler state per learner and card."""

    @abstractmethod
    def load_state(self, user_id: str, card_id: UUID) -> Optional[dict]:
        """Return the stored scheduler state, if present."""

    @abstractmethod
    def save_state(self, user_id: str, card_id: UUID, state: dict) -> None:
        """Persist the scheduler state."""


class UserMetadataRepository(ABC):
    """Store per-learner metadata required for rendering heuristics."""

    @abstractmethod
    def get_user_state(self, user_id: str) -> UserState:
        """Return the persisted user state, creating it if necessary."""

    @abstractmethod
    def save_user_state(self, user_id: str, state: UserState) -> None:
        """Persist the updated user state."""


__all__ = [
    "NoteRepository",
    "QuizCardRepository",
    "FSRSStateRepository",
    "UserMetadataRepository",
]
