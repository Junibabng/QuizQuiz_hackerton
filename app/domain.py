"""Domain models shared across services and repositories."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List


@dataclass
class UserState:
    """Keeps track of position bias for a single user."""

    answer_position_histogram: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    recent_positions: Deque[int] = field(default_factory=lambda: deque(maxlen=20))

    def register_position(self, position: int) -> None:
        self.answer_position_histogram[position] += 1
        self.recent_positions.append(position)

    def least_used_position(self) -> int:
        return min(
            range(len(self.answer_position_histogram)),
            key=self.answer_position_histogram.__getitem__,
        )

    def most_used_position(self) -> int:
        return max(
            range(len(self.answer_position_histogram)),
            key=self.answer_position_histogram.__getitem__,
        )

    def has_bias(self, threshold: float = 0.35) -> bool:
        total = len(self.recent_positions)
        if not total:
            return False
        counts = {pos: list(self.recent_positions).count(pos) for pos in set(self.recent_positions)}
        return any(count / total > threshold for count in counts.values())

    def to_dict(self) -> dict:
        return {
            "answer_position_histogram": list(self.answer_position_histogram),
            "recent_positions": list(self.recent_positions),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "UserState":
        state = cls()
        histogram = payload.get("answer_position_histogram")
        if histogram and len(histogram) == len(state.answer_position_histogram):
            state.answer_position_histogram = list(histogram)
        positions = payload.get("recent_positions") or []
        state.recent_positions = deque(positions, maxlen=20)
        return state


__all__ = ["UserState"]
