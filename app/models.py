"""Pydantic models for the QuizQuiz Hackerton backend."""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, root_validator, validator


QuizType = Literal["mcq", "cloze", "short"]


class QuizOption(BaseModel):
    """Multiple-choice option representation."""

    id: UUID = Field(default_factory=uuid4)
    text: str


class QuizCard(BaseModel):
    """Quiz card payload persisted for a learner."""

    id: UUID = Field(default_factory=uuid4)
    type: QuizType
    front: str
    options: Optional[List[QuizOption]] = None
    answer: str
    explanation: str
    sources: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    difficulty: Literal["auto", "easy", "medium", "hard"] = "auto"

    @validator("options", always=True)
    def validate_options(cls, value: Optional[List[QuizOption]], values: dict) -> Optional[List[QuizOption]]:
        quiz_type: QuizType = values.get("type", "mcq")
        if quiz_type == "mcq" and (not value or len(value) < 2):
            raise ValueError("MCQ items require at least two options")
        if quiz_type != "mcq" and value:
            raise ValueError("Only MCQ items may define options")
        return value


class LearnPrepareRequest(BaseModel):
    """Input body for /v1/learn/prepare."""

    user_id: str
    document_id: Optional[str] = None
    topic_text: Optional[str] = None
    note_style: Optional[Literal["bullet", "outline", "anki"]] = "bullet"
    quiz_types: Optional[List[QuizType]] = Field(default_factory=lambda: ["mcq", "cloze", "short"])
    per_type: int = 1

    @root_validator
    def validate_input(cls, values: dict) -> dict:
        if not values.get("document_id") and not values.get("topic_text"):
            raise ValueError("Either document_id or topic_text must be provided")
        return values


class LearnPrepareResponse(BaseModel):
    """Response body for /v1/learn/prepare."""

    note_id: UUID
    content_md: str
    sources: List[str]
    items: List[QuizCard]


class ReviewCard(BaseModel):
    """Card returned to the learner for review."""

    card_id: UUID
    front: str
    options: Optional[List[QuizOption]]
    answer_index: Optional[int]
    type: QuizType
    due_at: datetime
    variant_id: Optional[str] = None
    stale_pool: bool = False
    is_leech: bool = False


class ReviewDueResponse(BaseModel):
    due: List[ReviewCard]


class ReviewGradeRequest(BaseModel):
    user_id: str
    card_id: UUID
    grade: Literal[1, 2, 3, 4]


class ReviewGradeResponse(BaseModel):
    card_id: UUID
    next_due_at: datetime
    state: dict


__all__ = [
    "QuizCard",
    "QuizOption",
    "LearnPrepareRequest",
    "LearnPrepareResponse",
    "ReviewCard",
    "ReviewDueResponse",
    "ReviewGradeRequest",
    "ReviewGradeResponse",
]
