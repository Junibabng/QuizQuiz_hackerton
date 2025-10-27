"""Core services implementing the learning and review workflows."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Iterable, List, MutableMapping, Optional, Tuple
from uuid import UUID, uuid4

from .models import (
    LearnPrepareRequest,
    LearnPrepareResponse,
    QuizCard,
    QuizOption,
    ReviewCard,
    ReviewDueResponse,
    ReviewGradeRequest,
    ReviewGradeResponse,
)


DEFAULT_SOURCES = [
    "https://en.wikipedia.org/wiki/Spaced_repetition",
    "https://en.wikipedia.org/wiki/Flashcard",
    "https://en.wikipedia.org/wiki/Active_recall",
]


@dataclass
class UserState:
    """Keeps track of position bias for a single user."""

    answer_position_histogram: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    recent_positions: Deque[int] = field(default_factory=lambda: deque(maxlen=20))

    def register_position(self, position: int) -> None:
        self.answer_position_histogram[position] += 1
        self.recent_positions.append(position)

    def least_used_position(self) -> int:
        return min(range(len(self.answer_position_histogram)), key=self.answer_position_histogram.__getitem__)

    def most_used_position(self) -> int:
        return max(range(len(self.answer_position_histogram)), key=self.answer_position_histogram.__getitem__)

    def has_bias(self, threshold: float = 0.35) -> bool:
        total = sum(self.recent_positions and [1] * len(self.recent_positions) or [0])
        if not total:
            return False
        counts = {pos: list(self.recent_positions).count(pos) for pos in set(self.recent_positions)}
        return any(count / total > threshold for count in counts.values())


class InMemoryRepository:
    """Toy repository that simulates persistence for early development."""

    def __init__(self) -> None:
        self._notes: Dict[UUID, str] = {}
        self._cards: Dict[UUID, QuizCard] = {}
        self._due: Dict[UUID, datetime] = {}
        self._user_state: Dict[str, UserState] = {}

    # region Notes
    def save_note(self, note_id: UUID, markdown: str) -> None:
        self._notes[note_id] = markdown

    def get_note(self, note_id: UUID) -> str:
        return self._notes[note_id]

    # endregion

    # region Cards
    def save_cards(self, cards: Iterable[QuizCard], due_in_minutes: int = 60) -> None:
        for card in cards:
            self._cards[card.id] = card
            self._due[card.id] = datetime.utcnow() + timedelta(minutes=due_in_minutes)

    def get_due_cards(self) -> List[Tuple[QuizCard, datetime]]:
        now = datetime.utcnow()
        return [
            (self._cards[card_id], due_at)
            for card_id, due_at in self._due.items()
            if due_at <= now
        ]

    def update_due(self, card_id: UUID, next_due_at: datetime) -> None:
        self._due[card_id] = next_due_at

    # endregion

    def get_user_state(self, user_id: str) -> UserState:
        if user_id not in self._user_state:
            self._user_state[user_id] = UserState()
        return self._user_state[user_id]


def build_markdown_summary(request: LearnPrepareRequest) -> Tuple[str, List[str]]:
    """Creates a deterministic placeholder markdown summary for early testing."""

    heading = request.topic_text or f"Document {request.document_id}"
    tldr = "\n".join(
        [
            "## TL;DR",
            "- Active recall and spaced repetition accelerate retention.",
            "- Personalized scheduling ensures efficient studying.",
            "- Combining notes with quizzes encourages retrieval practice.",
            "- FSRS adapts intervals based on learner feedback.",
            "- Consistent review prevents forgetting curves from dominating.",
        ]
    )
    body = f"# {heading}\n\n" + tldr
    body += "\n\n## Key Points\n"
    body += "- «Spacing effect» supports distributing practice over time. [1]\n"
    body += "- «Retrieval practice» strengthens neural pathways. [2]\n"
    body += "- Adaptive schedulers like FSRS leverage review quality scores. [3]\n"
    body += "\n## Cloze Candidates\n"
    body += "- The spacing effect demonstrates improved recall when study sessions are spread over time.\n"
    body += "- FSRS updates the next review interval using grades from 1 to 4.\n"
    return body, DEFAULT_SOURCES.copy()


def build_quiz_items(request: LearnPrepareRequest) -> List[QuizCard]:
    """Produces a minimal deterministic quiz set to unblock integration."""

    items: List[QuizCard] = []
    topic = request.topic_text or "the uploaded document"

    def _clone_card(card: QuizCard) -> QuizCard:
        """Return a deep copy of a quiz card with fresh identifiers."""

        cloned_options = None
        if card.options:
            cloned_options = [QuizOption(text=option.text) for option in card.options]

        return QuizCard(
            type=card.type,
            front=card.front,
            options=cloned_options,
            answer=card.answer,
            explanation=card.explanation,
            sources=list(card.sources),
            skills=list(card.skills),
            difficulty=card.difficulty,
        )

    if "mcq" in request.quiz_types:
        options = [
            QuizOption(text="It optimizes review intervals using learner feedback."),
            QuizOption(text="It removes the need for spaced repetition."),
            QuizOption(text="It limits reviews to a single attempt."),
            QuizOption(text="It tracks only multiple-choice answers."),
        ]
        card = QuizCard(
            type="mcq",
            front=f"What is a key benefit of FSRS when studying {topic}?",
            options=options,
            answer=options[0].text,
            explanation="FSRS adjusts the timing of future reviews according to performance.",
            sources=DEFAULT_SOURCES[:2],
            skills=["spaced_repetition"],
        )
        items.append(card)

    if "cloze" in request.quiz_types:
        items.append(
            QuizCard(
                type="cloze",
                front="Spaced repetition combats the «forgetting curve» by revisiting material at increasing intervals.",
                answer="forgetting curve",
                explanation="The forgetting curve describes how information decays without reinforcement.",
                sources=DEFAULT_SOURCES[:1],
                skills=["spacing_effect"],
            )
        )

    if "short" in request.quiz_types:
        items.append(
            QuizCard(
                type="short",
                front="Describe how retrieval practice enhances long-term retention.",
                answer="It forces the learner to recall information, strengthening memory traces.",
                explanation="Retrieval practice requires active recall, which reinforces neural pathways.",
                sources=DEFAULT_SOURCES[1:2],
                skills=["retrieval_practice"],
            )
        )

    per_type = max(request.per_type, 1)
    if per_type == 1:
        return items

    expanded: List[QuizCard] = []
    for iteration in range(per_type):
        for card in items:
            if iteration == 0:
                expanded.append(card)
            else:
                expanded.append(_clone_card(card))

    return expanded


def fisher_yates_shuffle(options: List[QuizOption], seed: int) -> Tuple[List[QuizOption], MutableMapping[UUID, int]]:
    """Runs a deterministic Fisher–Yates shuffle on quiz options."""

    shuffled = options[:]
    rng = random.Random(seed)
    mapping: Dict[UUID, int] = {}
    for i in range(len(shuffled) - 1, 0, -1):
        j = rng.randint(0, i)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    for idx, option in enumerate(shuffled):
        mapping[option.id] = idx
    return shuffled, mapping


def render_mcq(card: QuizCard, user_state: UserState, session_seed: int) -> Tuple[List[QuizOption], int]:
    """Applies Fisher–Yates shuffle and bias correction for MCQ rendering."""

    shuffled, position_map = fisher_yates_shuffle(card.options or [], session_seed)
    correct_option = next(opt for opt in shuffled if opt.text == card.answer)
    correct_index = shuffled.index(correct_option)

    if user_state.has_bias() and correct_index == user_state.most_used_position():
        target = user_state.least_used_position()
        shuffled[correct_index], shuffled[target] = shuffled[target], shuffled[correct_index]
        correct_index = target

    user_state.register_position(correct_index)
    return shuffled, correct_index


class LearningService:
    """Facade exposing the prepare endpoint behaviour."""

    def __init__(self, repository: InMemoryRepository) -> None:
        self._repository = repository

    def prepare(self, request: LearnPrepareRequest) -> LearnPrepareResponse:
        note_id = uuid4()
        markdown, sources = build_markdown_summary(request)
        self._repository.save_note(note_id, markdown)
        items = build_quiz_items(request)
        self._repository.save_cards(items)
        return LearnPrepareResponse(note_id=note_id, content_md=markdown, sources=sources, items=items)


class ReviewService:
    """Manages the spaced repetition review loop."""

    def __init__(self, repository: InMemoryRepository) -> None:
        self._repository = repository

    def get_due(self, user_id: str) -> ReviewDueResponse:
        user_state = self._repository.get_user_state(user_id)
        due_cards: List[ReviewCard] = []
        for card, due_at in self._repository.get_due_cards():
            options = None
            answer_index = None
            if card.type == "mcq" and card.options:
                shuffled, answer_index = render_mcq(card, user_state, session_seed=hash((card.id, user_id, due_at)))
                options = shuffled
            due_cards.append(
                ReviewCard(
                    card_id=card.id,
                    front=card.front,
                    options=options,
                    answer_index=answer_index,
                    type=card.type,
                    due_at=due_at,
                )
            )
        return ReviewDueResponse(due=due_cards)

    def grade(self, request: ReviewGradeRequest) -> ReviewGradeResponse:
        now = datetime.utcnow()
        # Simplified FSRS placeholder: increase interval by grade multiplier.
        multiplier = {1: 5, 2: 30, 3: 60, 4: 120}[request.grade]
        next_due = now + timedelta(minutes=multiplier)
        self._repository.update_due(request.card_id, next_due)
        return ReviewGradeResponse(card_id=request.card_id, next_due_at=next_due, state={"interval_minutes": multiplier})


__all__ = [
    "InMemoryRepository",
    "LearningService",
    "ReviewService",
    "build_markdown_summary",
    "build_quiz_items",
    "fisher_yates_shuffle",
    "render_mcq",
]
