"""Core services implementing the learning and review workflows."""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import cycle
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


def extract_document(document_id: str) -> Tuple[Dict[str, List[str]], List[Dict[str, str]]]:
    """Simulated document extraction returning structured sections with citations."""

    sections = {
        "Overview": [
            "«Spaced repetition» schedules are refined in the uploaded analysis to balance retention and study time.",
            "The document emphasises aligning review difficulty with learner confidence to avoid burnout.",
        ],
        "Evidence": [
            "Comparative experiments show FSRS outperforming static intervals for digital flashcards.",
            "Learner diaries highlight the motivational impact of quick feedback loops.",
        ],
        "Practice Ideas": [
            "Integrating bite-sized quizzes after each note page keeps retrieval practice active.",
            "Weekly reflections encourage metacognitive monitoring of progress.",
        ],
    }
    citations = [
        {
            "id": "DOC1",
            "title": f"Document {document_id} Research Summary",
            "url": f"https://docs.local/{document_id}/summary",
            "snippet": "Key findings describing adaptive review cadences.",
        },
        {
            "id": "DOC2",
            "title": f"Document {document_id} Learner Interviews",
            "url": f"https://docs.local/{document_id}/interviews",
            "snippet": "Qualitative reflections on motivation and pacing.",
        },
        {
            "id": "DOC3",
            "title": "Open research on active recall",
            "url": DEFAULT_SOURCES[2],
            "snippet": "Overview article explaining active recall benefits.",
        },
    ]
    return sections, citations


def enrich_topic(topic_text: str) -> Tuple[Dict[str, List[str]], List[Dict[str, str]]]:
    """Fallback enrichment that mimics external knowledge fetching for a free-text topic."""

    topic = topic_text or "Learning Science"
    sections = {
        "Foundations": [
            f"«Spaced repetition» keeps {topic.lower()} content fresh by revisiting knowledge just before forgetting.",
            "Retrieval practice turns studying into an active challenge that deepens encoding.",
        ],
        "Techniques": [
            "Layering multiple quiz formats supports transfer across recognition and recall.",
            "Interleaving related subtopics prevents overfitting to a single pattern.",
        ],
        "Habits": [
            "Short feedback cycles sustain motivation, especially when sessions are tracked visually.",
            "Reflection prompts encourage learners to monitor confidence and adjust schedules.",
        ],
    }
    citations = [
        {
            "id": "EXT1",
            "title": "Spacing effect in practice",
            "url": DEFAULT_SOURCES[0],
            "snippet": "Research summary on distributed practice benefits.",
        },
        {
            "id": "EXT2",
            "title": "Flashcards as active learning tools",
            "url": DEFAULT_SOURCES[1],
            "snippet": "Article explaining why flashcards aid durable memory.",
        },
        {
            "id": "EXT3",
            "title": "Active recall and metacognition",
            "url": DEFAULT_SOURCES[2],
            "snippet": "Overview linking retrieval practice to monitoring skills.",
        },
    ]
    return sections, citations


def summarize_with_citations(
    heading: str, sections: Dict[str, List[str]], citations: List[Dict[str, str]]
) -> Tuple[str, List[str], List[Dict[str, str]]]:
    """Create a structured markdown summary and enrich citation metadata."""

    if not citations:
        citations = [
            {
                "id": "S1",
                "title": "Spacing effect in practice",
                "url": DEFAULT_SOURCES[0],
                "snippet": "Research summary on distributed practice benefits.",
            }
        ]

    for index, citation in enumerate(citations, start=1):
        citation.setdefault("id", f"S{index}")
        citation.setdefault("title", f"Source {index}")
        citation.setdefault("url", DEFAULT_SOURCES[(index - 1) % len(DEFAULT_SOURCES)])
        citation.setdefault("snippet", "Supporting evidence for the generated summary.")

    citation_cycle = cycle(citations)
    section_entries: List[Tuple[str, List[Tuple[str, Dict[str, str]]]]] = []
    ordered_points: List[Tuple[str, Dict[str, str]]] = []
    cloze_candidates: List[str] = []

    for section_name, bullet_points in sections.items():
        formatted_points: List[Tuple[str, Dict[str, str]]] = []
        for point in bullet_points:
            citation_info = next(citation_cycle)
            formatted = f"{point} [{citation_info['id']}]"
            formatted_points.append((formatted, citation_info))
            ordered_points.append((formatted, citation_info))
            if "«" in point and "»" in point:
                start = point.index("«") + 1
                end = point.index("»", start)
                term = point[start:end]
                cloze_text = point.replace(f"«{term}»", "____")
                cloze_candidates.append(f"{cloze_text} ({term})")
        section_entries.append((section_name, formatted_points))

    if not cloze_candidates and ordered_points:
        fallback_point = ordered_points[0][0]
        cloze_candidates.append(fallback_point.replace("[", "(").replace("]", ")"))

    tldr_points = [point for point, _ in ordered_points[:3]]

    lines: List[str] = [f"# {heading}", "", "## TL;DR"]
    if tldr_points:
        lines.extend(f"- {point}" for point in tldr_points)
    else:
        lines.append("- Key takeaways will appear here once content is provided.")

    lines.append("")
    lines.append("## Sectioned Notes")
    for section_name, formatted_points in section_entries:
        lines.append(f"### {section_name}")
        for formatted, _ in formatted_points:
            lines.append(f"- {formatted}")
        lines.append("")

    lines.append("## Cloze Cues")
    if cloze_candidates:
        lines.extend(f"- {candidate}" for candidate in cloze_candidates)
    else:
        lines.append("- Additional review questions will be generated soon.")

    lines.append("")
    lines.append("## Sources")
    for index, citation in enumerate(citations, start=1):
        lines.append(
            f"{index}. [{citation['title']}]({citation['url']}) — {citation['snippet']}"
        )

    markdown = "\n".join(lines).rstrip()
    source_list = [citation["url"] for citation in citations]
    return markdown, source_list, citations


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


DEFAULT_DIFFICULTY = 5.0
MIN_DIFFICULTY = 1.0
MAX_DIFFICULTY = 10.0
MIN_STABILITY = 0.1
INITIAL_STABILITY = {1: 0.2, 2: 0.6, 3: 1.5, 4: 3.0}
DIFFICULTY_DELTA = {1: 1.0, 2: 0.4, 3: -0.2, 4: -0.5}
STABILITY_GROWTH = {1: -0.5, 2: 0.0, 3: 1.2, 4: 2.0}
LEECH_LAPSES = 3


@dataclass
class FSRSState:
    """Persisted FSRS scheduling parameters for a card."""

    stability: float
    difficulty: float
    last_review: Optional[datetime]
    due: datetime
    lapses: int = 0
    reviews: int = 0

    @property
    def is_leech(self) -> bool:
        return self.lapses >= LEECH_LAPSES and self.stability <= 1.0


class InMemoryRepository:
    """Toy repository that simulates persistence for early development."""

    def __init__(self) -> None:
        self._notes: Dict[UUID, str] = {}
        self._note_citations: Dict[UUID, List[Dict[str, str]]] = {}
        self._cards: Dict[UUID, QuizCard] = {}
        self._states: Dict[UUID, FSRSState] = {}
        self._user_state: Dict[str, UserState] = {}

    # region Notes
    def save_note(
        self, note_id: UUID, markdown: str, citations: Optional[List[Dict[str, str]]] = None
    ) -> None:
        self._notes[note_id] = markdown
        self._note_citations[note_id] = list(citations or [])

    def get_note(self, note_id: UUID) -> str:
        return self._notes[note_id]

    def get_note_citations(self, note_id: UUID) -> List[Dict[str, str]]:
        return self._note_citations.get(note_id, [])

    # endregion

    # region Cards
    def save_cards(self, cards: Iterable[QuizCard], due_in_minutes: int = 60) -> None:
        now = datetime.utcnow()
        initial_due = now + timedelta(minutes=due_in_minutes)
        initial_stability = max(due_in_minutes / (60 * 24), 0.2)
        for card in cards:
            self._cards[card.id] = card
            self._states[card.id] = FSRSState(
                stability=initial_stability,
                difficulty=DEFAULT_DIFFICULTY,
                last_review=None,
                due=initial_due,
            )

    def get_due_cards(self) -> List[Tuple[QuizCard, "FSRSState"]]:
        now = datetime.utcnow()
        return [
            (self._cards[card_id], state)
            for card_id, state in self._states.items()
            if state.due <= now
        ]

    def get_card_state(self, card_id: UUID) -> "FSRSState":
        return self._states[card_id]

    def save_card_state(self, card_id: UUID, state: "FSRSState") -> None:
        self._states[card_id] = state

    # endregion

    def get_user_state(self, user_id: str) -> UserState:
        if user_id not in self._user_state:
            self._user_state[user_id] = UserState()
        return self._user_state[user_id]


def build_markdown_summary(
    request: LearnPrepareRequest,
) -> Tuple[str, List[str], List[Dict[str, str]]]:
    """Generate a markdown summary enriched with citations for notes."""

    if request.document_id:
        sections, citations = extract_document(request.document_id)
        heading = f"Insights from document {request.document_id}"
    else:
        sections, citations = enrich_topic(request.topic_text or "Learning Science")
        heading = request.topic_text or "Learning Science Overview"

    markdown, sources, citation_metadata = summarize_with_citations(heading, sections, citations)
    return markdown, sources, citation_metadata


def build_quiz_items(request: LearnPrepareRequest) -> List[QuizCard]:
    """Produces a minimal deterministic quiz set to unblock integration."""

    items: List[QuizCard] = []
    topic = request.topic_text or "the uploaded document"

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

    return items * max(request.per_type, 1)


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

    shuffled, _ = fisher_yates_shuffle(card.options or [], session_seed)
    correct_option = next(opt for opt in shuffled if opt.text == card.answer)
    correct_index = shuffled.index(correct_option)

    if user_state.has_bias() and correct_index == user_state.most_used_position():
        target = user_state.least_used_position()
        shuffled[correct_index], shuffled[target] = shuffled[target], shuffled[correct_index]
        correct_index = target

    user_state.register_position(correct_index)
    return shuffled, correct_index


def compute_fsrs_state(state: FSRSState, grade: int, now: datetime) -> FSRSState:
    """Updates FSRS scheduling parameters based on the provided grade."""

    if grade < 1 or grade > 4:
        raise ValueError("FSRS grades must be between 1 and 4 inclusive")

    elapsed_days = 0.0
    if state.last_review:
        elapsed_days = max((now - state.last_review).total_seconds() / 86400, 0.0)

    retrievability = 1.0
    if state.last_review and state.stability > 0:
        retrievability = math.exp(math.log(0.9) * elapsed_days / max(state.stability, MIN_STABILITY))

    difficulty_adjustment = DIFFICULTY_DELTA[grade] * (1 - retrievability)
    new_difficulty = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, state.difficulty + difficulty_adjustment))

    if state.last_review is None:
        new_stability = INITIAL_STABILITY[grade]
    else:
        growth_factor = STABILITY_GROWTH[grade]
        if grade == 1:
            new_stability = max(MIN_STABILITY, state.stability * 0.3)
        else:
            growth = 1 + growth_factor * (1 - new_difficulty / MAX_DIFFICULTY) * (1 - retrievability)
            new_stability = max(MIN_STABILITY, state.stability * growth)

    if grade == 1:
        lapses = state.lapses + 1
    else:
        lapses = state.lapses

    updated = FSRSState(
        stability=new_stability,
        difficulty=new_difficulty,
        last_review=now,
        due=now + timedelta(days=new_stability),
        lapses=lapses,
        reviews=state.reviews + 1,
    )
    return updated


class LearningService:
    """Facade exposing the prepare endpoint behaviour."""

    def __init__(self, repository: InMemoryRepository) -> None:
        self._repository = repository

    def prepare(self, request: LearnPrepareRequest) -> LearnPrepareResponse:
        note_id = uuid4()
        markdown, sources, citations = build_markdown_summary(request)
        if not sources:
            raise ValueError("Generated notes must reference at least one source")
        self._repository.save_note(note_id, markdown, citations)
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
        for card, state in self._repository.get_due_cards():
            options = None
            answer_index = None
            if card.type == "mcq" and card.options:
                shuffled, answer_index = render_mcq(
                    card,
                    user_state,
                    session_seed=hash((card.id, user_id, state.due)),
                )
                options = shuffled
            due_cards.append(
                ReviewCard(
                    card_id=card.id,
                    front=card.front,
                    options=options,
                    answer_index=answer_index,
                    type=card.type,
                    due_at=state.due,
                    is_leech=state.is_leech,
                )
            )
        return ReviewDueResponse(due=due_cards)

    def grade(self, request: ReviewGradeRequest) -> ReviewGradeResponse:
        now = datetime.utcnow()
        state = self._repository.get_card_state(request.card_id)
        updated_state = compute_fsrs_state(state, request.grade, now)
        self._repository.save_card_state(request.card_id, updated_state)
        state_payload = {
            "stability_days": round(updated_state.stability, 4),
            "difficulty": round(updated_state.difficulty, 4),
            "last_review_at": updated_state.last_review.isoformat() if updated_state.last_review else None,
            "lapses": updated_state.lapses,
        }
        return ReviewGradeResponse(
            card_id=request.card_id,
            next_due_at=updated_state.due,
            state=state_payload,
        )


__all__ = [
    "FSRSState",
    "InMemoryRepository",
    "LearningService",
    "ReviewService",
    "build_markdown_summary",
    "build_quiz_items",
    "fisher_yates_shuffle",
    "compute_fsrs_state",
    "render_mcq",
]
