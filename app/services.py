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
class BiasConfig:
    """Tunable configuration covering bias detection heuristics."""

    position_bias_threshold: float = 0.35
    option_reuse_threshold: float = 0.65
    position_history_window: int = 20
    option_history_window: int = 80
    correct_text_window: int = 10
    stale_regeneration_trigger: int = 2


@dataclass
class UserState:
    """Keeps track of position, option and answer text history for a single user."""

    config: BiasConfig = field(default_factory=BiasConfig)
    answer_position_histogram: List[int] = field(default_factory=list)
    recent_positions: Deque[int] = field(init=False)
    position_history: Deque[Tuple[datetime, int]] = field(init=False)
    recent_correct_texts: Deque[str] = field(init=False)
    recent_option_ids: Deque[UUID] = field(init=False)
    option_use_counter: Dict[UUID, int] = field(default_factory=dict)
    answer_variant_counts: Dict[str, int] = field(default_factory=dict)
    regeneration_streak: int = 0

    def __post_init__(self) -> None:
        self._rebuild_windows()

    def _rebuild_windows(self) -> None:
        self.recent_positions = deque(
            getattr(self, "recent_positions", []), maxlen=self.config.position_history_window
        )
        self.position_history = deque(
            getattr(self, "position_history", []), maxlen=self.config.position_history_window
        )
        self.recent_correct_texts = deque(
            getattr(self, "recent_correct_texts", []), maxlen=self.config.correct_text_window
        )
        self.recent_option_ids = deque(
            getattr(self, "recent_option_ids", []), maxlen=self.config.option_history_window
        )

    def apply_config(self, config: BiasConfig) -> None:
        self.config = config
        self._rebuild_windows()

    # region Position helpers
    def _ensure_histogram_size(self, total_positions: int) -> None:
        if total_positions <= 0:
            return
        if len(self.answer_position_histogram) < total_positions:
            self.answer_position_histogram.extend([0] * (total_positions - len(self.answer_position_histogram)))

    def register_position(self, position: int, total_positions: int) -> None:
        self._ensure_histogram_size(total_positions)
        self.answer_position_histogram[position] += 1
        now = datetime.utcnow()
        self.recent_positions.append(position)
        self.position_history.append((now, position))

    def least_used_position(self) -> int:
        if not self.answer_position_histogram:
            return 0
        return min(range(len(self.answer_position_histogram)), key=self.answer_position_histogram.__getitem__)

    def most_used_position(self) -> int:
        if not self.answer_position_histogram:
            return 0
        return max(range(len(self.answer_position_histogram)), key=self.answer_position_histogram.__getitem__)

    def has_bias(self, threshold: Optional[float] = None) -> bool:
        total = len(self.recent_positions)
        if not total:
            return False
        threshold = threshold if threshold is not None else self.config.position_bias_threshold
        counts = {pos: list(self.recent_positions).count(pos) for pos in set(self.recent_positions)}
        return any(count / total > threshold for count in counts.values())

    def position_bias_weights(self, total_positions: int) -> List[float]:
        self._ensure_histogram_size(total_positions)
        if total_positions <= 0:
            return []
        subset = self.answer_position_histogram[:total_positions]
        max_count = max(subset) if subset else 0
        return [(max_count - count) + 1 for count in subset]

    # endregion

    # region Option history helpers
    def option_pool_reuse_ratio(self, option_ids: Iterable[UUID]) -> float:
        option_ids = list(option_ids)
        if not option_ids:
            return 0.0
        recent_seen = set(self.recent_option_ids)
        reused = sum(1 for option_id in option_ids if option_id in recent_seen)
        return reused / len(option_ids)

    def is_option_pool_stale(self, option_ids: Iterable[UUID]) -> bool:
        ratio = self.option_pool_reuse_ratio(option_ids)
        if ratio >= self.config.option_reuse_threshold:
            self.regeneration_streak += 1
            return self.regeneration_streak >= self.config.stale_regeneration_trigger
        self.regeneration_streak = 0
        return False

    def record_option_usage(self, option_ids: Iterable[UUID]) -> None:
        for option_id in option_ids:
            self.option_use_counter[option_id] = self.option_use_counter.get(option_id, 0) + 1
            self.recent_option_ids.append(option_id)

    def refresh_option_history(self, current_option_ids: Iterable[UUID]) -> None:
        self.recent_option_ids.clear()
        for option_id in current_option_ids:
            self.recent_option_ids.append(option_id)
        # soften old counters so regeneration has an effect without wiping history completely
        for option_id in list(self.option_use_counter.keys()):
            self.option_use_counter[option_id] = max(0, self.option_use_counter[option_id] - 1)
            if self.option_use_counter[option_id] == 0:
                del self.option_use_counter[option_id]
        self.regeneration_streak = 0

    # endregion

    # region Answer text helpers
    def register_correct_text(self, text: str) -> None:
        self.recent_correct_texts.append(text)

    def seen_answer_recently(self, text: str) -> bool:
        return text in self.recent_correct_texts

    def generate_answer_variant(self, base_text: str) -> Tuple[str, str]:
        occurrence = self.answer_variant_counts.get(base_text, 0) + 1
        self.answer_variant_counts[base_text] = occurrence
        variant_id = f"{abs(hash(base_text)) & 0xFFFF:04x}-v{occurrence}"
        variant_text = self._format_variant(base_text, occurrence)
        self.register_correct_text(base_text)
        return variant_text, variant_id

    def _format_variant(self, base_text: str, occurrence: int) -> str:
        synonym_map = {
            "fsrs": [
                "Flexible Spaced Repetition System",
                "Free Spaced Repetition Scheduler",
            ],
            "spacing effect": [
                "distributed practice effect",
                "temporal spacing benefit",
            ],
            "retrieval practice": [
                "active recall",
                "memory retrieval drills",
            ],
        }
        lower = base_text.lower()
        for keyword, synonyms in synonym_map.items():
            if keyword in lower and occurrence - 1 < len(synonyms):
                synonym = synonyms[occurrence - 1]
                return f"{base_text} (aka {synonym})"

        styles = [
            lambda text: text,
            lambda text: f"{text} (alternate phrasing)",
            lambda text: f"**{text}**",
            lambda text: f"_{text}_",
            lambda text: f"{text} — variant {occurrence}",
        ]
        style = styles[min(occurrence - 1, len(styles) - 1)]
        return style(base_text)

    # endregion


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

    def get_user_state(self, user_id: str, bias_config: Optional[BiasConfig] = None) -> UserState:
        if user_id not in self._user_state:
            self._user_state[user_id] = UserState(config=bias_config or BiasConfig())
        state = self._user_state[user_id]
        if bias_config is not None and state.config != bias_config:
            state.apply_config(bias_config)
        return state


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


def _weighted_position_assignment(
    options: List[QuizOption], weights: List[float], rng: random.Random
) -> List[QuizOption]:
    if not options:
        return []
    available_positions = list(range(len(options)))
    output: List[Optional[QuizOption]] = [None] * len(options)
    shuffled_options = options[:]
    rng.shuffle(shuffled_options)
    for option in shuffled_options:
        local_weights = [weights[idx] for idx in available_positions]
        total = sum(local_weights)
        if total <= 0:
            choice_index = rng.choice(range(len(available_positions)))
        else:
            pick = rng.uniform(0, total)
            cumulative = 0.0
            choice_index = 0
            for idx, weight in enumerate(local_weights):
                cumulative += weight
                if pick <= cumulative:
                    choice_index = idx
                    break
        position = available_positions.pop(choice_index)
        output[position] = option
    return [option for option in output if option is not None]


def render_mcq(
    card: QuizCard,
    user_state: UserState,
    session_seed: int,
    bias_config: Optional[BiasConfig] = None,
) -> Tuple[List[QuizOption], int, str, bool]:
    """Applies weighted shuffles, bias correction and answer variants for MCQ rendering."""

    if bias_config is not None and user_state.config != bias_config:
        user_state.apply_config(bias_config)

    options = card.options or []
    if not options:
        return [], -1, "", False

    option_ids = [option.id for option in options]
    stale_pool = user_state.is_option_pool_stale(option_ids)
    seed = session_seed + (1 if stale_pool else 0)
    base_shuffled, _ = fisher_yates_shuffle(options, seed)
    weights = user_state.position_bias_weights(len(base_shuffled))
    rng = random.Random(seed)
    weighted = _weighted_position_assignment(base_shuffled, weights, rng)

    correct_source = next(opt for opt in options if opt.text == card.answer)
    correct_index = next(idx for idx, opt in enumerate(weighted) if opt.id == correct_source.id)

    if user_state.has_bias() and correct_index == user_state.most_used_position():
        target = user_state.least_used_position()
        weighted[correct_index], weighted[target] = weighted[target], weighted[correct_index]
        correct_index = target

    variant_text, variant_id = user_state.generate_answer_variant(correct_source.text)
    regenerated_suffix = ""
    if stale_pool:
        regenerated_suffix = "-regen"
        user_state.refresh_option_history(option_ids)

    presented_options: List[QuizOption] = []
    for idx, option in enumerate(weighted):
        text = option.text
        if option.id == correct_source.id:
            text = variant_text
        presented_options.append(QuizOption(id=option.id, text=text))

    user_state.register_position(correct_index, len(presented_options))
    user_state.record_option_usage([option.id for option in presented_options])

    return presented_options, correct_index, variant_id + regenerated_suffix, stale_pool


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

    def __init__(self, repository: InMemoryRepository, bias_config: Optional[BiasConfig] = None) -> None:
        self._repository = repository
        self._bias_config = bias_config or BiasConfig()

    def get_due(self, user_id: str) -> ReviewDueResponse:
        user_state = self._repository.get_user_state(user_id, bias_config=self._bias_config)
        due_cards: List[ReviewCard] = []
        for card, due_at in self._repository.get_due_cards():
            options = None
            answer_index = None
            variant_id = None
            regenerated = False
            if card.type == "mcq" and card.options:
                (
                    shuffled,
                    answer_index,
                    variant_id,
                    regenerated,
                ) = render_mcq(
                    card,
                    user_state,
                    session_seed=hash((card.id, user_id, due_at)),
                    bias_config=self._bias_config,
                )
                options = shuffled
            due_cards.append(
                ReviewCard(
                    card_id=card.id,
                    front=card.front,
                    options=options,
                    answer_index=answer_index,
                    type=card.type,
                    due_at=due_at,
                    variant_id=variant_id,
                    stale_pool=regenerated,
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
    "BiasConfig",
    "InMemoryRepository",
    "LearningService",
    "ReviewService",
    "UserState",
    "build_markdown_summary",
    "build_quiz_items",
    "fisher_yates_shuffle",
    "render_mcq",
]
