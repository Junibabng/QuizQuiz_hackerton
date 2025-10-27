"""Core services implementing the learning and review workflows."""
from __future__ import annotations

import itertools
import math
import random
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence, Tuple
from itertools import cycle
from typing import Deque, Dict, Iterable, List, MutableMapping, Optional, Tuple
from uuid import UUID, uuid4

from .metrics import METRICS
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
from .validators import validate_note, validate_quiz_cards


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
class KeyFact:
    term: str
    statement: str
    blank_statement: str
    skill: str
    sources: List[str]


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
        self._option_pools: Dict[str, List[str]] = {}
        self._lapses: Dict[UUID, int] = {}

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
            self._validate_card(card)
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

    def register_grade_outcome(self, card_id: UUID, grade: int) -> int:
        if grade == 1:
            self._lapses[card_id] = self._lapses.get(card_id, 0) + 1
        else:
            self._lapses[card_id] = 0
        return self._lapses[card_id]

    # endregion

    def get_user_state(self, user_id: str) -> UserState:
        if user_id not in self._user_state:
            self._user_state[user_id] = UserState()
        return self._user_state[user_id]

    # region Option pools
    def get_option_pool(self, skill: str) -> List[str]:
        return list(self._option_pools.get(skill, []))

    def extend_option_pool(self, skill: str, options: Sequence[str]) -> None:
        pool = self._option_pools.setdefault(skill, [])
        for option in options:
            if option not in pool:
                pool.append(option)

    # endregion

    def _validate_card(self, card: QuizCard) -> None:
        banned_patterns = ["all of the above", "none of the above"]
        answer_lower = card.answer.strip().lower()
        if not card.answer.strip():
            raise ValueError(f"Card {card.id} rejected due to empty answer")
        if any(pattern in answer_lower for pattern in banned_patterns):
            raise ValueError(f"Card {card.id} rejected due to banned answer pattern")

        if card.type == "mcq":
            option_texts = [opt.text for opt in card.options or []]
            if card.answer not in option_texts:
                raise ValueError(f"Card {card.id} rejected because answer missing from options")
            if len(set(option_texts)) != len(option_texts):
                raise ValueError(f"Card {card.id} rejected due to duplicate options")


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


def _parse_markdown_sections(markdown: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in markdown.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("## "):
            current = stripped[3:].strip()
            sections.setdefault(current, [])
        elif current:
            sections[current].append(stripped)
    return sections


def _extract_bullets(lines: Iterable[str]) -> List[str]:
    return [line[2:].strip() for line in lines if line.strip().startswith("- ")]


def _slugify_skill(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "general_concepts"


def _normalize_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _normalize_question(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return ""
    if cleaned.endswith("?"):
        return cleaned
    cleaned = cleaned.rstrip(".")
    return cleaned + "?"


def _sources_from_indices(indices: Iterable[str]) -> List[str]:
    sources: List[str] = []
    for idx in indices:
        if not idx.isdigit():
            continue
        pointer = int(idx) - 1
        if 0 <= pointer < len(DEFAULT_SOURCES):
            candidate = DEFAULT_SOURCES[pointer]
            if candidate not in sources:
                sources.append(candidate)
    return sources or DEFAULT_SOURCES[:1]


def _derive_key_facts(markdown: str) -> List[KeyFact]:
    sections = _parse_markdown_sections(markdown)
    key_points = _extract_bullets(sections.get("Key Points", []))
    facts: List[KeyFact] = []
    fallback_sources = DEFAULT_SOURCES[:1]
    for point in key_points:
        indices = re.findall(r"\[(\d+)\]", point)
        sources = _sources_from_indices(indices) if indices else fallback_sources
        term_match = re.search(r"«([^»]+)»", point)
        if term_match:
            term = term_match.group(1).strip()
            blank_template = re.sub(r"«[^»]+»", "_____", point)
        else:
            like_match = re.search(r"like ([A-Za-z0-9\- ]+)", point)
            if like_match:
                term = like_match.group(1).strip().split()[0]
            else:
                term = point.split()[0].strip("\"“”‘’.,")
            blank_template = point.replace(term, "_____", 1)
        statement = _normalize_sentence(re.sub(r"[«»]", "", re.sub(r"\[[^\]]*\]", "", point)))
        blank_statement = _normalize_sentence(re.sub(r"\[[^\]]*\]", "", blank_template))
        skill = _slugify_skill(term)
        facts.append(
            KeyFact(
                term=term,
                statement=statement,
                blank_statement=blank_statement,
                skill=skill,
                sources=sources,
            )
        )
    return facts


def _infer_difficulty(statement: str, answer: str) -> str:
    statement_tokens = statement.split()
    answer_tokens = answer.split()
    if len(statement_tokens) <= 12 and len(answer_tokens) <= 2:
        return "easy"
    if len(statement_tokens) <= 24 and len(answer_tokens) <= 4:
        return "medium"
    return "hard"


def _mcq_items_from_notes(facts: Sequence[KeyFact], repository: InMemoryRepository) -> Iterator[QuizCard]:
    if not facts:
        return iter(())

    fallback_distractors = [
        "Interleaving practice",
        "Passive rereading",
        "Short-term cramming",
        "Linear note review",
    ]
    all_terms = [fact.term for fact in facts]

    def generator() -> Iterator[QuizCard]:
        for fact in facts:
            correct = fact.term
            skill = fact.skill
            distractor_candidates: List[str] = [term for term in all_terms if term != correct]
            distractor_candidates.extend(repository.get_option_pool(skill))
            distractor_candidates.extend(fallback_distractors)

            unique_distractors: List[str] = []
            for candidate in distractor_candidates:
                candidate = candidate.strip()
                if not candidate or candidate == correct:
                    continue
                if candidate not in unique_distractors:
                    unique_distractors.append(candidate)
                if len(unique_distractors) >= 3:
                    break

            while len(unique_distractors) < 3:
                filler = f"Misconception {len(unique_distractors) + 1}"
                if filler != correct and filler not in unique_distractors:
                    unique_distractors.append(filler)

            option_texts = [correct] + unique_distractors[:3]
            options = [QuizOption(text=text) for text in option_texts]
            repository.extend_option_pool(skill, option_texts)
            front = _normalize_question(
                f"Which concept best completes the statement: {fact.blank_statement.strip()}"
            )
            difficulty = _infer_difficulty(fact.statement, correct)
            yield QuizCard(
                type="mcq",
                front=front,
                options=options,
                answer=correct,
                explanation=fact.statement,
                sources=fact.sources,
                skills=[skill],
                difficulty=difficulty,
            )

    return generator()


def _cloze_items_from_notes(markdown: str) -> Iterator[QuizCard]:
    sections = _parse_markdown_sections(markdown)
    candidates = _extract_bullets(sections.get("Cloze Candidates", []))

    def generator() -> Iterator[QuizCard]:
        for sentence in candidates:
            match = re.search(r"«([^»]+)»", sentence)
            if not match:
                continue
            answer = match.group(1).strip()
            front = _normalize_sentence(
                re.sub(r"\[[^\]]*\]", "", re.sub(r"«[^»]+»", "_____", sentence))
            )
            explanation = _normalize_sentence(
                re.sub(r"[«»]", "", re.sub(r"\[[^\]]*\]", "", sentence))
            )
            difficulty = _infer_difficulty(explanation, answer)
            yield QuizCard(
                type="cloze",
                front=front,
                answer=answer,
                explanation=explanation,
                sources=DEFAULT_SOURCES[:1],
                skills=[_slugify_skill(answer)],
                difficulty=difficulty,
            )

    return generator()


def _derive_subject(statement: str) -> str:
    lowered = statement.lower()
    for verb in ["accelerate", "ensures", "encourages", "adapts", "prevents"]:
        index = lowered.find(verb)
        if index != -1:
            return statement[:index].strip()
    return statement


def _short_answer_items_from_notes(markdown: str) -> Iterator[QuizCard]:
    sections = _parse_markdown_sections(markdown)
    bullet_points = _extract_bullets(sections.get("TL;DR", []))

    def generator() -> Iterator[QuizCard]:
        for point in bullet_points:
            statement = _normalize_sentence(point)
            subject = _derive_subject(statement)
            subject_clean = subject.rstrip(".,")
            if subject_clean:
                front = f"What does the summary emphasize about {subject_clean}?"
            else:
                front = "What key idea does the summary emphasize?"
            front = _normalize_question(front)
            skill = _slugify_skill(subject_clean or "summary_insight")
            difficulty = _infer_difficulty(statement, statement)
            yield QuizCard(
                type="short",
                front=front,
                answer=statement,
                explanation=statement,
                sources=DEFAULT_SOURCES[1:2],
                skills=[skill],
                difficulty=difficulty,
            )

    return generator()


def _take_first(generator: Iterator[QuizCard], count: int) -> List[QuizCard]:
    if count <= 0:
        return []
    items = list(itertools.islice(generator, count))
    return items


def build_quiz_items(
    request: LearnPrepareRequest, markdown: str, repository: InMemoryRepository
) -> List[QuizCard]:
    """Generate quiz items by analysing the synthesized markdown notes."""

    per_type = max(request.per_type, 1)
    items: List[QuizCard] = []
    key_facts = _derive_key_facts(markdown)

    if "mcq" in request.quiz_types:
        mcq_generator = _mcq_items_from_notes(key_facts, repository)
        items.extend(_take_first(mcq_generator, per_type))

    if "cloze" in request.quiz_types:
        cloze_generator = _cloze_items_from_notes(markdown)
        items.extend(_take_first(cloze_generator, per_type))

    if "short" in request.quiz_types:
        short_generator = _short_answer_items_from_notes(markdown)
        items.extend(_take_first(short_generator, per_type))

    return items
    per_type = max(request.per_type, 1)
    if per_type == 1:
        return items

    expanded: List[QuizCard] = []
    for _ in range(per_type):
        for base in items:
            clone = base.copy(deep=True)
            clone.id = uuid4()
            if clone.options:
                for option in clone.options:
                    option.id = uuid4()
            expanded.append(clone)
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
        METRICS.record_generation_attempt()
        note_id = uuid4()
        try:
            markdown, sources = build_markdown_summary(request)
            validate_note(markdown, sources)
            items = build_quiz_items(request)
            validate_quiz_cards(items)
        except Exception as exc:
            METRICS.record_generation_failure(exc.__class__.__name__)
            raise

        self._repository.save_note(note_id, markdown)
        items = build_quiz_items(request, markdown, self._repository)
        self._repository.save_cards(items)
        METRICS.record_generation_success(len(items))
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
                    card, user_state, session_seed=hash((card.id, user_id, due_at))
                )
                options = shuffled
                if answer_index is not None:
                    METRICS.record_answer_position(user_id, answer_index)
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
        # Simplified FSRS placeholder: increase interval by grade multiplier.
        multiplier = {1: 5, 2: 30, 3: 60, 4: 120}[request.grade]
        next_due = now + timedelta(minutes=multiplier)
        self._repository.update_due(request.card_id, next_due)
        lapse_count = self._repository.register_grade_outcome(request.card_id, request.grade)
        METRICS.record_fsrs_outcome(request.grade, multiplier)
        if lapse_count >= 3:
            METRICS.record_leech(str(request.card_id))
        return ReviewGradeResponse(
            card_id=request.card_id,
            next_due_at=next_due,
            state={"interval_minutes": multiplier},
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
