"""Core services implementing the learning and review workflows."""
from __future__ import annotations

import asyncio
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Deque, Dict, Iterable, List, MutableMapping, Optional, Tuple
from uuid import UUID, uuid4

from .models import (
    LearnPrepareArtifacts,
    LearnPrepareJobStatus,
    LearnPrepareJobStatusResponse,
    LearnPrepareMilestone,
    LearnPrepareMilestoneStatus,
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


PIPELINE_STAGES = ["ingestion", "summarization", "quiz_synthesis", "rendering"]


def _initial_milestones() -> Dict[str, LearnPrepareMilestoneStatus]:
    return {stage: LearnPrepareMilestoneStatus.PENDING for stage in PIPELINE_STAGES}


@dataclass
class JobEvent:
    """Message emitted to streaming clients about job progress."""

    type: str
    payload: Dict[str, Any]
    final: bool = False


@dataclass
class PipelineJob:
    """In-memory representation of a learning pipeline execution."""

    id: UUID
    status: LearnPrepareJobStatus
    milestones: Dict[str, LearnPrepareMilestoneStatus] = field(default_factory=_initial_milestones)
    artifacts: Optional[LearnPrepareArtifacts] = None
    error: Optional[str] = None
    history: List[JobEvent] = field(default_factory=list)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None


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
    """Facade exposing the asynchronous learning pipeline."""

    def __init__(self, repository: InMemoryRepository) -> None:
        self._repository = repository
        self._jobs: Dict[UUID, PipelineJob] = {}
        self._lock = asyncio.Lock()

    async def enqueue_prepare(self, request: LearnPrepareRequest) -> LearnPrepareResponse:
        """Schedule the learning pipeline and return the created job."""

        job_id = uuid4()
        job = PipelineJob(id=job_id, status=LearnPrepareJobStatus.QUEUED)
        async with self._lock:
            self._jobs[job_id] = job
        await self._emit(job, "status", {"status": job.status})
        task = asyncio.create_task(self._run_pipeline(job, request))
        async with self._lock:
            job.task = task
        return LearnPrepareResponse(job_id=job_id, status=job.status)

    async def get_job_status(self, job_id: UUID) -> LearnPrepareJobStatusResponse:
        """Return the status and artifacts for a prepared job."""

        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError(f"Unknown job_id: {job_id}")
            milestones_snapshot = [
                LearnPrepareMilestone(
                    name=stage,
                    status=job.milestones.get(stage, LearnPrepareMilestoneStatus.PENDING),
                )
                for stage in PIPELINE_STAGES
            ]
            artifacts = job.artifacts
            status = job.status
            error = job.error
        return LearnPrepareJobStatusResponse(
            job_id=job_id,
            status=status,
            milestones=milestones_snapshot,
            artifacts=artifacts,
            error=error,
        )

    async def iter_job_events(self, job_id: UUID) -> AsyncGenerator[JobEvent, None]:
        """Yield job events suitable for Server-Sent Events streaming."""

        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError(f"Unknown job_id: {job_id}")
            history_snapshot = list(job.history)
            finished = bool(history_snapshot and history_snapshot[-1].final)
            queue = job.queue

        for event in history_snapshot:
            yield event

        if finished:
            return

        while True:
            event = await queue.get()
            yield event
            if event.final:
                return

    async def _run_pipeline(self, job: PipelineJob, request: LearnPrepareRequest) -> None:
        note_id = uuid4()
        try:
            await self._update_status(job, LearnPrepareJobStatus.RUNNING)
            await self._run_stage(job, "ingestion", self._stage_ingestion, request)
            markdown, sources = await self._run_stage(job, "summarization", self._stage_summarization, request)
            items = await self._run_stage(job, "quiz_synthesis", self._stage_quiz_synthesis, request)
            await self._run_stage(
                job,
                "rendering",
                self._stage_rendering,
                note_id,
                markdown,
                sources,
                items,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            await self._handle_failure(job, exc)
            return

        artifacts = LearnPrepareArtifacts(
            note_id=note_id,
            content_md=markdown,
            sources=sources,
            items=items,
        )
        await self._set_artifacts(job, artifacts)
        await self._emit(job, "artifacts", {"note_id": note_id})
        await self._update_status(job, LearnPrepareJobStatus.COMPLETED)
        await self._emit(job, "complete", {"status": LearnPrepareJobStatus.COMPLETED}, final=True)

    async def _run_stage(self, job: PipelineJob, name: str, func, *args) -> Any:
        await self._set_milestone(job, name, LearnPrepareMilestoneStatus.STARTED)
        try:
            result = await func(*args)
        except Exception:
            await self._set_milestone(job, name, LearnPrepareMilestoneStatus.FAILED)
            raise
        await self._set_milestone(job, name, LearnPrepareMilestoneStatus.COMPLETED)
        return result

    async def _stage_ingestion(self, request: LearnPrepareRequest) -> None:
        await asyncio.sleep(0)

    async def _stage_summarization(self, request: LearnPrepareRequest) -> Tuple[str, List[str]]:
        return build_markdown_summary(request)

    async def _stage_quiz_synthesis(self, request: LearnPrepareRequest) -> List[QuizCard]:
        return build_quiz_items(request)

    async def _stage_rendering(
        self,
        note_id: UUID,
        markdown: str,
        sources: List[str],
        items: List[QuizCard],
    ) -> None:
        self._repository.save_note(note_id, markdown)
        self._repository.save_cards(items)

    async def _update_status(self, job: PipelineJob, status: LearnPrepareJobStatus) -> None:
        async with self._lock:
            job.status = status
        await self._emit(job, "status", {"status": status})

    async def _set_milestone(
        self, job: PipelineJob, name: str, status: LearnPrepareMilestoneStatus
    ) -> None:
        async with self._lock:
            job.milestones[name] = status
        await self._emit(job, "milestone", {"name": name, "status": status})

    async def _set_artifacts(self, job: PipelineJob, artifacts: LearnPrepareArtifacts) -> None:
        async with self._lock:
            job.artifacts = artifacts

    async def _handle_failure(self, job: PipelineJob, exc: Exception) -> None:
        message = str(exc)
        async with self._lock:
            job.error = message
        await self._emit(job, "error", {"message": message})
        await self._update_status(job, LearnPrepareJobStatus.FAILED)
        await self._emit(job, "complete", {"status": LearnPrepareJobStatus.FAILED}, final=True)

    async def _emit(self, job: PipelineJob, event_type: str, payload: Dict[str, Any], final: bool = False) -> None:
        normalized = self._normalize_payload(payload)
        event = JobEvent(type=event_type, payload=normalized, final=final)
        async with self._lock:
            job.history.append(event)
        await job.queue.put(event)

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, Enum):
                normalized[key] = value.value
            elif isinstance(value, UUID):
                normalized[key] = str(value)
            else:
                normalized[key] = value
        return normalized


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
