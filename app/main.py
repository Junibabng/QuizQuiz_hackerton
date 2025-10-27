"""FastAPI application exposing the QuizQuiz Hackerton endpoints."""
from __future__ import annotations

import json
from typing import AsyncGenerator
from uuid import UUID
from pathlib import Path
import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .models import (
    LearnPrepareJobStatusResponse,
    LearnPrepareRequest,
    LearnPrepareResponse,
    ReviewDueResponse,
    ReviewGradeRequest,
    ReviewGradeResponse,
)
from .services import LearningService, ReviewService
from .storage import LocalNoteRepository, SqliteStudyRepository
from .services import BiasConfig, InMemoryRepository, LearningService, ReviewService

app = FastAPI(title="QuizQuiz Hackerton", version="0.1.0")


def get_learning_service() -> LearningService:
    return LearningService(app.state.note_repository, app.state.study_repository)

def get_learning_service() -> LearningService:
    return app.state.learning_service

def get_review_service() -> ReviewService:
    return ReviewService(
        app.state.study_repository,
        app.state.study_repository,
        app.state.study_repository,
    )

def get_review_service() -> ReviewService:
    return app.state.review_service
def get_review_service(repository: InMemoryRepository = Depends(get_repository)) -> ReviewService:
    return ReviewService(repository, bias_config=app.state.bias_config)


@app.on_event("startup")
def startup() -> None:
    repository = InMemoryRepository()
    app.state.repository = repository
    app.state.learning_service = LearningService(repository)
    app.state.review_service = ReviewService(repository)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    notes_dir = data_dir / "notes"
    db_path = data_dir / "study.sqlite3"
    app.state.note_repository = LocalNoteRepository(notes_dir)
    app.state.study_repository = SqliteStudyRepository(db_path)


@app.on_event("shutdown")
def shutdown() -> None:
    if hasattr(app.state, "study_repository"):
        app.state.study_repository.close()
    app.state.repository = InMemoryRepository()
    position_bias_threshold = float(os.getenv("QUIZQUIZ_POSITION_BIAS_THRESHOLD", "0.35"))
    option_reuse_threshold = float(os.getenv("QUIZQUIZ_OPTION_REUSE_THRESHOLD", "0.65"))
    app.state.bias_config = BiasConfig(
        position_bias_threshold=position_bias_threshold,
        option_reuse_threshold=option_reuse_threshold,
    )


@app.post("/v1/learn/prepare", response_model=LearnPrepareResponse)
async def learn_prepare(
    request: LearnPrepareRequest, service: LearningService = Depends(get_learning_service)
) -> LearnPrepareResponse:
    try:
        return await service.enqueue_prepare(request)
    except ValueError as exc:  # validation from service level
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/learn/jobs/{job_id}", response_model=LearnPrepareJobStatusResponse)
async def learn_prepare_status(
    job_id: UUID, service: LearningService = Depends(get_learning_service)
) -> LearnPrepareJobStatusResponse:
    try:
        return await service.get_job_status(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/learn/jobs/{job_id}/events")
async def learn_prepare_events(job_id: UUID, service: LearningService = Depends(get_learning_service)) -> StreamingResponse:
    try:
        await service.get_job_status(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def event_stream() -> AsyncGenerator[str, None]:
        async for event in service.iter_job_events(job_id):
            payload = json.dumps(event.payload)
            yield f"event: {event.type}\ndata: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/review/due", response_model=ReviewDueResponse)
def review_due(user_id: str, service: ReviewService = Depends(get_review_service)) -> ReviewDueResponse:
    return service.get_due(user_id=user_id)


@app.post("/v1/review/grade", response_model=ReviewGradeResponse)
def review_grade(
    request: ReviewGradeRequest, service: ReviewService = Depends(get_review_service)
) -> ReviewGradeResponse:
    try:
        return service.grade(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["app"]
