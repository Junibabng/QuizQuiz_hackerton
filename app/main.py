"""FastAPI application exposing the QuizQuiz Hackerton endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from .models import (
    LearnPrepareRequest,
    LearnPrepareResponse,
    ReviewDueResponse,
    ReviewGradeRequest,
    ReviewGradeResponse,
)
from .services import LearningService, ReviewService
from .storage import LocalNoteRepository, SqliteStudyRepository

app = FastAPI(title="QuizQuiz Hackerton", version="0.1.0")


def get_learning_service() -> LearningService:
    return LearningService(app.state.note_repository, app.state.study_repository)


def get_review_service() -> ReviewService:
    return ReviewService(
        app.state.study_repository,
        app.state.study_repository,
        app.state.study_repository,
    )


@app.on_event("startup")
def startup() -> None:
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


@app.post("/v1/learn/prepare", response_model=LearnPrepareResponse)
def learn_prepare(
    request: LearnPrepareRequest, service: LearningService = Depends(get_learning_service)
) -> LearnPrepareResponse:
    try:
        return service.prepare(request)
    except ValueError as exc:  # validation from service level
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
