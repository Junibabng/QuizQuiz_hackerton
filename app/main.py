"""FastAPI application exposing the QuizQuiz Hackerton endpoints."""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException

from .models import (
    LearnPrepareRequest,
    LearnPrepareResponse,
    ReviewDueResponse,
    ReviewGradeRequest,
    ReviewGradeResponse,
)
from .services import BiasConfig, InMemoryRepository, LearningService, ReviewService

app = FastAPI(title="QuizQuiz Hackerton", version="0.1.0")


def get_repository() -> InMemoryRepository:
    return app.state.repository


def get_learning_service(repository: InMemoryRepository = Depends(get_repository)) -> LearningService:
    return LearningService(repository)


def get_review_service(repository: InMemoryRepository = Depends(get_repository)) -> ReviewService:
    return ReviewService(repository, bias_config=app.state.bias_config)


@app.on_event("startup")
def startup() -> None:
    app.state.repository = InMemoryRepository()
    position_bias_threshold = float(os.getenv("QUIZQUIZ_POSITION_BIAS_THRESHOLD", "0.35"))
    option_reuse_threshold = float(os.getenv("QUIZQUIZ_OPTION_REUSE_THRESHOLD", "0.65"))
    app.state.bias_config = BiasConfig(
        position_bias_threshold=position_bias_threshold,
        option_reuse_threshold=option_reuse_threshold,
    )


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
    return service.grade(request)


__all__ = ["app"]
