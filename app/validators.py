"""Validation utilities for generated content prior to persistence."""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from .models import QuizCard, QuizOption


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
FORBIDDEN_PATTERNS = (
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bTBD\b", re.IGNORECASE),
    re.compile(r"lorem ipsum", re.IGNORECASE),
    re.compile(r"\?{3,}"),
)
UNIT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(seconds?|minutes?|hours?|days?)", re.IGNORECASE)
REQUIRED_SECTIONS = ("# ", "## TL;DR", "## Key Points")


class ValidationError(ValueError):
    """Raised when generated artefacts fail validation."""


def _assert_required_sections(markdown: str) -> None:
    for section in REQUIRED_SECTIONS:
        if section not in markdown:
            raise ValidationError(f"Missing required section '{section.strip()}' in generated note")


def _assert_forbidden_patterns(text: str, context: str) -> None:
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(text):
            raise ValidationError(f"Forbidden pattern detected in {context}: '{pattern.pattern}'")


def _assert_unit_consistency(text: str, context: str) -> None:
    for value_str, unit in UNIT_PATTERN.findall(text):
        try:
            value = float(value_str)
        except ValueError:
            raise ValidationError(f"Invalid numeric value '{value_str}' in {context}") from None
        unit_normalized = unit.lower()
        is_plural = unit_normalized.endswith("s")
        if abs(value - 1.0) < 1e-9 and is_plural:
            raise ValidationError(f"Unit '{unit}' should be singular for value {value_str} in {context}")
        if abs(value - 1.0) >= 1e-9 and not is_plural:
            raise ValidationError(f"Unit '{unit}' should be plural for value {value_str} in {context}")


def _assert_citations(markdown: str, sources: Sequence[str]) -> None:
    citations = {int(match) for match in CITATION_PATTERN.findall(markdown)}
    if not citations:
        raise ValidationError("Generated note must contain at least one citation reference")
    expected = set(range(1, max(citations) + 1))
    if citations != expected:
        raise ValidationError("Citations must be sequential starting from [1]")
    if len(sources) < max(citations):
        raise ValidationError("Number of sources does not match citation references")


def validate_note(markdown: str, sources: Sequence[str]) -> None:
    """Validate the generated markdown note content before persistence."""

    if not markdown.strip():
        raise ValidationError("Generated note markdown is empty")
    if not sources:
        raise ValidationError("Generated notes must include at least one source")

    _assert_required_sections(markdown)
    _assert_citations(markdown, sources)
    _assert_forbidden_patterns(markdown, "note")
    _assert_unit_consistency(markdown, "note")


def _validate_option(option: QuizOption) -> None:
    if not option.text.strip():
        raise ValidationError("Quiz options must provide non-empty text")
    _assert_forbidden_patterns(option.text, "quiz option")
    _assert_unit_consistency(option.text, "quiz option")


def validate_quiz_cards(cards: Iterable[QuizCard]) -> None:
    """Validate generated quiz cards for structural and content quality."""

    seen_ids = set()
    for card in cards:
        if card.id in seen_ids:
            raise ValidationError(f"Duplicate quiz card identifier detected: {card.id}")
        seen_ids.add(card.id)

        if not card.front.strip():
            raise ValidationError("Quiz card fronts must be non-empty")
        if not card.answer.strip():
            raise ValidationError("Quiz card answers must be non-empty")
        if not card.explanation.strip():
            raise ValidationError("Quiz card explanations must be non-empty")
        if not card.sources:
            raise ValidationError("Quiz cards must reference at least one source")

        _assert_forbidden_patterns(card.front, "quiz front")
        _assert_forbidden_patterns(card.answer, "quiz answer")
        _assert_forbidden_patterns(card.explanation, "quiz explanation")

        _assert_unit_consistency(card.front, "quiz front")
        _assert_unit_consistency(card.answer, "quiz answer")
        _assert_unit_consistency(card.explanation, "quiz explanation")

        if card.type == "mcq":
            if not card.options:
                raise ValidationError("MCQ cards must define options")
            for option in card.options:
                _validate_option(option)
            option_texts = {opt.text for opt in card.options}
            if card.answer not in option_texts:
                raise ValidationError("MCQ answers must match one of the provided options")
        else:
            if card.options:
                raise ValidationError("Only MCQ cards may include options")

        for source in card.sources:
            if not source.strip():
                raise ValidationError("Quiz card sources must be non-empty strings")

__all__ = ["ValidationError", "validate_note", "validate_quiz_cards"]
