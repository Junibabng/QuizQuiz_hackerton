"""Simple in-process metrics registry for service instrumentation."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List


@dataclass
class MetricsRegistry:
    """Holds counters and histograms exposed by the application."""

    generation_attempts: int = 0
    generation_successes: int = 0
    generation_failures: int = 0
    generation_failure_reasons: Counter = field(default_factory=Counter)
    generated_item_counts: List[int] = field(default_factory=list)
    answer_position_histograms: DefaultDict[str, List[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0, 0, 0])
    )
    answer_position_bias_ratio: Dict[str, float] = field(default_factory=dict)
    fsrs_outcomes: Counter = field(default_factory=Counter)
    fsrs_interval_buckets: Counter = field(default_factory=Counter)
    leech_detections: Counter = field(default_factory=Counter)

    def record_generation_attempt(self) -> None:
        self.generation_attempts += 1

    def record_generation_success(self, item_count: int) -> None:
        self.generation_successes += 1
        self.generated_item_counts.append(item_count)

    def record_generation_failure(self, reason: str) -> None:
        self.generation_failures += 1
        self.generation_failure_reasons[reason] += 1

    def record_answer_position(self, cohort: str, position: int) -> None:
        """Track answer placement frequency for a learner cohort."""

        histogram = self.answer_position_histograms[cohort]
        if position >= len(histogram):
            histogram.extend([0] * (position - len(histogram) + 1))
        histogram[position] += 1
        total = sum(histogram)
        if total:
            self.answer_position_bias_ratio[cohort] = max(histogram) / total

    def record_fsrs_outcome(self, grade: int, interval_minutes: int) -> None:
        self.fsrs_outcomes[(grade, interval_minutes)] += 1
        bucket_key = (grade, interval_minutes)
        self.fsrs_interval_buckets[bucket_key] += 1

    def record_leech(self, card_id: str) -> None:
        self.leech_detections[card_id] += 1

    @property
    def generation_success_rate(self) -> float:
        if self.generation_attempts == 0:
            return 0.0
        return self.generation_successes / self.generation_attempts


METRICS = MetricsRegistry()

__all__ = ["METRICS", "MetricsRegistry"]
