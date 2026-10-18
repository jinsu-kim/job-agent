from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any


LinePrediction = dict[str, Any]
LinePredictionMap = dict[str, tuple[LinePrediction, ...]]


def line_predictions(report: dict[str, Any]) -> LinePredictionMap:
    """Group every atomic requirement prediction by immutable posting line ID."""
    grouped: dict[str, list[LinePrediction]] = defaultdict(list)

    for item in report.get("requirements", []):
        judgment = item.get("judgment") or {}
        posting_line_ids = tuple(
            sorted(dict.fromkeys(item.get("posting_line_ids", [])))
        )
        prediction = {
            "classification": item.get("requirement_type"),
            "verdict": judgment.get("verdict"),
            "requirement_key": item.get("requirement_key"),
            "text": item.get("text"),
            "group": posting_line_ids,
            "evidence_line_ids": tuple(
                sorted(judgment.get("resume_line_ids", []))
            ),
        }
        for line_id in posting_line_ids:
            grouped[line_id].append(prediction)

    for item in report.get("excluded_posting_lines", []):
        line_id = item["line_id"]
        grouped[line_id].append(
            {
                "classification": "non_requirement",
                "verdict": None,
                "requirement_key": None,
                "text": None,
                "group": (line_id,),
                "evidence_line_ids": (),
            }
        )

    return {
        line_id: tuple(sorted(predictions, key=_prediction_sort_key))
        for line_id, predictions in grouped.items()
    }


def requirements_by_key(
    predictions: LinePredictionMap,
) -> dict[str, LinePrediction]:
    """Flatten aligned lines while keeping each atomic requirement only once."""
    result: dict[str, LinePrediction] = {}
    for line_items in predictions.values():
        for item in line_items:
            key = item.get("requirement_key")
            if key is not None:
                result[key] = item
    return result


def line_classification(
    predictions: Sequence[LinePrediction] | None,
) -> str | None:
    """Return one line-level class, or ``mixed`` when atomic classes disagree."""
    if not predictions:
        return None
    classes = {item["classification"] for item in predictions}
    return next(iter(classes)) if len(classes) == 1 else "mixed"


def line_verdict(predictions: Sequence[LinePrediction] | None) -> str | None:
    """Aggregate atomic verdicts for compatibility with line-level gold labels."""
    if not predictions:
        return None
    verdicts = {
        item["verdict"]
        for item in predictions
        if item["classification"] in {"required", "preferred"}
    }
    if not verdicts or None in verdicts:
        return None
    if len(verdicts) == 1:
        return next(iter(verdicts))
    return "partial"


def _prediction_sort_key(prediction: LinePrediction) -> tuple[str, str, str]:
    return (
        prediction.get("classification") or "",
        prediction.get("requirement_key") or "",
        prediction.get("text") or "",
    )
