from __future__ import annotations

from typing import Any

from .models import Document, Judgment, Requirement, RequirementType, Verdict


VERDICT_POINTS = {
    Verdict.SATISFIED: 1.0,
    Verdict.PARTIAL: 0.5,
    Verdict.UNSATISFIED: 0.0,
}


def calculate_score(
    requirements: dict[str, Requirement],
    judgments: dict[str, Judgment],
) -> dict[str, Any] | None:
    """Convert completed judgments into the public fit score."""
    if not requirements:
        return None

    scores: dict[RequirementType, list[float]] = {
        RequirementType.REQUIRED: [],
        RequirementType.PREFERRED: [],
    }
    for requirement_id, requirement in requirements.items():
        judgment = judgments.get(requirement_id)
        if judgment is None:
            return None
        scores[requirement.requirement_type].append(VERDICT_POINTS[judgment.verdict])

    required = _average(scores[RequirementType.REQUIRED])
    preferred = _average(scores[RequirementType.PREFERRED])

    if required is not None and preferred is not None:
        overall = required * 0.8 + preferred * 0.2
    else:
        overall = required if required is not None else preferred
    if overall is None:
        return None

    return {
        "required_percent": _percent(required),
        "preferred_percent": _percent(preferred),
        "overall_fit_percent": round(overall * 100, 1),
        "formula": "required*0.8 + preferred*0.2; satisfied=1, partial=0.5, unsatisfied=0",
    }


def build_report(
    *,
    posting: Document,
    resume: Document,
    requirements: dict[str, Requirement],
    judgments: dict[str, Judgment],
    searches: dict[str, list[dict[str, Any]]],
    excluded_lines: dict[str, str],
    status: str,
    incomplete_reason: str | None,
) -> dict[str, Any]:
    covered_lines = {
        line_id
        for requirement in requirements.values()
        for line_id in requirement.posting_line_ids
    } | set(excluded_lines)

    return {
        "status": status,
        "incomplete_reason": incomplete_reason,
        "posting_fingerprint": posting.fingerprint,
        "resume_fingerprint": resume.fingerprint,
        "posting_source": posting.source,
        "resume_source": resume.source,
        "requirements": [
            _requirement_result(requirement, posting, resume, judgments, searches)
            for requirement in requirements.values()
        ],
        "excluded_posting_lines": [
            {**posting.by_id[line_id].to_dict(), "reason": reason}
            for line_id, reason in excluded_lines.items()
        ],
        "coverage": {
            "posting_lines_percent": round(
                len(covered_lines) / max(1, len(posting.lines)) * 100, 1
            ),
            "judgments_percent": round(
                len(judgments) / max(1, len(requirements)) * 100, 1
            ),
        },
        "score": calculate_score(requirements, judgments) if status == "complete" else None,
        "limits": {
            "guaranteed": [
                "공고와 이력서의 인용 줄 ID가 실제 문서에 존재함",
                "등록된 모든 요건에 판정이 기록된 뒤에만 완료됨",
                "판정에서 점수로의 변환이 규칙 기반임",
            ],
            "not_guaranteed": [
                "공고 줄의 요건/비요건 분류가 의미적으로 정확함",
                "이력서 인용 줄이 해당 요건을 실제로 뒷받침함",
                "LLM의 충족 여부 판정이 정확하거나 실행 간 동일함",
            ],
        },
    }


def _requirement_result(
    requirement: Requirement,
    posting: Document,
    resume: Document,
    judgments: dict[str, Judgment],
    searches: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    judgment = judgments.get(requirement.requirement_id)
    return {
        **requirement.to_dict(),
        "posting_evidence": [
            posting.by_id[line_id].to_dict()
            for line_id in requirement.posting_line_ids
        ],
        "judgment": judgment.to_dict() if judgment else None,
        "resume_evidence": [
            resume.by_id[line_id].to_dict()
            for line_id in (judgment.resume_line_ids if judgment else [])
        ],
        "resume_searches": searches.get(requirement.requirement_id, []),
    }


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percent(value: float | None) -> float | None:
    return round(value * 100, 1) if value is not None else None
