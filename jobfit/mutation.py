from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import OpenAICompatibleClient, run_agent
from .documents import remove_document_lines
from .models import Document
from .state import FitState


VERDICT_VALUE = {
    "unsatisfied": 0.0,
    "partial": 0.5,
    "satisfied": 1.0,
}


def _by_key(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["requirement_key"]: item for item in report.get("requirements", [])}


def run_removal_mutations(
    *,
    posting: Document,
    resume: Document,
    baseline_report: dict[str, Any],
    client: OpenAICompatibleClient,
    output_dir: Path,
    max_tool_calls: int,
    max_turns: int,
    limit: int | None = None,
) -> dict[str, Any]:
    """Remove each positive judgment's cited evidence and rerun the full agent."""
    if baseline_report.get("posting_fingerprint") != posting.fingerprint:
        raise ValueError("baseline 보고서와 입력 공고의 fingerprint가 다릅니다.")
    if baseline_report.get("resume_fingerprint") != resume.fingerprint:
        raise ValueError("baseline 보고서와 입력 이력서의 fingerprint가 다릅니다.")
    candidates = [
        item
        for item in baseline_report.get("requirements", [])
        if item.get("judgment", {}).get("verdict") in {"satisfied", "partial"}
        and item.get("judgment", {}).get("resume_line_ids")
    ]
    if limit is not None:
        candidates = candidates[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    for index, target in enumerate(candidates, start=1):
        removed_ids = set(target["judgment"]["resume_line_ids"])
        mutated_resume = remove_document_lines(resume, removed_ids)
        state = FitState(posting, mutated_resume, max_tool_calls=max_tool_calls)
        report = run_agent(state, client, max_turns=max_turns)
        report["system"] = "evidence_removal_mutation"
        report["mutation"] = {
            "target_requirement_key": target["requirement_key"],
            "removed_resume_line_ids": sorted(removed_ids),
        }

        case_dir = output_dir / f"case-{index:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        report["run_id"] = f"{output_dir.name}/case-{index:03d}"
        (case_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (case_dir / "trace.json").write_text(
            json.dumps(state.trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        result_items = _by_key(report)
        result_target = result_items.get(target["requirement_key"])
        baseline_value = VERDICT_VALUE[target["judgment"]["verdict"]]
        result_verdict = (
            None if result_target is None or result_target.get("judgment") is None
            else result_target["judgment"]["verdict"]
        )
        result_value = VERDICT_VALUE.get(result_verdict)

        baseline_other = {
            key: item.get("judgment", {}).get("verdict")
            for key, item in _by_key(baseline_report).items()
            if key != target["requirement_key"]
        }
        result_other = {
            key: item.get("judgment", {}).get("verdict")
            for key, item in result_items.items()
            if key != target["requirement_key"]
        }
        common_other = set(baseline_other) & set(result_other)
        unrelated_flips = sum(
            baseline_other[key] != result_other[key] for key in common_other
        )

        cases.append(
            {
                "case": index,
                "target_requirement_key": target["requirement_key"],
                "removed_resume_line_ids": sorted(removed_ids),
                "baseline_verdict": target["judgment"]["verdict"],
                "mutated_verdict": result_verdict,
                "target_found": result_target is not None,
                "verdict_nonincrease": result_value is not None and result_value <= baseline_value,
                "verdict_decreased": result_value is not None and result_value < baseline_value,
                "unrelated_common_requirements": len(common_other),
                "unrelated_verdict_flips": unrelated_flips,
                "report_path": str((case_dir / "report.json").resolve()),
            }
        )

    found = [case for case in cases if case["target_found"]]
    unrelated_total = sum(case["unrelated_common_requirements"] for case in cases)
    summary = {
        "evaluation": "evidence_removal_mutation",
        "attempted_cases": len(cases),
        "target_match_rate": None
        if not cases
        else round(sum(case["target_found"] for case in cases) / len(cases), 4),
        "verdict_nonincrease_rate": None
        if not found
        else round(sum(case["verdict_nonincrease"] for case in found) / len(found), 4),
        "verdict_decrease_rate": None
        if not found
        else round(sum(case["verdict_decreased"] for case in found) / len(found), 4),
        "unrelated_flip_rate": None
        if not unrelated_total
        else round(
            sum(case["unrelated_verdict_flips"] for case in cases) / unrelated_total,
            4,
        ),
        "cases": cases,
        "interpretation": (
            "인용 근거를 제거했을 때 대상 판정이 상승하지 않는지 측정한다. "
            "target_match_rate가 낮으면 요건 추출 자체가 불안정하므로 성공으로 간주하지 않는다."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
