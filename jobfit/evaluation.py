from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .alignment import (
    LinePredictionMap,
    line_classification,
    line_predictions,
    line_verdict,
)


def _load_reports(paths: list[str]) -> list[tuple[str, dict[str, Any]]]:
    loaded = []
    for raw_path in paths:
        path = Path(raw_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        run_id = report.get("run_id") or path.parent.name
        loaded.append((run_id, report))
    return loaded


def _read_csv(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ratio(values: list[bool]) -> float | None:
    return None if not values else round(sum(values) / len(values), 4)


def _score_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return round(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)), 3)


def _evaluate_evidence(
    reports: list[tuple[str, dict[str, Any]]],
    evidence_labels: list[dict[str, str]],
) -> dict[str, Any]:
    report_index: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for run_id, report in reports:
        posting = report["posting_fingerprint"]
        for item in report.get("requirements", []):
            judgment = item.get("judgment") or {}
            report_index[(run_id, posting, item["requirement_key"])] = tuple(
                sorted(judgment.get("resume_line_ids", []))
            )

    matched_values: list[bool] = []
    mismatched_evidence_sets = 0
    for row in evidence_labels:
        key = (row["run_id"], row["posting_fingerprint"], row["requirement_key"])
        actual_ids = report_index.get(key)
        if actual_ids is None:
            continue
        labeled_ids = tuple(
            sorted(
                value.strip()
                for value in row.get("evidence_line_ids", "").split("|")
                if value.strip()
            )
        )
        if labeled_ids != actual_ids:
            mismatched_evidence_sets += 1
            continue
        raw = row.get("evidence_valid", "").strip().lower()
        if raw in {"true", "1", "yes", "y"}:
            matched_values.append(True)
        elif raw in {"false", "0", "no", "n"}:
            matched_values.append(False)

    return {
        "evidence_labels": len(evidence_labels),
        "matched_evidence_labels": len(matched_values),
        "mismatched_evidence_sets": mismatched_evidence_sets,
        "evidence_validity": _ratio(matched_values),
    }


def _evaluate_system(
    gold_rows: list[dict[str, str]],
    reports: list[tuple[str, dict[str, Any]]],
    evidence_labels: list[dict[str, str]],
) -> dict[str, Any]:
    grouped: dict[
        str, list[tuple[str, dict[str, Any], LinePredictionMap]]
    ] = defaultdict(list)
    for run_id, report in reports:
        grouped[report["posting_fingerprint"]].append(
            (run_id, report, line_predictions(report))
        )

    representative = {
        posting: runs[0][2] for posting, runs in grouped.items() if runs
    }
    classification_checks: list[bool] = []
    line_coverage_checks: list[bool] = []
    requirement_recall_checks: list[bool] = []
    false_exclusions: list[bool] = []
    verdict_checks: list[bool] = []
    joint_checks: list[bool] = []

    labeled_lines_by_posting: dict[str, set[str]] = defaultdict(set)
    for row in gold_rows:
        posting = row["posting_fingerprint"]
        line_id = row["posting_line_id"]
        gold_class = row["gold_class"]
        gold_verdict = (row.get("gold_verdict") or "").strip() or None
        labeled_lines_by_posting[posting].add(line_id)
        predicted = representative.get(posting, {}).get(line_id)
        present = predicted is not None
        line_coverage_checks.append(present)
        predicted_class = line_classification(predicted)
        class_correct = predicted_class == gold_class
        classification_checks.append(class_correct)

        is_gold_requirement = gold_class in {"required", "preferred"}
        if is_gold_requirement:
            predicted_as_requirement = predicted_class in {"required", "preferred"}
            requirement_recall_checks.append(predicted_as_requirement)
            false_exclusions.append(predicted_class == "non_requirement")
            if gold_verdict is not None:
                verdict_correct = present and line_verdict(predicted) == gold_verdict
                verdict_checks.append(verdict_correct)
                joint_checks.append(class_correct and verdict_correct)
        else:
            joint_checks.append(class_correct)

    classification_stability: list[bool] = []
    verdict_stability: list[bool] = []
    grouping_stability: list[bool] = []
    all_runs_line_presence: list[bool] = []
    score_stds: dict[str, float] = {}

    for posting, runs in grouped.items():
        if len(runs) < 2:
            continue
        line_ids = set(labeled_lines_by_posting.get(posting, set()))
        for _, _, prediction in runs:
            line_ids.update(prediction)
        for line_id in line_ids:
            entries = [prediction.get(line_id) for _, _, prediction in runs]
            all_runs_line_presence.append(all(entry is not None for entry in entries))
            classifications = [
                (("missing",) if entry is None else tuple(
                    item["classification"] for item in entry
                ))
                for entry in entries
            ]
            verdicts = [
                (("missing", None),)
                if entry is None
                else tuple(
                    (
                        item["requirement_key"],
                        item["classification"],
                        item["verdict"],
                    )
                    for item in entry
                )
                for entry in entries
            ]
            groups = [
                (("missing",),)
                if entry is None
                else tuple(
                    (item["classification"], *item["group"])
                    for item in entry
                )
                for entry in entries
            ]
            classification_stability.append(len(set(classifications)) == 1)
            verdict_stability.append(len(set(verdicts)) == 1)
            grouping_stability.append(len(set(groups)) == 1)

        scores = [
            float(report["score"]["overall_fit_percent"])
            for _, report, _ in runs
            if report.get("score")
            and report["score"].get("overall_fit_percent") is not None
        ]
        std = _score_std(scores)
        if std is not None:
            score_stds[posting] = std

    evidence_result = _evaluate_evidence(reports, evidence_labels)
    complete_runs = [report.get("status") == "complete" for _, report in reports]
    return {
        "postings_evaluated": len(grouped),
        "total_runs": len(reports),
        "complete_run_rate": _ratio(complete_runs),
        "labeled_posting_lines": len(gold_rows),
        "labeled_line_prediction_coverage": _ratio(line_coverage_checks),
        "line_classification_accuracy": _ratio(classification_checks),
        "requirement_line_recall": _ratio(requirement_recall_checks),
        "gold_requirement_false_exclusion_rate": _ratio(false_exclusions),
        "verdict_accuracy_including_missing": _ratio(verdict_checks),
        "joint_line_and_verdict_accuracy": _ratio(joint_checks),
        "repeated_postings": sum(len(runs) >= 2 for runs in grouped.values()),
        "all_runs_line_presence": _ratio(all_runs_line_presence),
        "line_classification_stability": _ratio(classification_stability),
        "line_verdict_stability": _ratio(verdict_stability),
        "requirement_grouping_stability": _ratio(grouping_stability),
        "fit_score_std_by_posting": score_stds,
        **evidence_result,
    }


def _delta(agent: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    left, right = agent.get(key), baseline.get(key)
    if left is None or right is None:
        return None
    return round(left - right, 4)


def evaluate(
    labels_path: str,
    report_paths: list[str],
    *,
    evidence_labels_path: str | None = None,
    baseline_report_paths: list[str] | None = None,
) -> dict[str, Any]:
    gold_rows = _read_csv(labels_path)
    evidence_labels = _read_csv(evidence_labels_path)
    agent = _evaluate_system(gold_rows, _load_reports(report_paths), evidence_labels)
    result: dict[str, Any] = {"agent": agent}

    if baseline_report_paths:
        baseline = _evaluate_system(
            gold_rows, _load_reports(baseline_report_paths), evidence_labels
        )
        result["one_shot_baseline"] = baseline
        result["agent_minus_baseline"] = {
            key: _delta(agent, baseline, key)
            for key in [
                "complete_run_rate",
                "line_classification_accuracy",
                "requirement_line_recall",
                "verdict_accuracy_including_missing",
                "joint_line_and_verdict_accuracy",
            ]
        }
    return result
