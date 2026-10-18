import csv
import json
import tempfile
import unittest
from pathlib import Path

from jobfit.evaluation import evaluate


def report(run_id, posting, requirement_type, verdict, score, *, excluded=False):
    requirements = []
    excluded_lines = [{"line_id": "P002", "reason": "제목"}]
    if excluded:
        excluded_lines.append({"line_id": "P001", "reason": "오분류"})
    else:
        requirements.append(
            {
                "requirement_key": f"{requirement_type}-P001",
                "requirement_type": requirement_type,
                "posting_line_ids": ["P001"],
                "judgment": {"verdict": verdict, "resume_line_ids": ["R001"]},
            }
        )
    return {
        "run_id": run_id,
        "status": "complete",
        "posting_fingerprint": posting,
        "requirements": requirements,
        "excluded_posting_lines": excluded_lines,
        "score": {"overall_fit_percent": score},
    }


class EvaluationTests(unittest.TestCase):
    def _write_csv(self, path, fieldnames, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_line_aligned_metrics_penalize_omission_and_capture_instability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.csv"
            self._write_csv(
                labels,
                ["posting_fingerprint", "posting_line_id", "gold_class", "gold_verdict"],
                [
                    {
                        "posting_fingerprint": "p1",
                        "posting_line_id": "P001",
                        "gold_class": "required",
                        "gold_verdict": "satisfied",
                    },
                    {
                        "posting_fingerprint": "p1",
                        "posting_line_id": "P002",
                        "gold_class": "non_requirement",
                        "gold_verdict": "",
                    },
                    {
                        "posting_fingerprint": "p2",
                        "posting_line_id": "P001",
                        "gold_class": "preferred",
                        "gold_verdict": "partial",
                    },
                ],
            )

            values = [
                report("p1-run1", "p1", "required", "satisfied", 80),
                report("p1-run2", "p1", "preferred", "satisfied", 82),
                report("p2-run1", "p2", "preferred", "partial", 50, excluded=True),
            ]
            paths = []
            for index, value in enumerate(values):
                path = root / f"report{index}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths.append(str(path))

            result = evaluate(str(labels), paths)["agent"]
            self.assertEqual(result["postings_evaluated"], 2)
            self.assertEqual(result["requirement_line_recall"], 0.5)
            self.assertEqual(result["gold_requirement_false_exclusion_rate"], 0.5)
            self.assertLess(result["line_classification_stability"], 1.0)
            self.assertEqual(result["fit_score_std_by_posting"]["p1"], 1.0)

    def test_evidence_labels_are_bound_to_run_and_exact_evidence_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.csv"
            evidence = root / "evidence.csv"
            self._write_csv(
                labels,
                ["posting_fingerprint", "posting_line_id", "gold_class", "gold_verdict"],
                [
                    {
                        "posting_fingerprint": "p1",
                        "posting_line_id": "P001",
                        "gold_class": "required",
                        "gold_verdict": "satisfied",
                    },
                    {
                        "posting_fingerprint": "p1",
                        "posting_line_id": "P002",
                        "gold_class": "non_requirement",
                        "gold_verdict": "",
                    },
                ],
            )
            self._write_csv(
                evidence,
                [
                    "run_id",
                    "posting_fingerprint",
                    "requirement_key",
                    "evidence_line_ids",
                    "evidence_valid",
                ],
                [
                    {
                        "run_id": "run1",
                        "posting_fingerprint": "p1",
                        "requirement_key": "required-P001",
                        "evidence_line_ids": "R001",
                        "evidence_valid": "true",
                    },
                    {
                        "run_id": "run1",
                        "posting_fingerprint": "p1",
                        "requirement_key": "required-P001",
                        "evidence_line_ids": "R999",
                        "evidence_valid": "true",
                    },
                ],
            )
            path = root / "report.json"
            path.write_text(
                json.dumps(report("run1", "p1", "required", "satisfied", 100)),
                encoding="utf-8",
            )
            result = evaluate(
                str(labels), [str(path)], evidence_labels_path=str(evidence)
            )["agent"]
            self.assertEqual(result["matched_evidence_labels"], 1)
            self.assertEqual(result["mismatched_evidence_sets"], 1)
            self.assertEqual(result["evidence_validity"], 1.0)

    def test_stability_keeps_all_requirements_from_the_same_posting_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.csv"
            self._write_csv(
                labels,
                ["posting_fingerprint", "posting_line_id", "gold_class", "gold_verdict"],
                [
                    {
                        "posting_fingerprint": "p1",
                        "posting_line_id": "P001",
                        "gold_class": "required",
                        "gold_verdict": "partial",
                    }
                ],
            )

            reports = []
            for run_id, python_verdict in [
                ("run1", "satisfied"),
                ("run2", "partial"),
            ]:
                reports.append(
                    {
                        "run_id": run_id,
                        "status": "complete",
                        "posting_fingerprint": "p1",
                        "requirements": [
                            {
                                "requirement_key": "python",
                                "requirement_type": "required",
                                "text": "Python 경험",
                                "posting_line_ids": ["P001"],
                                "judgment": {
                                    "verdict": python_verdict,
                                    "resume_line_ids": ["R001"],
                                },
                            },
                            {
                                "requirement_key": "pytorch",
                                "requirement_type": "required",
                                "text": "PyTorch 경험",
                                "posting_line_ids": ["P001"],
                                "judgment": {
                                    "verdict": "unsatisfied",
                                    "resume_line_ids": [],
                                },
                            },
                        ],
                        "excluded_posting_lines": [],
                        "score": {"overall_fit_percent": 50},
                    }
                )

            paths = []
            for index, value in enumerate(reports):
                path = root / f"multi-{index}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths.append(str(path))

            result = evaluate(str(labels), paths)["agent"]

            self.assertEqual(result["line_classification_stability"], 1.0)
            self.assertEqual(result["requirement_grouping_stability"], 1.0)
            self.assertEqual(result["line_verdict_stability"], 0.0)


if __name__ == "__main__":
    unittest.main()
