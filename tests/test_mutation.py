import json
import tempfile
import unittest
from pathlib import Path

from jobfit.documents import parse_document, remove_document_lines
from jobfit.mutation import run_removal_mutations
from jobfit.state import FitState


def call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages):
        return self.responses.pop(0)


class MutationTests(unittest.TestCase):
    def test_removes_lines_without_renumbering(self):
        resume = parse_document("첫 줄\n둘째 줄\n셋째 줄", prefix="R", source="resume")
        mutated = remove_document_lines(resume, {"R002"})
        self.assertEqual([line.line_id for line in mutated.lines], ["R001", "R003"])

    def test_evidence_removal_measures_target_decrease(self):
        posting = parse_document("자격요건\nPython 경험", prefix="P", source="posting")
        resume = parse_document("Python 모델 개발", prefix="R", source="resume")
        baseline_state = FitState(posting, resume)
        baseline_state.exclude_posting_lines(["P001"], "섹션 제목")
        requirement = baseline_state.record_requirement("required", "Python 경험", ["P002"])
        baseline_state.search_resume(requirement["requirement_id"], "Python")
        baseline_state.record_judgment(
            requirement["requirement_id"], "satisfied", ["R001"], "직접 경험"
        )
        baseline_report = baseline_state.finalize_report()

        fake = FakeClient(
            [
                {"role": "assistant", "content": None, "tool_calls": [call("1", "read_posting", {})]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        call(
                            "2",
                            "record_requirement",
                            {
                                "requirement_type": "required",
                                "text": "Python 경험",
                                "posting_line_ids": ["P002"],
                            },
                        ),
                        call(
                            "3",
                            "exclude_posting_lines",
                            {"posting_line_ids": ["P001"], "reason": "섹션 제목"},
                        ),
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        call(
                            "4",
                            "search_resume",
                            {"requirement_id": "Q001", "query": "Python"},
                        )
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        call(
                            "5",
                            "record_judgment",
                            {
                                "requirement_id": "Q001",
                                "verdict": "unsatisfied",
                                "resume_line_ids": [],
                                "explanation": "근거 없음",
                            },
                        )
                    ],
                },
                {"role": "assistant", "content": None, "tool_calls": [call("6", "finalize_report", {})]},
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            summary = run_removal_mutations(
                posting=posting,
                resume=resume,
                baseline_report=baseline_report,
                client=fake,
                output_dir=Path(directory),
                max_tool_calls=20,
                max_turns=10,
            )
        self.assertEqual(summary["attempted_cases"], 1)
        self.assertEqual(summary["target_match_rate"], 1.0)
        self.assertEqual(summary["verdict_nonincrease_rate"], 1.0)
        self.assertEqual(summary["verdict_decrease_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
