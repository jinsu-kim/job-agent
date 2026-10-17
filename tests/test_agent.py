import unittest

from jobfit.agent import run_agent
from jobfit.documents import parse_document
from jobfit.state import FitState


def tool_call(call_id, name, arguments):
    import json

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


class AgentLoopTests(unittest.TestCase):
    def test_runs_tools_until_finalized(self):
        posting = parse_document("자격요건\nPython 경험", prefix="P", source="posting")
        resume = parse_document("Python으로 모델 개발", prefix="R", source="resume")
        state = FitState(posting, resume)
        client = FakeClient(
            [
                {"role": "assistant", "content": None, "tool_calls": [tool_call("1", "read_posting", {})]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call(
                            "2",
                            "record_requirement",
                            {
                                "requirement_type": "required",
                                "text": "Python 경험",
                                "posting_line_ids": ["P002"],
                            },
                        ),
                        tool_call(
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
                        tool_call(
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
                        tool_call(
                            "5",
                            "record_judgment",
                            {
                                "requirement_id": "Q001",
                                "verdict": "satisfied",
                                "resume_line_ids": ["R001"],
                                "explanation": "Python 개발 경험이 명시됨",
                            },
                        )
                    ],
                },
                {"role": "assistant", "content": None, "tool_calls": [tool_call("6", "finalize_report", {})]},
            ]
        )

        report = run_agent(state, client, max_turns=10)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["score"]["overall_fit_percent"], 100.0)
        self.assertEqual(len(state.trace), 6)


if __name__ == "__main__":
    unittest.main()
