import json
import unittest

from jobfit.baseline import run_one_shot_baseline
from jobfit.documents import parse_document


class FakePlainClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete_plain(self, messages):
        self.calls += 1
        return {"role": "assistant", "content": json.dumps(self.payload)}


class BaselineTests(unittest.TestCase):
    def test_one_shot_output_uses_same_validation_and_scoring(self):
        posting = parse_document("자격요건\nPython 경험", prefix="P", source="posting")
        resume = parse_document("Python 모델 개발", prefix="R", source="resume")
        client = FakePlainClient(
            {
                "requirements": [
                    {
                        "requirement_type": "required",
                        "text": "Python 경험",
                        "posting_line_ids": ["P002"],
                        "verdict": "satisfied",
                        "resume_line_ids": ["R001"],
                        "explanation": "직접 경험",
                    }
                ],
                "excluded_posting_lines": [
                    {"posting_line_ids": ["P001"], "reason": "섹션 제목"}
                ],
            }
        )
        report, _ = run_one_shot_baseline(posting, resume, client)
        self.assertEqual(client.calls, 1)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["score"]["overall_fit_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
