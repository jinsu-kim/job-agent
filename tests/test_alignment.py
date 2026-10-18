import unittest

from jobfit.alignment import (
    line_classification,
    line_predictions,
    line_verdict,
    requirements_by_key,
)


class AlignmentTests(unittest.TestCase):
    def test_preserves_multiple_requirements_from_one_posting_line(self):
        report = {
            "requirements": [
                {
                    "requirement_key": "python",
                    "requirement_type": "required",
                    "text": "Python 경험",
                    "posting_line_ids": ["P002"],
                    "judgment": {
                        "verdict": "satisfied",
                        "resume_line_ids": ["R001"],
                    },
                },
                {
                    "requirement_key": "pytorch",
                    "requirement_type": "required",
                    "text": "PyTorch 경험",
                    "posting_line_ids": ["P002"],
                    "judgment": {
                        "verdict": "unsatisfied",
                        "resume_line_ids": [],
                    },
                },
            ],
            "excluded_posting_lines": [
                {"line_id": "P001", "reason": "섹션 제목"}
            ],
        }

        aligned = line_predictions(report)

        self.assertEqual(len(aligned["P002"]), 2)
        self.assertEqual(
            {item["requirement_key"] for item in aligned["P002"]},
            {"python", "pytorch"},
        )
        self.assertEqual(line_classification(aligned["P002"]), "required")
        self.assertEqual(line_verdict(aligned["P002"]), "partial")
        self.assertEqual(set(requirements_by_key(aligned)), {"python", "pytorch"})
        self.assertEqual(line_classification(aligned["P001"]), "non_requirement")


if __name__ == "__main__":
    unittest.main()
