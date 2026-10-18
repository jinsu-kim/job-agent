import unittest

from jobfit.documents import parse_document
from jobfit.state import FitState, ToolError


def make_state() -> FitState:
    posting = parse_document(
        "자격요건\nPython 경험\n\n우대사항\nDocker 경험",
        prefix="P",
        source="posting",
    )
    resume = parse_document(
        "기술\nPython과 PyTorch로 모델 개발\n\n경력\nAPI 배포 경험",
        prefix="R",
        source="resume",
    )
    return FitState(posting, resume, max_tool_calls=50)


def search_for(state: FitState, requirement_id: str, query: str = "Python") -> None:
    state.search_resume(requirement_id, query)


class StateTests(unittest.TestCase):
    def test_rejects_nonexistent_posting_evidence(self):
        state = make_state()
        with self.assertRaises(ToolError):
            state.record_requirement("required", "Python 경험", ["P999"])

    def test_positive_judgment_requires_resume_evidence(self):
        state = make_state()
        requirement = state.record_requirement("required", "Python 경험", ["P002"])
        search_for(state, requirement["requirement_id"])
        with self.assertRaises(ToolError):
            state.record_judgment(
                requirement["requirement_id"], "satisfied", [], "경험이 있음"
            )

    def test_rejects_nonexistent_resume_evidence(self):
        state = make_state()
        requirement = state.record_requirement("required", "Python 경험", ["P002"])
        search_for(state, requirement["requirement_id"])
        with self.assertRaises(ToolError):
            state.record_judgment(
                requirement["requirement_id"], "satisfied", ["R999"], "경험이 있음"
            )

    def test_judgment_requires_search_for_same_requirement(self):
        state = make_state()
        requirement = state.record_requirement("required", "Python 경험", ["P002"])
        with self.assertRaises(ToolError):
            state.record_judgment(
                requirement["requirement_id"], "unsatisfied", [], "근거 없음"
            )

    def test_positive_evidence_must_have_been_observed(self):
        state = make_state()
        requirement = state.record_requirement("required", "API 경험", ["P002"])
        state.search_resume(requirement["requirement_id"], "존재하지않는검색어")
        with self.assertRaises(ToolError):
            state.record_judgment(
                requirement["requirement_id"], "satisfied", ["R004"], "API 경험"
            )

    def test_allows_multiple_requirements_from_one_posting_line(self):
        state = make_state()

        python = state.record_requirement("required", "Python 경험", ["P002"])
        pytorch = state.record_requirement("required", "PyTorch 경험", ["P002"])

        self.assertEqual(python["posting_line_ids"], ["P002"])
        self.assertEqual(pytorch["posting_line_ids"], ["P002"])
        self.assertNotEqual(python["requirement_id"], pytorch["requirement_id"])
        self.assertNotEqual(python["requirement_key"], pytorch["requirement_key"])

    def test_finalize_requires_full_posting_coverage(self):
        state = make_state()
        requirement = state.record_requirement("required", "Python 경험", ["P002"])
        search_for(state, requirement["requirement_id"])
        state.record_judgment(
            requirement["requirement_id"], "satisfied", ["R002"], "직접 경험"
        )
        with self.assertRaises(ToolError):
            state.finalize_report()

    def test_deterministic_scoring_and_complete_report(self):
        state = make_state()
        state.exclude_posting_lines(["P001", "P003"], "섹션 제목")
        required = state.record_requirement("required", "Python 경험", ["P002"])
        preferred = state.record_requirement("preferred", "Docker 경험", ["P004"])
        search_for(state, required["requirement_id"])
        search_for(state, preferred["requirement_id"], "Docker")
        state.record_judgment(
            required["requirement_id"], "satisfied", ["R002"], "Python 직접 사용"
        )
        state.record_judgment(
            preferred["requirement_id"], "unsatisfied", [], "Docker 근거 없음"
        )
        report = state.finalize_report()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["score"]["required_percent"], 100.0)
        self.assertEqual(report["score"]["preferred_percent"], 0.0)
        self.assertEqual(report["score"]["overall_fit_percent"], 80.0)

    def test_incomplete_report_does_not_invent_judgments(self):
        state = make_state()
        requirement = state.record_requirement("required", "Python 경험", ["P002"])

        report = state.incomplete_report("도구 호출 한도 초과")

        self.assertEqual(report["status"], "incomplete")
        self.assertNotIn(requirement["requirement_id"], state.judgments)
        self.assertIsNone(report["requirements"][0]["judgment"])
        self.assertIsNone(report["score"])

    def test_dispatch_rejects_internal_method(self):
        state = make_state()
        with self.assertRaises(ToolError):
            state.dispatch("incomplete_report", {"reason": "early exit"})


if __name__ == "__main__":
    unittest.main()
