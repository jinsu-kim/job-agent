import unittest

from jobfit.documents import parse_document, search_document


class DocumentTests(unittest.TestCase):
    def test_assigns_line_and_section_ids(self):
        document = parse_document("첫 줄\n둘째 줄\n\n새 섹션", prefix="R", source="test")
        self.assertEqual([line.line_id for line in document.lines], ["R001", "R002", "R003"])
        self.assertEqual(document.lines[0].section_id, "S001")
        self.assertEqual(document.lines[2].section_id, "S002")

    def test_search_finds_technology(self):
        document = parse_document(
            "PyTorch로 GNN 모델 개발\n\nSelenium 수집 파이프라인",
            prefix="R",
            source="test",
        )
        results = search_document(document, "PyTorch", limit=3)
        self.assertEqual(results[0]["line_id"], "R001")


if __name__ == "__main__":
    unittest.main()
