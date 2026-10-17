import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jobfit.documents import literal_text, load_document, load_text


class InputTests(unittest.TestCase):
    def test_mistyped_path_is_not_treated_as_literal_text(self):
        with self.assertRaises(FileNotFoundError):
            load_text("exmaples/resume.md")

    def test_explicit_literal_text(self):
        text, source = literal_text("직접 입력", source="posting:text")
        self.assertEqual(text, "직접 입력")
        self.assertEqual(source, "posting:text")

    def test_reads_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.txt"
            path.write_text("Python 경험", encoding="utf-8")
            text, source = load_text(str(path))
            self.assertEqual(text, "Python 경험")
            self.assertEqual(source, str(path.resolve()))

    @patch("jobfit.documents._read_url", return_value="경력\nPython 모델 개발")
    def test_url_works_for_any_document(self, read_url):
        document = load_document("https://example.com/resume", prefix="R")
        self.assertEqual(document.source, "https://example.com/resume")
        self.assertEqual(document.lines[1].line_id, "R002")
        read_url.assert_called_once_with("https://example.com/resume")


if __name__ == "__main__":
    unittest.main()
