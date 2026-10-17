from __future__ import annotations

import hashlib
import html
import re
import subprocess
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from .models import Document, DocumentLine


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in {
            "p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr"
        }:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        value = html.unescape("".join(self.parts))
        value = re.sub(r"[\t\f\v ]+", " ", value)
        value = re.sub(r"\n\s*\n+", "\n\n", value)
        return value.strip()


def _read_pdf(path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("PDF 입력에는 pdftotext가 필요합니다.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"PDF 텍스트 추출 실패: {exc.stderr.strip()}") from exc
    return result.stdout


def _read_url(url: str, timeout: float = 20.0) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "job-fit-agent/0.1 (single user-supplied page)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    parser = _VisibleTextParser()
    parser.feed(raw.decode(charset, errors="replace"))
    return parser.text()


def load_text(value: str) -> tuple[str, str]:
    """Read a text file, PDF, or web page."""
    if value.startswith(("https://", "http://")):
        return _read_url(value), value

    path = Path(value).expanduser()
    if path.exists():
        if not path.is_file():
            raise ValueError(f"파일이 아닙니다: {path}")
        if path.suffix.lower() == ".pdf":
            return _read_pdf(path), str(path.resolve())
        return path.read_text(encoding="utf-8"), str(path.resolve())
    raise FileNotFoundError(
        f"입력 파일을 찾을 수 없습니다: {value}. 직접 텍스트는 --posting-text 또는 --resume-text를 사용하세요."
    )


def literal_text(value: str, *, source: str) -> tuple[str, str]:
    if not value.strip():
        raise ValueError("직접 입력한 텍스트가 비어 있습니다.")
    return value, source


def load_document(value: str, *, prefix: str) -> Document:
    """Load one source and assign stable line IDs."""
    text, source = load_text(value)
    return parse_document(text, prefix=prefix, source=source)


def remove_document_lines(document: Document, line_ids: set[str]) -> Document:
    """Remove selected lines while preserving every remaining line ID."""
    unknown = sorted(line_ids - set(document.by_id))
    if unknown:
        raise ValueError(f"제거할 줄 ID가 문서에 없습니다: {unknown}")
    remaining = [line for line in document.lines if line.line_id not in line_ids]
    sections = {
        section_id: [line_id for line_id in ids if line_id not in line_ids]
        for section_id, ids in document.sections.items()
    }
    sections = {section_id: ids for section_id, ids in sections.items() if ids}
    fingerprint_payload = "\n".join(f"{line.line_id}:{line.text}" for line in remaining)
    fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16]
    return Document(
        source=f"{document.source}#removed={','.join(sorted(line_ids))}",
        fingerprint=fingerprint,
        lines=remaining,
        sections=sections,
    )


def parse_document(text: str, *, prefix: str, source: str) -> Document:
    """Assign stable IDs to non-empty lines and blank-line-delimited sections."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    lines: list[DocumentLine] = []
    sections: dict[str, list[str]] = {}
    section_number = 1
    seen_content = False
    previous_blank = False

    for raw in normalized.split("\n"):
        value = re.sub(r"\s+", " ", raw).strip()
        if not value:
            if seen_content:
                previous_blank = True
            continue
        if previous_blank:
            section_number += 1
            previous_blank = False
        seen_content = True
        line_id = f"{prefix}{len(lines) + 1:03d}"
        section_id = f"S{section_number:03d}"
        line = DocumentLine(line_id=line_id, text=value, section_id=section_id)
        lines.append(line)
        sections.setdefault(section_id, []).append(line_id)

    if not lines:
        raise ValueError("입력 문서에 처리할 텍스트가 없습니다.")
    return Document(source=source, fingerprint=fingerprint, lines=lines, sections=sections)


def _tokens(text: str) -> set[str]:
    words = set(re.findall(r"[\w+#.\-/]+", text.lower(), flags=re.UNICODE))
    compact = re.sub(r"\s+", "", text.lower())
    grams = {compact[i : i + 2] for i in range(max(0, len(compact) - 1))}
    return {token for token in words | grams if token}


def search_document(document: Document, query: str, limit: int = 8) -> list[dict[str, object]]:
    query_tokens = _tokens(query)
    query_lower = query.lower().strip()
    ranked: list[tuple[float, DocumentLine]] = []

    for line in document.lines:
        line_tokens = _tokens(line.text)
        overlap = len(query_tokens & line_tokens)
        union = len(query_tokens | line_tokens) or 1
        score = overlap / union
        if query_lower and query_lower in line.text.lower():
            score += 2.0
        if score > 0:
            ranked.append((score, line))

    ranked.sort(key=lambda item: (-item[0], item[1].line_id))
    return [
        {
            "line_id": line.line_id,
            "section_id": line.section_id,
            "text": line.text,
            "score": round(score, 4),
        }
        for score, line in ranked[:limit]
    ]
