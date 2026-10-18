from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RequirementType(StrEnum):
    REQUIRED = "required"
    PREFERRED = "preferred"


class Verdict(StrEnum):
    SATISFIED = "satisfied"
    PARTIAL = "partial"
    UNSATISFIED = "unsatisfied"

@dataclass(frozen=True)
class DocumentLine:
    line_id: str
    text: str
    section_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Document:
    source: str
    fingerprint: str
    lines: list[DocumentLine]
    sections: dict[str, list[str]] = field(default_factory=dict)

    @property
    def by_id(self) -> dict[str, DocumentLine]:
        return {line.line_id: line for line in self.lines}

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "fingerprint": self.fingerprint,
            "lines": [line.to_dict() for line in self.lines],
            "sections": self.sections,
        }


@dataclass
class Requirement:
    requirement_id: str
    requirement_key: str
    requirement_type: RequirementType
    text: str
    posting_line_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["requirement_type"] = self.requirement_type.value
        return data


@dataclass
class Judgment:
    requirement_id: str
    verdict: Verdict
    resume_line_ids: list[str]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data
