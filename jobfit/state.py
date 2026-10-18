from __future__ import annotations

import hashlib
from typing import Any

from .documents import search_document
from .models import Document, Judgment, Requirement, RequirementType, Verdict
from .report import build_report
from .tools import TOOL_NAMES


class ToolError(ValueError):
    pass


class ToolBudgetExceeded(RuntimeError):
    pass


class FitState:
    """State machine and hard validation boundary for the LLM tool loop."""

    def __init__(self, posting: Document, resume: Document, *, max_tool_calls: int = 80) -> None:
        self.posting = posting
        self.resume = resume
        self.max_tool_calls = max_tool_calls
        self.tool_calls = 0
        self.requirements: dict[str, Requirement] = {}
        self.judgments: dict[str, Judgment] = {}
        self.resume_searches: dict[str, list[dict[str, Any]]] = {}
        self.observed_resume_lines: dict[str, set[str]] = {}
        self.excluded_posting_lines: dict[str, str] = {}
        self.trace: list[dict[str, Any]] = []
        self.finalized = False
        self.final_report: dict[str, Any] | None = None

    def _require_posting_ids(self, line_ids: list[str]) -> None:
        missing = [line_id for line_id in line_ids if line_id not in self.posting.by_id]
        if missing:
            raise ToolError(f"존재하지 않는 공고 줄 ID: {missing}")

    def _require_resume_ids(self, line_ids: list[str]) -> None:
        missing = [line_id for line_id in line_ids if line_id not in self.resume.by_id]
        if missing:
            raise ToolError(f"존재하지 않는 이력서 줄 ID: {missing}")

    def _requirement_lines(self) -> set[str]:
        return {
            line_id
            for requirement in self.requirements.values()
            for line_id in requirement.posting_line_ids
        }

    def _status(self) -> dict[str, Any]:
        covered = self._requirement_lines() | set(self.excluded_posting_lines)
        return {
            "posting_lines": len(self.posting.lines),
            "covered_posting_lines": len(covered),
            "unhandled_posting_line_ids": [
                line.line_id for line in self.posting.lines if line.line_id not in covered
            ],
            "requirements": len(self.requirements),
            "judged_requirements": len(self.judgments),
            "unjudged_requirement_ids": [
                requirement_id
                for requirement_id in self.requirements
                if requirement_id not in self.judgments
            ],
            "tool_calls": self.tool_calls,
            "max_tool_calls": self.max_tool_calls,
        }

    def read_posting(self) -> dict[str, Any]:
        return {
            "source": self.posting.source,
            "fingerprint": self.posting.fingerprint,
            "lines": [line.to_dict() for line in self.posting.lines],
            "status": self._status(),
        }

    def record_requirement(
        self,
        requirement_type: str,
        text: str,
        posting_line_ids: list[str],
    ) -> dict[str, Any]:
        try:
            kind = RequirementType(requirement_type)
        except ValueError as exc:
            raise ToolError(f"잘못된 요건 유형: {requirement_type}") from exc
        text = text.strip()
        line_ids = list(dict.fromkeys(posting_line_ids))
        if not text:
            raise ToolError("요건 내용이 비어 있습니다.")
        if not line_ids:
            raise ToolError("공고 근거 줄 ID가 필요합니다.")
        self._require_posting_ids(line_ids)
        conflicts = [line_id for line_id in line_ids if line_id in self.excluded_posting_lines]
        if conflicts:
            raise ToolError(f"이미 비요건으로 제외된 공고 줄: {conflicts}")

        normalized = " ".join(text.lower().split())
        for existing in self.requirements.values():
            if existing.requirement_type == kind and " ".join(existing.text.lower().split()) == normalized:
                raise ToolError(f"중복 요건입니다: {existing.requirement_id}")
        claimed_lines = self._requirement_lines()
        duplicated_lines = [line_id for line_id in line_ids if line_id in claimed_lines]
        if duplicated_lines:
            raise ToolError(
                "하나의 공고 줄을 여러 요건으로 분할할 수 없습니다. "
                f"이미 다른 요건에 사용된 줄: {duplicated_lines}"
            )

        requirement_id = f"Q{len(self.requirements) + 1:03d}"
        # The key must remain stable even when the LLM paraphrases requirement text.
        # One source line can belong to only one requirement, so source IDs identify it.
        key_payload = f"{kind.value}|{'|'.join(sorted(line_ids))}"
        requirement_key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()[:12]
        requirement = Requirement(
            requirement_id=requirement_id,
            requirement_key=requirement_key,
            requirement_type=kind,
            text=text,
            posting_line_ids=line_ids,
        )
        self.requirements[requirement_id] = requirement
        return requirement.to_dict()

    def exclude_posting_lines(self, posting_line_ids: list[str], reason: str) -> dict[str, Any]:
        line_ids = list(dict.fromkeys(posting_line_ids))
        reason = reason.strip()
        if not line_ids:
            raise ToolError("제외할 공고 줄 ID가 필요합니다.")
        if not reason:
            raise ToolError("제외 이유가 필요합니다.")
        self._require_posting_ids(line_ids)
        requirement_lines = self._requirement_lines()
        conflicts = [line_id for line_id in line_ids if line_id in requirement_lines]
        if conflicts:
            raise ToolError(f"이미 요건 근거로 사용된 공고 줄: {conflicts}")
        for line_id in line_ids:
            self.excluded_posting_lines[line_id] = reason
        return {"excluded": line_ids, "reason": reason, "status": self._status()}

    def search_resume(self, requirement_id: str, query: str, limit: int = 8) -> dict[str, Any]:
        if requirement_id not in self.requirements:
            raise ToolError(f"존재하지 않는 요건 ID: {requirement_id}")
        query = query.strip()
        if not query:
            raise ToolError("검색어가 비어 있습니다.")
        if not 1 <= limit <= 20:
            raise ToolError("limit은 1~20이어야 합니다.")
        results = search_document(self.resume, query, limit)
        self.resume_searches.setdefault(requirement_id, []).append(
            {"query": query, "result_line_ids": [item["line_id"] for item in results]}
        )
        self.observed_resume_lines.setdefault(requirement_id, set()).update(
            item["line_id"] for item in results
        )
        return {"requirement_id": requirement_id, "query": query, "results": results}

    def get_resume_section(self, requirement_id: str, section_id: str) -> dict[str, Any]:
        if requirement_id not in self.requirements:
            raise ToolError(f"존재하지 않는 요건 ID: {requirement_id}")
        if requirement_id not in self.resume_searches:
            raise ToolError("이 요건에 대해 먼저 search_resume을 호출해야 합니다.")
        if section_id not in self.resume.sections:
            raise ToolError(f"존재하지 않는 이력서 섹션 ID: {section_id}")
        self.observed_resume_lines.setdefault(requirement_id, set()).update(
            self.resume.sections[section_id]
        )
        return {
            "requirement_id": requirement_id,
            "section_id": section_id,
            "lines": [
                self.resume.by_id[line_id].to_dict()
                for line_id in self.resume.sections[section_id]
            ],
        }

    def record_judgment(
        self,
        requirement_id: str,
        verdict: str,
        resume_line_ids: list[str],
        explanation: str,
    ) -> dict[str, Any]:
        if requirement_id not in self.requirements:
            raise ToolError(f"존재하지 않는 요건 ID: {requirement_id}")
        if requirement_id not in self.resume_searches:
            raise ToolError("판정 전에 해당 요건 ID로 search_resume을 한 번 이상 호출해야 합니다.")
        try:
            verdict_value = Verdict(verdict)
        except ValueError as exc:
            raise ToolError(f"잘못된 판정: {verdict}") from exc
        line_ids = list(dict.fromkeys(resume_line_ids))
        explanation = explanation.strip()
        self._require_resume_ids(line_ids)
        if verdict_value in {Verdict.SATISFIED, Verdict.PARTIAL} and not line_ids:
            raise ToolError("충족 또는 부분 충족 판정에는 이력서 근거 줄이 필요합니다.")
        unobserved = sorted(
            set(line_ids) - self.observed_resume_lines.get(requirement_id, set())
        )
        if unobserved:
            raise ToolError(
                "해당 요건의 검색 또는 섹션 조회에서 확인하지 않은 줄은 근거로 사용할 수 없습니다: "
                f"{unobserved}"
            )
        if not explanation:
            raise ToolError("판정 설명이 필요합니다.")

        judgment = Judgment(
            requirement_id=requirement_id,
            verdict=verdict_value,
            resume_line_ids=line_ids,
            explanation=explanation,
        )
        self.judgments[requirement_id] = judgment
        return {**judgment.to_dict(), "status": self._status()}

    def _build_report(self, *, status: str, incomplete_reason: str | None = None) -> dict[str, Any]:
        return build_report(
            posting=self.posting,
            resume=self.resume,
            requirements=self.requirements,
            judgments=self.judgments,
            searches=self.resume_searches,
            excluded_lines=self.excluded_posting_lines,
            status=status,
            incomplete_reason=incomplete_reason,
        )

    def finalize_report(self) -> dict[str, Any]:
        status = self._status()
        errors: list[str] = []
        if status["unhandled_posting_line_ids"]:
            errors.append(
                f"처리되지 않은 공고 줄: {status['unhandled_posting_line_ids']}"
            )
        if not self.requirements:
            errors.append("등록된 자격요건 또는 우대사항이 없습니다.")
        if status["unjudged_requirement_ids"]:
            errors.append(
                f"판정되지 않은 요건: {status['unjudged_requirement_ids']}"
            )
        if errors:
            raise ToolError("; ".join(errors))

        self.final_report = self._build_report(
            status="complete",
            incomplete_reason=None,
        )
        self.finalized = True
        return self.final_report

    def incomplete_report(self, reason: str) -> dict[str, Any]:
        self.final_report = self._build_report(
            status="incomplete",
            incomplete_reason=reason,
        )
        self.finalized = True
        return self.final_report

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.tool_calls >= self.max_tool_calls:
            raise ToolBudgetExceeded(f"도구 호출 예산 {self.max_tool_calls}회를 초과했습니다.")
        self.tool_calls += 1
        record: dict[str, Any] = {
            "call_number": self.tool_calls,
            "tool": tool_name,
            "arguments": arguments,
        }
        try:
            if tool_name not in TOOL_NAMES:
                raise ToolError(f"허용되지 않은 도구: {tool_name}")
            handler = getattr(self, tool_name, None)
            if handler is None:
                raise ToolError(f"알 수 없는 도구: {tool_name}")
            result = handler(**arguments)
            record.update({"ok": True, "result": result})
            return result
        except Exception as exc:
            record.update({"ok": False, "error": str(exc)})
            raise
        finally:
            self.trace.append(record)
