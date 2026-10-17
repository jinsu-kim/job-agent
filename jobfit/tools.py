from __future__ import annotations

from typing import Any


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_DEFINITIONS = [
    _function(
        "read_posting",
        "ID가 부여된 채용공고 전체와 현재 처리 상태를 읽는다.",
        {},
        [],
    ),
    _function(
        "record_requirement",
        "채용공고의 자격요건 또는 우대사항을 공고 원문 줄 ID와 함께 등록한다.",
        {
            "requirement_type": {
                "type": "string",
                "enum": ["required", "preferred"],
            },
            "text": {"type": "string"},
            "posting_line_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        ["requirement_type", "text", "posting_line_ids"],
    ),
    _function(
        "exclude_posting_lines",
        "요건이 아닌 공고 줄을 제외 처리한다. 제목, 회사 소개, 복지, 근무지 등이 대상이다.",
        {
            "posting_line_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "reason": {"type": "string"},
        },
        ["posting_line_ids", "reason"],
    ),
    _function(
        "search_resume",
        "키워드로 이력서 줄을 검색한다. 표현이 다를 수 있으므로 동의어로 여러 번 검색할 수 있다.",
        {
            "requirement_id": {"type": "string"},
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ["requirement_id", "query"],
    ),
    _function(
        "get_resume_section",
        "검색된 줄이 속한 이력서 섹션의 전체 원문을 읽는다.",
        {
            "requirement_id": {"type": "string"},
            "section_id": {"type": "string"},
        },
        ["requirement_id", "section_id"],
    ),
    _function(
        "record_judgment",
        "요건의 판정과 실제 이력서 근거 줄 ID를 기록한다.",
        {
            "requirement_id": {"type": "string"},
            "verdict": {
                "type": "string",
                "enum": ["satisfied", "partial", "unsatisfied", "unjudgeable"],
            },
            "resume_line_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "explanation": {"type": "string"},
        },
        ["requirement_id", "verdict", "resume_line_ids", "explanation"],
    ),
    _function(
        "finalize_report",
        "모든 공고 줄과 요건이 처리됐는지 검사하고 규칙 기반 결과 보고서를 생성한다.",
        {},
        [],
    ),
]

TOOL_NAMES = frozenset(tool["function"]["name"] for tool in TOOL_DEFINITIONS)
