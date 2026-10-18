from __future__ import annotations

import json
import re
from typing import Any

from .agent import OpenAICompatibleClient
from .models import Document
from .state import FitState


BASELINE_SYSTEM_PROMPT = """당신은 채용공고와 이력서의 적합도를 한 번에 평가합니다.
도구를 사용할 수 없습니다. 제공된 줄 ID만 인용하고 JSON 객체 하나만 출력하세요.

스키마:
{
  "requirements": [
    {
      "requirement_type": "required 또는 preferred",
      "text": "요건",
      "posting_line_ids": ["P001"],
      "verdict": "satisfied, partial, unsatisfied 중 하나",
      "resume_line_ids": ["R001"],
      "explanation": "한 줄 설명"
    }
  ],
  "excluded_posting_lines": [
    {"posting_line_ids": ["P002"], "reason": "제외 이유"}
  ]
}

자격요건과 우대사항만 요건으로 등록하세요. 모든 공고 줄은 requirements 또는
excluded_posting_lines 중 하나에 포함해야 합니다. satisfied와 partial에는 직접적인
이력서 근거 줄 ID를 넣으세요. 점수는 출력하지 마세요.
"""


def _line_text(document: Document) -> str:
    return "\n".join(f"[{line.line_id}] {line.text}" for line in document.lines)


def _parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    if not isinstance(content, str):
        raise ValueError("baseline 응답 content가 문자열이 아닙니다.")
    value = content.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("baseline 응답에서 JSON 객체를 찾지 못했습니다.")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("baseline 응답 최상위 값은 객체여야 합니다.")
    return parsed


def run_one_shot_baseline(
    posting: Document,
    resume: Document,
    client: OpenAICompatibleClient,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a single no-tool LLM call, then validate and score with the same state rules."""
    message = client.complete_plain(
        [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "채용공고:\n"
                    f"{_line_text(posting)}\n\n"
                    "이력서:\n"
                    f"{_line_text(resume)}"
                ),
            },
        ]
    )
    raw = _parse_json_content(message.get("content"))
    state = FitState(posting, resume, max_tool_calls=0)

    try:
        for item in raw.get("requirements", []):
            requirement = state.record_requirement(
                item["requirement_type"], item["text"], item["posting_line_ids"]
            )
            requirement_id = requirement["requirement_id"]
            # The one-shot model saw the full resume in its prompt. Mark that explicit
            # observation without pretending that it used the agent's search tool.
            state.resume_searches[requirement_id] = [
                {"query": "[one-shot full resume]", "result_line_ids": list(resume.by_id)}
            ]
            state.observed_resume_lines[requirement_id] = set(resume.by_id)
            state.record_judgment(
                requirement_id,
                item["verdict"],
                item.get("resume_line_ids", []),
                item["explanation"],
            )
        for exclusion in raw.get("excluded_posting_lines", []):
            state.exclude_posting_lines(
                exclusion["posting_line_ids"], exclusion["reason"]
            )
        report = state.finalize_report()
    except Exception as exc:
        report = state.incomplete_report(f"단일 프롬프트 출력 검증 실패: {exc}")

    report["system"] = "one_shot_baseline"
    return report, {"message": message, "parsed": raw}
