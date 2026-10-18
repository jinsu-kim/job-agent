from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .state import FitState, ToolBudgetExceeded
from .tools import TOOL_DEFINITIONS


SYSTEM_PROMPT = """당신은 채용공고와 이력서 사이의 적합도를 평가하는 도구 사용 에이전트입니다.

반드시 다음 순서와 규칙을 따르세요.
1. read_posting으로 공고의 모든 줄을 읽으세요.
2. 자격요건과 우대사항만 record_requirement로 등록하세요. 각 요건은 반드시 실제 공고 줄 ID를 인용해야 합니다. 한 줄에 독립적으로 충족 여부를 판단할 수 있는 복수 요건이 있으면 각각 분리하여 등록하고, 분리된 요건들은 같은 공고 줄 ID를 인용할 수 있습니다. 여러 줄에 걸친 하나의 요건은 한 번에 묶어 등록하세요.
3. 제목, 회사소개, 주요업무, 복지, 근무지처럼 평가 요건이 아닌 모든 나머지 줄은 exclude_posting_lines로 처리하세요. 주요업무에 명시된 필수 역량이 자격요건 역할을 하는 경우에만 required로 등록할 수 있습니다.
4. 각 요건마다 해당 requirement_id를 넣어 search_resume을 한 번 이상 사용하세요. 표현이 다르면 동의어와 구체 기술명으로 다시 검색하고, 필요하면 같은 requirement_id로 get_resume_section을 호출해 문맥을 확인하세요.
5. record_judgment로 모든 요건을 판정하세요.
   - satisfied: 이력서에 직접적이고 충분한 근거가 있음
   - partial: 관련 근거는 있으나 범위, 숙련도, 기간 또는 직접성이 부족함
   - unsatisfied: 이력서에서 근거를 찾지 못했거나 명시적 기준에 미달함
6. satisfied와 partial에는 실제 이력서 줄 ID가 반드시 필요합니다. 줄이 존재한다는 사실과 의미적으로 타당한 근거라는 사실을 혼동하지 마세요.
7. 점수는 직접 만들거나 수정하지 마세요. 모든 처리가 끝나면 finalize_report를 호출하세요.
8. 도구 오류가 나면 오류 내용을 고쳐 다시 호출하세요. 최종 답변만 작성하고 도구를 생략하지 마세요.
"""


@dataclass
class OpenAICompatibleClient:
    api_base: str
    model: str
    api_key: str = ""
    timeout: float = 90.0
    temperature: float | None = None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API 오류 {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API 연결 실패: {exc.reason}") from exc

        try:
            return body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"예상하지 못한 LLM 응답: {body}") from exc

    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto",
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return self._request(payload)

    def complete_plain(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return self._request(payload)


def run_agent(
    state: FitState,
    client: OpenAICompatibleClient,
    *,
    max_turns: int = 30,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "입력된 채용공고와 이력서의 핏을 평가하세요. "
                "먼저 공고를 읽고 모든 구조적 제약을 만족시킨 뒤 보고서를 확정하세요."
            ),
        },
    ]

    for _ in range(max_turns):
        message = client.complete(messages)
        messages.append(message)
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            if state.finalized and state.final_report is not None:
                return state.final_report
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "작업이 아직 완료되지 않았습니다. 도구를 사용해 남은 공고 줄과 "
                        "요건을 처리하고 finalize_report를 호출하세요."
                    ),
                }
            )
            continue

        for call in tool_calls:
            call_id = call.get("id", "missing-call-id")
            function = call.get("function") or {}
            name = function.get("name", "")
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
                result = state.dispatch(name, arguments)
                content = json.dumps({"ok": True, "result": result}, ensure_ascii=False)
            except ToolBudgetExceeded as exc:
                return state.incomplete_report(str(exc))
            except Exception as exc:
                content = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            )
            if state.finalized and state.final_report is not None:
                return state.final_report

    return state.incomplete_report(f"최대 대화 턴 {max_turns}회를 초과했습니다.")
