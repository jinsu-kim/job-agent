from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from .agent import OpenAICompatibleClient, run_agent
from .baseline import run_one_shot_baseline
from .documents import literal_text, load_document, parse_document
from .evaluation import evaluate
from .mutation import run_removal_mutations
from .state import FitState


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_documents(args: argparse.Namespace):
    if args.posting is not None:
        posting = load_document(args.posting, prefix="P")
    else:
        posting_text, posting_source = literal_text(
            args.posting_text, source="posting:direct-text"
        )
        posting = parse_document(posting_text, prefix="P", source=posting_source)
    if args.resume is not None:
        resume = load_document(args.resume, prefix="R")
    else:
        resume_text, resume_source = literal_text(
            args.resume_text, source="resume:direct-text"
        )
        resume = parse_document(resume_text, prefix="R", source=resume_source)
    return posting, resume


def _client(args: argparse.Namespace) -> OpenAICompatibleClient:
    api_base = args.api_base or os.getenv("JOBFIT_API_BASE")
    api_key = os.getenv("JOBFIT_API_KEY", "")
    model = args.model or os.getenv("JOBFIT_MODEL")
    if not api_base or not model:
        raise SystemExit(
            "--api-base/JOBFIT_API_BASE와 --model/JOBFIT_MODEL을 설정해야 합니다."
        )
    return OpenAICompatibleClient(
        api_base=api_base,
        api_key=api_key,
        model=model,
        timeout=args.timeout,
        temperature=args.temperature,
    )


def _run_dir(value: str | None, prefix: str = "run") -> Path:
    return (
        Path(value)
        if value
        else Path("runs") / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )


def _analyze(args: argparse.Namespace) -> int:
    posting, resume = _load_documents(args)

    if args.dry_run:
        print(json.dumps({"posting": posting.to_dict(), "resume": resume.to_dict()}, ensure_ascii=False, indent=2))
        return 0

    state = FitState(posting, resume, max_tool_calls=args.max_tool_calls)
    report = run_agent(state, _client(args), max_turns=args.max_turns)

    run_dir = _run_dir(args.output)
    report["run_id"] = run_dir.name
    report["system"] = "tool_agent"
    _write_json(run_dir / "report.json", report)
    _write_json(run_dir / "trace.json", state.trace)
    _write_json(run_dir / "posting.json", posting.to_dict())
    _write_json(run_dir / "resume.json", resume.to_dict())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n저장 위치: {run_dir.resolve()}")
    return 0 if report["status"] == "complete" else 2


def _baseline(args: argparse.Namespace) -> int:
    posting, resume = _load_documents(args)
    report, raw = run_one_shot_baseline(posting, resume, _client(args))
    run_dir = _run_dir(args.output, prefix="baseline")
    report["run_id"] = run_dir.name
    _write_json(run_dir / "report.json", report)
    _write_json(run_dir / "raw_response.json", raw)
    _write_json(run_dir / "posting.json", posting.to_dict())
    _write_json(run_dir / "resume.json", resume.to_dict())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n저장 위치: {run_dir.resolve()}")
    return 0 if report["status"] == "complete" else 2


def _mutation_eval(args: argparse.Namespace) -> int:
    posting, resume = _load_documents(args)
    baseline_report = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    output_dir = _run_dir(args.output, prefix="mutation")
    summary = run_removal_mutations(
        posting=posting,
        resume=resume,
        baseline_report=baseline_report,
        client=_client(args),
        output_dir=output_dir,
        max_tool_calls=args.max_tool_calls,
        max_turns=args.max_turns,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n저장 위치: {output_dir.resolve()}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    result = evaluate(
        args.labels,
        args.reports,
        evidence_labels_path=args.evidence_labels,
        baseline_report_paths=args.baseline_reports,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jobfit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="공고 하나와 이력서 하나의 핏을 평가")
    _add_document_arguments(analyze)
    _add_api_arguments(analyze)
    analyze.add_argument("--output", help="실행 결과 디렉터리")
    analyze.add_argument("--max-tool-calls", type=int, default=80)
    analyze.add_argument("--max-turns", type=int, default=30)
    analyze.add_argument("--dry-run", action="store_true", help="API 호출 없이 입력 줄 ID만 출력")
    analyze.set_defaults(func=_analyze)

    baseline = subparsers.add_parser("baseline", help="도구 없는 단일 프롬프트 baseline 실행")
    _add_document_arguments(baseline)
    _add_api_arguments(baseline)
    baseline.add_argument("--output", help="실행 결과 디렉터리")
    baseline.set_defaults(func=_baseline)

    mutation = subparsers.add_parser(
        "mutation-eval", help="인용한 이력서 근거 줄을 제거하고 에이전트를 다시 실행"
    )
    _add_document_arguments(mutation)
    _add_api_arguments(mutation)
    mutation.add_argument("--baseline-report", required=True)
    mutation.add_argument("--output", help="실행 결과 디렉터리")
    mutation.add_argument("--max-tool-calls", type=int, default=80)
    mutation.add_argument("--max-turns", type=int, default=30)
    mutation.add_argument("--limit", type=int, help="비용 제어를 위한 최대 변형 수")
    mutation.set_defaults(func=_mutation_eval)

    evaluation = subparsers.add_parser("evaluate", help="수동 라벨과 반복 실행 결과 평가")
    evaluation.add_argument("--labels", required=True, help="평가 라벨 CSV")
    evaluation.add_argument("--evidence-labels", help="실행별 근거 타당성 라벨 CSV")
    evaluation.add_argument(
        "--baseline-report",
        dest="baseline_reports",
        action="append",
        default=[],
        help="단일 프롬프트 report.json; 여러 개면 옵션을 반복",
    )
    evaluation.add_argument("reports", nargs="+", help="report.json 파일들")
    evaluation.set_defaults(func=_evaluate)
    return parser


def _add_document_arguments(parser: argparse.ArgumentParser) -> None:
    posting = parser.add_mutually_exclusive_group(required=True)
    posting.add_argument("--posting", help="공고 파일, PDF 또는 URL")
    posting.add_argument("--posting-text", help="공고 원문 직접 입력")
    resume = parser.add_mutually_exclusive_group(required=True)
    resume.add_argument("--resume", help="이력서 파일, PDF 또는 URL")
    resume.add_argument("--resume-text", help="이력서 원문 직접 입력")


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-base", help="OpenAI 호환 API의 /v1 기준 URL")
    parser.add_argument("--model", help="호출할 모델 이름")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="미지정 시 요청에서 제외; 사용하는 모델이 지원할 때만 설정",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))
