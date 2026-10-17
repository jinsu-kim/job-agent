# Job–Resume Fit Agent

채용공고와 이력서를 받아 요건별 근거를 제시하고, 규칙 기반으로 적합도를 계산하는 에이전트입니다.

## Why

일반적인 단일 프롬프트 평가는 요건을 빠뜨리거나, 이력서에 없는 근거를 인용하거나, 실행마다 다른 기준으로 점수를 매길 수 있습니다. 이 프로젝트는 판단을 더 똑똑하게 만드는 대신 **검증 가능한 근거와 일관된 계산 절차**를 강제합니다.

## How it works

`공고 분해 → 요건별 이력서 검색 → 근거 검증 → 충족 여부 판정 → 코드 기반 점수 계산`

- 공고와 이력서의 모든 줄에 ID를 부여합니다.
- 모든 공고 줄을 요건 또는 비요건으로 처리해야 합니다.
- 각 요건은 이력서를 검색한 뒤에만 판정할 수 있습니다.
- `충족`과 `부분 충족`에는 검색에서 확인한 이력서 줄이 필요합니다.
- 점수는 LLM이 아니라 코드가 계산합니다.

코드는 근거 줄의 존재와 처리 절차를 보장하지만, 그 근거가 의미적으로 타당한지와 LLM의 판정이 정확한지는 평가로 측정합니다.

## Quick start

Python 3.11 이상과 tool calling을 지원하는 OpenAI 호환 API가 필요합니다.

```bash
export JOBFIT_API_BASE="https://YOUR_OPENAI_COMPATIBLE_HOST/v1"
export JOBFIT_API_KEY="YOUR_KEY"
export JOBFIT_MODEL="YOUR_TOOL_CALLING_MODEL"

python3.11 -m jobfit analyze \
  --posting examples/posting.txt \
  --resume examples/resume.md
```

입력 구조만 확인하려면 API 없이 `--dry-run`을 사용합니다. 공고와 이력서 모두 텍스트 파일·PDF·URL을 지원하며, 결과는 `runs/`에 저장됩니다. 로그인이나 JavaScript 렌더링이 필요한 페이지는 파일로 저장해서 입력해야 합니다.

## Evaluation

- 인용한 이력서 근거를 제거한 뒤 판정 변화 측정
- 단일 프롬프트 방식과 비교
- 공고 줄 분류, 판정 정확도, 근거 타당성, 반복 실행 안정성 평가

자세한 방법은 [평가 문서](docs/evaluation.md)를 참고하세요.

## Tests

```bash
python3.11 -m unittest discover -s tests -v
```

현재 자동화 테스트 22개가 통과합니다. 실제 모델 평가 전에는 정확도나 안정성 수치를 주장하지 않습니다.
