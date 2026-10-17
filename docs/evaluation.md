# Evaluation

## 1. Evidence-removal mutation

완료된 보고서가 인용한 이력서 줄을 제거한 뒤 같은 공고를 다시 평가한다.

```bash
python3.11 -m jobfit mutation-eval \
  --posting examples/posting.txt \
  --resume examples/resume.md \
  --baseline-report runs/run-ORIGINAL/report.json \
  --limit 5
```

대상 판정 비상승률·하락률, 변형 후 같은 요건을 다시 찾은 비율, 대상 외 판정 변화율을 기록한다. 요건을 다시 찾지 못한 실행은 성공으로 세지 않고 추출 불안정성으로 집계한다.

## 2. One-shot baseline

도구 없이 공고와 이력서를 한 번에 입력하는 비교 대상을 실행한다. 출력에는 에이전트와 동일한 줄 ID 검증과 점수 계산을 적용한다.

```bash
python3.11 -m jobfit baseline \
  --posting examples/posting.txt \
  --resume examples/resume.md
```

## 3. Manual labels and repeated runs

`evaluation/labels.example.csv` 형식으로 공고 줄별 `required / preferred / non_requirement` 정답과 요건별 판정을 기록한다. 근거 타당성은 `evaluation/evidence_labels.example.csv` 형식으로 실행 ID와 실제 인용 줄 묶음에 연결한다.

```bash
python3.11 -m jobfit evaluate \
  --labels evaluation/labels.csv \
  --evidence-labels evaluation/evidence_labels.csv \
  --baseline-report runs/baseline-1/report.json \
  runs/agent-1/report.json runs/agent-2/report.json runs/agent-3/report.json
```

측정 항목:

- 공고 줄 분류 정확도와 실제 요건 recall·오제외율
- 누락을 오답에 포함한 판정 정확도
- 실행별 근거 타당성
- 공고 줄별 분류·판정·요건 묶음 안정성
- 종합 핏 점수의 실행 간 표준편차
- 도구 에이전트와 단일 프롬프트 baseline의 차이

반복 실행에는 URL을 다시 읽지 않고 동일한 공고 텍스트 파일을 사용한다.
