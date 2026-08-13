# Phase 3

## Durable controlled runner

`run_controlled_taskfamily.py`는 three-fold task-family-held-out 실험의 재사용 runner다.
Mounted Drive의 Phase 2D input-clean file을 읽고 각 fold 뒤 checkpoint를 저장한 다음
full report, config, summary를 쓴다. Notebook에서 긴 experiment cell을 다시 만들지 말고
이 runner를 호출한다.

- `dataset_io.py`: Phase 2D JSONL을 streaming으로 읽고 demo-level split을 보존한다.
- `task_split.py`: task-family-held-out protocol을 만든다.
- `offline_probe.py`: flat/no-message/GNN 계열과 no-action, shuffled-action,
  shuffled-edge controls를 비교한다.
- `sampling.py`: original episode cap, balanced-v3 재현 sampler, 수정된 true
  episode-round-robin sampler를 구분해 제공한다.
- `analyze_holding_results.py`: saved full report에서 holding, action, edge control
  결과와 run-level row를 다시 계산한다.

`offline_probe.py`는 config의 `target_relations`를 읽으므로 `inside`를 제외하기 위해
Colab에서 source string을 치환하지 않는다. `run_controlled_taskfamily.py`도 config의
sampling method와 dataset provenance를 직접 기록한다.

Smoke run은 배선 확인일 뿐 Phase 4 진입 근거가 아니다. 고정 split과 여러 seed의 full
control 결과가 action-conditioned changed-relation prediction을 보여야 한다.

Balanced-v3 결과 재현에는 `configs/phase3_holding_target_balanced_v3.json`을 사용한다.
새 실험은 sampler bug를 수정한
`configs/phase3_holding_target_balanced_v4_samplingfix.json`을 사용하되, 두 결과를 같은
protocol로 합치지 않는다.
