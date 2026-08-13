# Graph-CLaD.ipynb 최근 셀 이관 기록 — 2026-08-07

Colab notebook의 셀 295~304를 읽어 source-of-truth 코드와 일회성 셀을 분류했다. Notebook은 실행 이력이며 재사용 구현은 `scripts/`, 조건은 `configs/`, 결과 요약은 `data/`에 둔다.

| Colab 셀 | 역할 | 로컬 처리 |
|---:|---|---|
| 295 | 초기 holding event-window controlled run | target-aligned dataset으로 대체되어 별도 runner로 승격하지 않음 |
| 296 | 초기 event-window 결과 분석 | historical result로 연구노트에만 유지 |
| 297 | target-aligned holding dataset 생성 | `scripts/phase2d/build_holding_target_dataset.py` |
| 298 | episode-cap preflight | 셀 299와 통합 |
| 299 | 셀 298의 축약 중복 | `scripts/phase2d/audit_holding_target_dataset.py`로 통합 |
| 300 | target-aligned controlled experiment 실행 | `scripts/phase3/run_controlled_taskfamily.py` 일반화 + config 파일 |
| 301 | target-aligned report 분석 | `scripts/phase3/analyze_holding_results.py` |
| 302 | category-aware sampler patch와 balanced-v3 run | `scripts/phase3/sampling.py` + `configs/phase3_holding_target_balanced_v3.json` |
| 303 | saved report key 구조 확인용 debug | 재사용 코드로 승격하지 않음 |
| 304 | 셀 302 후반 analysis bug 수정 | `scripts/phase3/analyze_holding_results.py`로 통합 |

## 중요한 재현성 메모

- Colab balanced-v3의 category sampler는 quota stage에서 true episode round-robin이 아니었다.
- 기존 artifact를 재현하는 함수는 `category_aware_cap_v1`이다.
- 새 실험용 수정 함수는 `category_aware_episode_round_robin_cap`이며 config method는 `category_aware_episode_round_robin_v2`다.
- 두 sampler 결과는 같은 실험으로 합치면 안 된다.
- `inside`를 제거하기 위해 Colab에서 Python source string을 직접 치환한 방식은 폐기했다. 이제 `offline_probe.py`가 config의 `target_relations`를 직접 읽는다.
- report analysis의 `evaluations`/`evaluations["correct"]` 혼동도 독립 analyzer에서 제거했다.

## 결과 파일

- `data/phase2d_holding_target_summary.json`
- `data/phase3_holding_target_v2_summary.json`
- `data/phase3_holding_target_balanced_v3_summary.json`

Drive의 full report와 dataset이 원본 artifact이며 로컬 JSON은 검토와 재현을 위한 compact summary다.
