# 연구 Phase별 실행 코드

실제 구현의 기준 경로는 아래 Phase 디렉터리다. `scripts/*.py`에 남아 있는 옛
파일명은 기존 테스트와 과거 Colab 명령을 위한 호환 진입점이며 새 코드는 그곳에
추가하지 않는다.

| Phase | 목적 | 기준 코드 |
|---|---|---|
| 0 | 받은 CLaD baseline train/eval/EMA 무결성 검사 | `phase0/smoke.py` |
| 1 | LIBERO state/API 조사, Colab runtime 호환성 | `phase1/state_inspection.py`, `phase1/runtime_compat.py` |
| 2A | 정적 GraphSpec과 snapshot graph 추출 | `phase2a/graph_extractor.py` |
| 2R | scripted diagnostic probe와 relation handler 단위 검증 | `phase2r/` |
| 2D | official demo state replay, temporal holding, split, dataset 생성·QA·보존 | `phase2d/` |
| 3 | dataset loading, task-held-out split, GNN relational-dynamics controls | `phase3/` |
| 4 | CLaD latent foresight 통합 | 아직 구현하지 않음; Phase 3 gate 이후 진행 |

환경별 경로는 `research_paths.py`가 담당한다. Historical config의 절대 Drive 경로는
재현을 위해 유지하고, 새 notebook/experiment는 환경 변수 또는 명시적 CLI path를
사용한다.

## 권장 실행 순서

```text
Phase 0 baseline smoke
  -> Phase 1 state/runtime preflight
  -> Phase 2A graph schema tests
  -> Phase 2D split manifest 고정
  -> Phase 2D official-demo state replay와 dataset 생성
  -> Phase 2D relation/input-leakage QA
  -> Phase 2D target-aligned holding dataset 생성과 sampler preflight
  -> Phase 3 corrected natural-validation protocol
  -> B1/G1/S0/train-shuffled architecture gate
  -> pair-local H0--H3 causal history/action factorial
  -> action-alignment + weak-label audit gate
  -> Phase 4 frozen representation comparison (gate 통과 시)
```

## 주요 명령

저장소 루트에서 모듈 방식으로 실행한다.

```bash
python -m scripts.phase0.smoke --config configs/phase0_synthetic.json
python -m scripts.phase2d.split_manifest --output data/phase2d_demo_split_manifest.json
python -m scripts.phase2d.audit_relations \
  --dataset TASK0.jsonl.gz --dataset TASK1.jsonl.gz --dataset TASK2.jsonl.gz \
  --output phase2d_relation_audit.json
python -m scripts.phase3.dataset_io --help
python -m scripts.phase3.offline_probe --help
python -m scripts.phase2d.build_holding_target_dataset --help
python -m scripts.phase2d.audit_holding_target_dataset --help
python -m scripts.phase3.run_controlled_taskfamily --help
python -m scripts.phase3.analyze_holding_results --help
python -m scripts.research_paths
python -m scripts.phase3.run_corrected_architecture_gate \
  --config configs/phase3_pair_local_temporal_smoke_v1.json
python -m scripts.phase3.analyze_corrected_predictions --help
```

Aligned와 shuffled처럼 result JSON이 서로 다른 artifact root에 있을 때는 두 result와
prediction root를 각각 지정한다. `--left-prediction-root`와
`--right-prediction-root`는 artifact root 또는 `predictions/` 디렉터리일 수 있다.

```bash
python -m scripts.phase3.analyze_corrected_predictions \
  --left-result ALIGNED_RESULT.json \
  --right-result SHUFFLED_RESULT.json \
  --left-prediction-root ALIGNED_ARTIFACT_ROOT \
  --right-prediction-root SHUFFLED_ARTIFACT_ROOT \
  --output ALIGNED_VS_SHUFFLED_ANALYSIS.json \
  --left-model H3-history-action \
  --right-model H3-train-shuffled \
  --replicates 2000 --seed 20260816
```

Phase 2D 전체 변환은 다음 형태다. 실제 HDF5/BDDL/출력 경로는 Colab Drive 위치로
바꾼다.

```bash
python -m scripts.phase2d.build_demo_dataset \
  --task 0=/path/task0.hdf5 \
  --task 1=/path/task1.hdf5 \
  --task 2=/path/task2.hdf5 \
  --split-manifest /path/phase2d_demo_split_manifest.json \
  --bddl-root /content/LIBERO \
  --output-root /content/drive/MyDrive/Graph-CLaD/artifacts/phase2d/data/full_demo_v3
```

이 변환기는 action replay로 label을 만들지 않는다. 각 HDF5 state를 직접 복원하고,
저장 action은 `G_t -> G_(t+tau)`의 conditioning window로만 보존한다. demo별 shard를
먼저 저장하므로 런타임이 끊겨도 같은 명령을 실행하면 완료된 demo는 건너뛴다.

Holding target 실험은 notebook source를 임시로 치환하지 않는다. Dataset 생성 -> sampler
audit -> `run_controlled_taskfamily.py --config ...` -> saved report 분석 순서로 실행한다.
기존 balanced-v3는 `category_aware_v1`, 새 sampling-fix 실험은
`category_aware_episode_round_robin_v2`로 명시한다.

현재 architecture gate의 primary metric은 natural held-out test의
conditional/oracle-current holding event PR-AUC다. Checkpoint와 threshold는 natural
validation에서 선택하며 같은 frozen threshold를 test/stress/control에 적용한다.
Challenge는 stress view이며 독립 test로 해석하지 않는다.
