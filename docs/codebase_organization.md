# Graph-CLaD 코드 보존 및 분류 기준

## Source of truth

- `baseline_code/`: 원 연구자에게 받은 세 baseline 모듈. 연구 비교의 통제를 위해
  직접 수정하지 않는다.
- `scripts/phase*/`: 현재 재사용 가능한 구현의 기준 경로.
- `scripts/*.py`: 과거 import/명령을 깨뜨리지 않기 위한 compatibility alias.
- `configs/`: 실행 설정과 고정 실험 조건.
- `tests/`: LIBERO 없이 가능한 계약 검사와 synthetic smoke.
- `data/`: 작은 fixture와 재현 가능한 요약만 보관. 대용량 demo artifact는 Drive에
  두고 manifest/checksum으로 식별한다.
- `notebooks/`: 실행 순서만 담는 runbook. 핵심 구현은 셀에 두지 않는다.
- `outputs/`: 가벼운 로컬 실행의 기본 출력. 대용량 artifact는 Git에서 제외한다.
- `archive/`: 과거 bundle, staging copy, 임시 실행 기록. 현재 코드에서 import하지
  않는다.

## Phase ownership

```text
Phase 0   scripts/phase0/smoke.py + baseline_code/
Phase 1   scripts/phase1/state_inspection.py
          scripts/phase1/runtime_compat.py
Phase 2A  scripts/phase2a/graph_extractor.py
Phase 2R  scripts/phase2r/*                 (diagnostic only)
Phase 2D  scripts/phase2d/state_replay.py
          scripts/phase2d/temporal_holding.py
          scripts/phase2d/split_manifest.py
          scripts/phase2d/build_demo_dataset.py
          scripts/phase2d/input_clean.py
          scripts/phase2d/audit_relations.py
          scripts/phase2d/persistence.py
          scripts/phase2d/build_holding_event_dataset.py
          scripts/phase2d/build_holding_target_dataset.py
          scripts/phase2d/audit_holding_target_dataset.py
Phase 3   scripts/phase3/dataset_io.py
          scripts/phase3/task_split.py
          scripts/phase3/offline_probe.py
          scripts/phase3/sampling.py
          scripts/phase3/run_controlled_taskfamily.py
          scripts/phase3/analyze_holding_results.py
          scripts/phase3/build_eval_manifest.py
          scripts/phase3/run_corrected_architecture_gate.py
          scripts/phase3/pair_local_temporal.py
          scripts/phase3/analyze_corrected_predictions.py
          scripts/phase3/build_weak_label_audit.py
Phase 4   미구현; Phase 3 control gate 이후에만 시작
```

공통 로컬/Colab 경로 해석은 `scripts/research_paths.py`가 담당한다. 기존 snapshot과
config의 절대 경로는 바꾸지 않고, 새 실행만 환경 변수 또는 explicit CLI override를
사용한다.

## 이번 복구에서 바로잡은 Colab 전용 코드 문제

1. 전체 demo 생성 셀 앞에 미완성 검증 loop와 다른 셀 조각이 붙어 있어 원문은
   standalone Python 파일로 실행할 수 없었다. 기능별 모듈로 다시 작성했다.
2. 최초 split lookup은 manifest의 `demo_key="demo_N"` 대신 정수 demo ID를 사용해
   null split을 만들었다. 새 생성기는 처음부터 `(task_id, demo_key)` 계약을 쓴다.
3. holding은 모든 object에 대해 contact, closed gripper, relative-pose stability,
   object-followed-EFF evidence를 기록하고 `free -> contact_candidate -> holding ->
   release` 상태 전이를 만든다.
4. `robot0_*`뿐 아니라 `gripper0_*` body contact도 `robot0` node에 매핑하는 기존
   보정을 기준 collector에 유지했다.
5. `is_object_of_interest`는 label 생성·감사에는 사용할 수 있지만 `graph_t`와
   `graph_target` 입력에서는 제거한다.
6. 전체 생성은 demo별 원자적 shard와 resume를 사용하고, 최종 merge 및 QA가 끝난
   결과만 release로 취급한다.
7. Colab의 최근 target-aligned 생성·sampling·analysis 셀은
   `docs/colab_code_migration_20260807.md`의 대응표에 따라 모듈로 이관했다. 중복 QA와
   일회성 debug 셀은 source-of-truth로 승격하지 않았다.
8. balanced-v3의 category sampler는 재현용 `category_aware_v1`로 동결하고, 수정
   sampler는 `category_aware_episode_round_robin_v2`로 별도 버전 관리한다.

## 호환성과 보존

기존 `from scripts.graph_extractor import ...` 형태는 계속 동작한다. 새 코드와 문서는
`from scripts.phase2a.graph_extractor import ...` 같은 기준 경로를 사용한다. Colab
runtime package patch는 `phase1/runtime_compat.py`에서 dry-run이 기본이며 `--apply`
시에만 site-package를 수정하고 백업을 남긴다.
