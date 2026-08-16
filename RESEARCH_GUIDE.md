# Graph-CLaD 연구 폴더 설명서

Last updated: 2026-08-13

## 1. 연구 목적과 핵심 아이디어

기존 CLaD는 두 시점의 semantic feature와 action history로 transition
representation을 학습하지만 물체별 상태와 물체 간 관계 변화를 명시적으로
표현하지 않는다. 이 연구는 다음 가설을 단계적으로 검사한다.

1. 각 시점의 상태를 object–relation graph로 구성한다.
2. GNN 또는 pair-level encoder로 object/pair token을 만든다.
3. action history를 조건으로 두 graph 사이의 관계 변화를 학습한다.
4. representation이 semantic baseline보다 robot–object interaction과 spatial
   transition을 잘 보존하는지 같은-capacity probe로 비교한다.

Holding은 최종 과제가 아니라 architecture/probe 검사다. Primary 지표는 natural
held-out test의 conditional/oracle-current holding event PR-AUC다. Thresholded F1,
onset/release F1, hard-negative FPR, calibration, action/edge perturbation은 보조
진단이다. Challenge는 natural held-out episode에서 미래 event로 선택한 stress
view이며 독립 generalization test가 아니다.

## 2. 현재 진행 상태

- Phase 0–2D: 완료.
- Legacy Phase 3 controlled experiment: 45 runs 완료.
- Phase3B-R1 reduced cross-fold gate: 36/36 runs 완료.
- Corrected protocol v2: 구현 및 three-fold seed-0 gate 완료.
- Late/global-action G1: B1 pair MLP를 일관되게 이기지 못해 3-seed 확대 중단.
- Pair-local H0–H3: three-fold seed-0 12/12 runs 완료.
- H3 train-action-shuffled alignment control: 2026-08-13 Colab에서 별도 3-run으로
  시작했으며 이 문서 정리 시점에는 실행 중이었다. 결과 폴더를 덮어쓰지 않는다.
- Weak-label audit: 90-item trajectory evidence package 준비 완료. Human decision은
  아직 완료되지 않았으므로 weak label을 ground truth로 표현하지 않는다.
- Phase 4 representation comparison: 아직 시작하지 않음.

최근 pair-local 결과는
`docs/phase3_pair_local_temporal_threefold_seed0_result.md`를 본다. 연구 판단의
시간순 기록은 `docs/research_log.md`가 기준이다.

## 3. 전체 연구 흐름

```text
Phase 0  supplied CLaD baseline 무결성 검사
  -> Phase 1  LIBERO state/API와 runtime 계약 확인
  -> Phase 2A snapshot object–relation graph 추출
  -> Phase 2R scripted diagnostic probe (extractor 진단 전용)
  -> Phase 2D official-demo state replay, temporal weak label, split, dataset QA
  -> Phase 3A/B legacy relational probe와 holder–object architecture screening
  -> Phase 3 corrected protocol: natural validation, frozen threshold, calibration
  -> Pair-local causal history/action H0–H3 factorial
  -> action-alignment + weak-label audit gate
  -> Phase 4 frozen representation comparison (아직 gated)
```

## 4. 폴더 구조와 역할

```text
Graph-CLaD/
├─ baseline_code/       받은 CLaD 핵심 모듈; 직접 수정하지 않음
├─ configs/             versioned 실험 계약과 frozen 경로
├─ data/                작은 fixture와 compact summary만 저장
├─ docs/                설계, 결과, roadmap, 연구 로그
├─ notebooks/           얇은 실행 runbook; 구현 source가 아님
├─ outputs/             로컬 소규모 실행 출력; 대용량 파일은 Git 제외
├─ scripts/
│  ├─ phase0/           baseline smoke
│  ├─ phase1/           LIBERO state/runtime inspection
│  ├─ phase2a/          graph extraction
│  ├─ phase2r/          scripted diagnostic data
│  ├─ phase2d/          official-demo replay/dataset/weak-label pipeline
│  └─ phase3/           manifest, model, training, evaluation, audit
├─ tests/               synthetic/pure-Python/tensor contract tests
└─ archive/             과거 bundle/staging/cache; 현재 import 금지
```

`src/`로 대규모 이전하지 않았다. 현재 `scripts/phase*/`가 이미 phase ownership과
CLI compatibility를 제공하고, 파일 이동은 과거 Colab command와 code snapshot의
재현성을 깨뜨릴 수 있기 때문이다.

## 5. 주요 파일: 입력, 수행 내용, 출력

| 파일 | 입력 | 수행 | 출력 |
|---|---|---|---|
| `baseline_code/LatentDynamics.py` | semantic/action tensor | supplied latent dynamics | model tensor; 수정 금지 |
| `scripts/phase0/smoke.py` | synthetic config | baseline train/eval/EMA 계약 검사 | smoke JSON/console |
| `scripts/phase1/state_inspection.py` | LIBERO suite/task | observation, state, body/contact 조사 | compact capture JSON |
| `scripts/phase1/runtime_compat.py` | installed robosuite file | 알려진 MuJoCo API 차이 검사/선택적 patch | dry-run report 또는 backup+patch |
| `scripts/phase2a/graph_extractor.py` | simulator snapshot + GraphSpec | node/edge graph 추출 | graph snapshot JSON |
| `scripts/phase2d/build_demo_dataset.py` | official HDF5, BDDL, split manifest | state replay와 multi-horizon graph pair 생성 | per-demo shard와 merged gzip JSONL |
| `scripts/phase2d/temporal_holding.py` | 과거 contact/gripper/trajectory | heuristic holding state machine | weak holding relation/evidence |
| `scripts/phase2d/build_holding_target_dataset.py` | natural Phase2D dataset | onset/release/hard-negative target view 생성 | target-aligned dataset + manifest |
| `scripts/phase3/build_eval_manifest.py` | natural/target roots + split | task-local quota, leakage, payload hash QA | corrected eval manifest |
| `scripts/phase3/offline_probe.py` | graph records + config | B1/G1/S0/pair-local 모델 학습·평가 | checkpoint, nested metrics, prediction rows |
| `scripts/phase3/pair_local_temporal.py` | episode graph frames at `<= t` | causal pair history feature 구성 | history vector + validity mask + QA |
| `scripts/phase3/run_corrected_architecture_gate.py` | corrected config + manifest | natural-validation checkpoint/threshold protocol | result JSON, checkpoints, gzip predictions, snapshot |
| `scripts/phase3/analyze_corrected_predictions.py` | per-sample paired predictions | task/episode/event clustered bootstrap | paired differences + confidence interval JSON |
| `scripts/phase3/build_weak_label_audit.py` | graph dataset + labels | balanced 90-item audit evidence 구성 | audit manifest/review CSV/evidence |
| `scripts/phase3/weak_label_audit_viewer.py` | audit package | trajectory evidence interactive review | atomic human decisions + summary |
| `scripts/research_paths.py` | environment variables/explicit roots | local/Colab 경로를 side effect 없이 resolve | preflight JSON/`ResearchPaths` |

Root-level `scripts/phase3_offline_probe.py` 같은 파일은 compatibility wrapper다.
기준 구현은 `scripts/phase3/offline_probe.py`다.

## 6. 권장 노트북 실행 순서

1. 환경 준비: `notebooks/00_environment_and_paths.ipynb`
2. Phase 0: `notebooks/phase_0_clad_baseline_smoke.ipynb`
3. Phase 1A: `notebooks/phase_1a_state_api_audit.ipynb`
4. Phase 2A: `notebooks/phase_2a_static_graph_contract.ipynb`
5. Phase 2R: `notebooks/phase_2r_scripted_diagnostics.ipynb`
6. Phase 2D: `notebooks/phase_2d_official_demo_dataset.ipynb`
7. Phase 3A: `notebooks/phase_3a_dataset_and_label_qa.ipynb`
8. Phase 3B 학습: `notebooks/phase_3b_corrected_architecture_gate.ipynb`
9. Phase 3B 평가: `notebooks/phase_3b_evaluation_and_controls.ipynb`

기존 `notebooks/graph_clad_phase0_to3.ipynb`는 2026-08-07 이전 흐름을 보존하는
legacy 통합 runbook이다. 새 실험에는 위 공식 단계별 노트북을 사용한다. 현재 단계는
Phase 3B이며 Phase 3C, Phase 4, Phase 5 이후 notebook은 gate 통과 전까지 만들지 않는다.

## 7. 데이터에서 결과까지

```text
official LIBERO HDF5 + BDDL
  -> fixed episode split manifest
  -> state replay
  -> graph_t, graph_target, action[t:t+tau]
  -> natural Phase2D gzip JSONL
  -> holding target/stress rows + weak-label evidence
  -> corrected eval manifest
  -> model training with natural-validation checkpoint selection
  -> one frozen natural-validation threshold
  -> natural test + challenge stress + perturbation controls
  -> per-sample gzip predictions
  -> task-first paired report + hierarchical bootstrap
```

Phase 2R scripted trajectories는 extractor/relation 진단용이며 official-demo training
data와 혼합하지 않는다.

## 8. 경로와 환경 설정

새 코드와 노트북은 다음 환경 변수를 지원한다.

- `GRAPH_CLAD_PROJECT_ROOT`: repository root.
- `GRAPH_CLAD_ARTIFACT_ROOT`: 결과/데이터 artifact root.
- `GRAPH_CLAD_LIBERO_ROOT`: LIBERO checkout 또는 asset root.

확인 명령:

```bash
python -m scripts.research_paths
```

로컬 기본 artifact root는 `outputs/`다. Colab 기본은
`/content/drive/MyDrive/Graph-CLaD/artifacts`다. Historical config의 절대 Drive
경로는 결과 재현용으로 유지하며, 로컬에서 사용할 때는 새 config version을 만들거나
CLI override를 사용한다. 기존 config를 현장에서 덮어쓰지 않는다.

Colab `/content`는 임시다. source, config, manifest, checkpoint, prediction,
result는 Drive에 snapshot/version과 함께 남긴다. Drive mount는 노트북의 명시적
환경 셀에서만 수행한다.

### KCloudVPN Linux SSH 실행

앞으로의 학습 실행 기본 환경은 KCloudVPN Linux 서버
`ubuntu@172.10.5.118`로 전환한다. SSH 접속, 가상환경, 입력 artifact 전송,
manifest 재생성, `tmux` 실행은
[docs/kcloudvpn_linux_ssh_runbook_ko.md](docs/kcloudvpn_linux_ssh_runbook_ko.md)를
따른다. 서버에서는 다음 환경 변수를 설정한다.

```bash
export GRAPH_CLAD_PROJECT_ROOT="$HOME/Graph-CLaD"
export GRAPH_CLAD_ARTIFACT_ROOT="/path/to/persistent/Graph-CLaD-artifacts"
export GRAPH_CLAD_LIBERO_ROOT="/path/to/LIBERO"
```

KCloudVPN용 config는 `configs/phase3_kcloudvpn_linux_*.json`이다. Colab의
`/content/drive` 경로를 재사용하지 않고, `GRAPH_CLAD_ARTIFACT_ROOT` 아래에
manifest·자연/target-aligned dataset·split manifest를 둔다. 현재 Colab의 T4는
일시적인 runtime 상태이며 KCloudVPN GPU를 고정 제약으로 가정하지 않는다.
실제 GPU와 VRAM은 매 실행의 `runtime_manifest.json`에 기록한다.

### GPU 확인

현재 연결된 runtime에서는 NVIDIA T4(대개 16GB VRAM)를 사용하고 있다. 이는 현재
runtime의 일시적인 상태이며, 이후 runtime에서 GPU가 바뀔 수 있으므로 고정된 연구
제약으로 해석하지 않는다. 기존 corrected pair-local config의 batch size 64는 현재
모델 규모에서 우선 유지한다. OOM이 발생할 때만 새 config version에서 batch size를
32로 낮추며, batch size를 바꾼 실행은 기존 A100/64 결과와 동일 protocol 결과로 합치지
않는다.

Phase 00 preflight에서 매 runtime마다 실제 GPU 이름, VRAM, CUDA 사용 가능 여부를
기록한다. Mixed precision, gradient accumulation, worker 수 변경은 재현성에 영향을
주므로 corrected model 비교에서는 모든 조건에 동일하게 적용하고 config/result에 남긴다.

## 9. 학습, 평가, 추론

가벼운 테스트:

```bash
python -m unittest discover -s tests
```

Corrected gate 형식:

```bash
python -m scripts.phase3.run_corrected_architecture_gate \
  --config configs/phase3_pair_local_temporal_smoke_v1.json
```

이 config는 persistent Colab 경로를 고정한다. 전체 실행 전에는 반드시 새 output
directory, GPU, Drive mount, manifest SHA, code snapshot을 확인한다. 현재 실행 중인
output에는 같은 command를 다시 실행하지 않는다.

Paired prediction 분석:

```bash
python -m scripts.phase3.analyze_corrected_predictions \
  --result RESULT.json --output ANALYSIS.json \
  --left-model H3-history-action --right-model H1-history
```

현재 별도 deployment inference pipeline은 없다. “Inference”는 held-out graph pair에
대한 `evaluate_model`과 per-sample prediction artifact를 뜻한다. Representation
freeze/linear probe는 Phase 4에서 같은-capacity protocol로 추가해야 한다.

## 10. 설정값과 artifact 위치

- 모델/학습/평가 조건: `configs/*.json`.
- current corrected protocol: `configs/phase3_holder_action_eval_v2_corrected.json`.
- pair-local smoke/three-fold/alignment:
  `configs/phase3_pair_local_temporal_*_v1.json`.
- KCloudVPN Linux 실행용 portable config:
  `configs/phase3_kcloudvpn_linux_eval_manifest_v2.json`,
  `configs/phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json`,
  `configs/phase3_kcloudvpn_linux_pair_local_temporal_threefold_seed0_v1.json`.
- local small outputs: `outputs/{checkpoints,logs,figures,metrics}`; contents Git 제외.
- Colab full artifacts:
  `/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1`.
- corrected artifacts: 위 root의 `corrected_protocol_v2/`.
- 연구 해석: `docs/*_result.md`와 `docs/research_log.md`.

결과 한 세트는 config, manifest, checkpoint, per-sample prediction, aggregate result,
runtime manifest, code snapshot을 함께 가져야 한다.

## 11. 새 실험을 추가하는 방법

1. 기존 result directory를 재사용하지 않고 protocol/config version을 새로 만든다.
2. 바꾸는 가설과 fixed conditions를 plan/result 문서에 먼저 적는다.
3. manifest/sample IDs/split/loss/checkpoint/threshold/capacity를 대조군과 맞춘다.
4. task 1 seed 0 또는 가장 작은 relevant smoke를 먼저 실행한다.
5. natural PR-AUC, release, hard-negative, perturbation gate를 통과할 때만 확대한다.
6. Drive output에 source/config/manifest SHA와 code snapshot을 저장한다.
7. 결과가 불리해도 그대로 `docs/research_log.md`에 기록한다.

## 12. 미완성·주의사항

- Current holding event compatibility metric은 oracle current holding에 조건부다.
  End-to-end current/future event metric을 함께 보고한다.
- H3 action-alignment 결과와 90-item human weak-label review가 아직 gate다.
- `inside`는 valid label support가 없어 deferred다.
- 동일 task의 seeds는 test episode를 공유하므로 독립 표본이 아니다.
- Phase 4 semantic/pair-local/graph representation 비교는 미구현이다.
- LIBERO, robosuite, MuJoCo, HDF5 replay는 로컬 기본 환경에서 검증되지 않을 수 있다.
- `requirements-phase0.txt`는 baseline 최소 의존성만 담는다. Phase 2 runtime은
  LIBERO/robosuite/MuJoCo/h5py, 분석은 numpy/matplotlib, 모델은 PyTorch가 필요하다.

## 13. 이동·변경·보관 상태

이번 정리는 기존 source, config, notebook, artifact를 이동하거나 삭제하지 않았다.
추가된 경로 유틸리티와 phase 노트북이 새 canonical entry point다. Colab에서 실행한
action-alignment config의 exact semantic content를 로컬 config에 반영했다.

| 기존 위치 또는 방식 | 현재 권장 위치 또는 방식 | 처리 상태 |
|---|---|---|
| `notebooks/graph_clad_phase0_to3.ipynb` | 공식 Phase 0/1A/2A/2R/2D/3A/3B notebook | 기존 파일 보존, 새 runbook 추가 |
| notebook 셀의 `/content/...` 직접 지정 | `scripts/research_paths.py`와 환경 변수 | 역사 셀 보존, 신규 실행만 공통화 |
| Colab live action-alignment config | `configs/phase3_pair_local_temporal_action_alignment_seed0_v1.json` | semantic/byte-content 일치, 로컬 끝 newline만 추가 |
| Colab Drive artifact | `/content/drive/MyDrive/Graph-CLaD/artifacts` | 이동·복사 없이 persistent source로 문서화 |
| 로컬 개발 산출물 | `outputs/{checkpoints,logs,figures,metrics,predictions}` | 설명 추가; 실제 하위 폴더는 요청 시 생성 |
| `.tmp_pair_local_sync_*`와 bundle/stage | `docs/archive_candidates_20260813.md` | 현재 위치 유지, 보관 후보만 분류 |

보관 후보와 근거는 `docs/archive_candidates_20260813.md`, 전체 조사와 Colab hash
비교는 `docs/repository_audit_20260813.md`를 본다. 한국어화 범위와 영문 기술 용어 유지
원칙은 `docs/korean_translation_status_20260813.md`에 기록했다. 번역 전 Markdown 원문은
`archive/pre_korean_translation_20260813.zip`에 보존했다.
