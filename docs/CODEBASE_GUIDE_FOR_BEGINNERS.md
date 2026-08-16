# 처음 보는 사람을 위한 Graph-CLaD 코드 설명서

이 문서는 “어느 파일을 열어야 하고, 그 파일이 무엇을 입력받아 무엇을 생성하는가”를
설명한다. 연구 질문과 Phase의 이유는 `RESEARCH_WORKFLOW_FOR_BEGINNERS.md`, 현재
서버 실행 상태는 `CURRENT_STATUS.md`를 먼저 본다.

## 1. 코드 구조의 기본 원칙

이 저장소의 핵심 원칙은 세 가지다.

1. 재사용 가능한 구현은 `scripts/phase*/`에 둔다.
2. Notebook은 긴 구현을 복사하지 않고 module을 호출하는 실행 runbook으로 사용한다.
3. 대용량 dataset/checkpoint/result는 Git에 넣지 않고 config, manifest, SHA,
   code snapshot으로 식별한다.

루트의 일부 `scripts/*.py`는 과거 Colab command와 import를 보존하기 위한 wrapper다.
새 기능은 wrapper가 아니라 해당 `scripts/phase*/`에 추가한다.

## 2. 최상위 폴더

| 경로 | 역할 | 주의사항 |
|---|---|---|
| `baseline_code/` | 제공받은 CLaD Stage 1 core | 비교 통제를 위해 직접 수정하지 않음 |
| `configs/` | 모델, split, loss, output, runtime 조건 | 기존 config를 덮어쓰지 않고 version 추가 |
| `data/` | 작은 fixture와 compact summary | 대용량 demo를 넣지 않음 |
| `docs/` | roadmap, 설계, 결과, 연구 로그, 인계 | 현재 상태는 `CURRENT_STATUS.md` |
| `notebooks/` | Phase별 실행 순서와 결과 확인 | source of truth가 아님 |
| `outputs/` | 가벼운 로컬 산출물 기본 위치 | 대용량 canonical artifact 위치가 아님 |
| `scripts/` | 실제 데이터 처리, 모델, 학습, 평가 코드 | `scripts/phase*/`가 기준 |
| `tests/` | synthetic/pure-Python/tensor 계약 검사 | 전체 suite는 PyTorch 등 의존성 필요 |
| `archive/` | 과거 bundle, staging, cache | 현재 코드에서 import 금지 |

## 3. 경로를 결정하는 코드

### `scripts/research_paths.py`

로컬, Colab, KCloudVPN에서 같은 코드가 다른 저장 경로를 사용할 수 있도록 환경 변수를
해석한다.

- 입력: `GRAPH_CLAD_PROJECT_ROOT`, `GRAPH_CLAD_ARTIFACT_ROOT`,
  `GRAPH_CLAD_LIBERO_ROOT`.
- 수행: 경로 resolve, Colab 여부 판정, preflight, 선택적 output layout 생성.
- 출력: `ResearchPaths` 객체 또는 JSON preflight.

중요 함수:

- `resolve_research_paths`: 환경에 맞는 project/artifact/LIBERO 경로를 계산한다.
- `preflight`: 경로 존재 여부와 환경 정보를 보고한다.
- `mount_colab_drive`: 명시적으로 호출했을 때만 Drive를 mount한다.
- `ensure_local_output_layout`: 요청했을 때만 로컬 output 하위 폴더를 만든다.

## 4. 제공된 CLaD baseline core

### `baseline_code/LatentDynamics.py`

CLaD Stage 1 latent dynamics의 핵심 module이다.

입력:

- `v_history`: 이전/현재 multi-view semantic feature.
- `p_history`: 이전/현재 proprioception.
- `prev_action`: action history 또는 packed action feature.
- `lang`: language feature.
- 학습 시 `p_next`, `v_next`: future target.

처리:

1. Proprioception, semantic observation, action을 각각 backbone으로 encoding.
2. Previous state와 action을 key/value로, current state를 query로 cross-attention.
3. Proprioceptive transition과 semantic transition을 다시 결합.
4. Learnable query로 하나의 dynamics vector를 pooling.
5. Future proprio/semantic latent와 reconstruction을 예측.

출력:

- 학습: `loss_p`, `loss_s`, `loss_p_recon`, `loss_v_recon` dictionary.
- 평가: `pred_p_emb`, `pred_s_emb`.

Target backbone은 EMA copy다. `update_ema()`는 module이 자동 호출하지 않으므로 trainer가
호출해야 한다. 제공 저장소에는 공식 trainer가 없다.

### `baseline_code/Attentions.py`

`CrossAttnLayer`, `CrossAttnBlock`, `SelfAttnLayer`, `SelfAttnBlock`을 정의한다.
Multi-head attention, residual connection, layer normalization, feed-forward block을
여러 층 쌓는다. `LatentDynamics`의 transition encoding에 사용된다.

### `baseline_code/MlpResNet.py`

Proprioception/action encoder용 residual MLP와 language-conditioned FiLM MLP를 제공한다.
FiLM은 condition으로 scale과 bias를 만들어 feature를 조절한다.

## 5. Phase 0 코드

### `scripts/phase0/smoke.py`

- 입력: `configs/phase0_synthetic.json` 또는 작은 config.
- 수행: 작은 synthetic tensor로 `LatentDynamics` forward/backward, loss shape, gradient,
  EMA update, evaluation output을 검사한다.
- 출력: console/JSON smoke report.

이 단계는 공식 CLaD 성능 재현이 아니라 제공 core가 실행 가능한지 확인하는 계약 test다.

## 6. Phase 1 코드

### `scripts/phase1/state_inspection.py`

- 입력: LIBERO suite/task, environment observation.
- 수행: observation schema, robot joint/gripper/EEF state, object logical ID와 pose,
  contact/body mapping, simulator metadata를 수집한다.
- 출력: compact capture JSON 또는 frame snapshot.

주요 함수:

- `observation_schema`: key별 shape/dtype/preview를 기록한다.
- `extract_robot_state`: EEF, gripper, joint 관련 값을 표준 dictionary로 만든다.
- `extract_object_states`: environment object와 body 정보를 logical object record로 만든다.
- `make_snapshot`: 한 frame에 필요한 robot/object/contact 정보를 묶는다.

### `scripts/phase1/runtime_compat.py`

LIBERO/robosuite/MuJoCo version 차이를 검사한다. Known mass-matrix API 차이는 dry-run이
기본이고 `--apply`일 때만 installed package를 backup 후 patch한다.

## 7. Phase 2A graph 코드

### `scripts/phase2a/graph_extractor.py`

- 입력: 한 frame의 robot/object snapshot.
- 수행: deterministic node와 directed spatial edge를 만든다.
- 출력: `nodes`, `edges`, graph audit를 가진 JSON graph.

현재 node feature vector는 24차원이며 position, object-of-interest flag, gripper qpos,
joint position/velocity와 각 validity를 포함한다. 누락값은 zero vector와 validity 0으로
표시한다.

`extract_graph_snapshot`은 robot node와 object/fixture/site node를 만들고 위치가 유효한
모든 서로 다른 node pair에 directed edge를 만든다. Node identity는 runtime body index가
아니라 stable logical ID다.

`build_graph_sequence`는 여러 snapshot을 step 순서의 graph sequence로 변환한다.

## 8. Phase 2R scripted diagnostic 코드

### `scripts/phase2r/collect_libero_trajectory.py`

간단한 scripted action을 environment에 적용하고 frame별 state/contact를 수집한다.

### `scripts/phase2r/relation_handlers.py`

Distance/contact/spatial relation을 계산하고 directed edge relation record를 만든다.

### `scripts/phase2r/build_dataset.py`

Scripted graph sequence를 transition sample로 바꾼다.

### `validate_pilot.py`, `validate_scaleup.py`, `validate_semantics_frame.py`

Sample 수, relation support, frame semantics, leakage 같은 진단을 수행한다. Phase 2R은
extractor 검사용이고 official-demo training과 분리한다.

## 9. Phase 2D official-demo 데이터 코드

### `scripts/phase2d/state_replay.py`

- 입력: LIBERO HDF5 demo state, BDDL path.
- 수행: environment 생성, simulator state 복원, observation/state error 확인.
- 출력: frame별 snapshot과 replay QA.

### `scripts/phase2d/split_manifest.py`

Demo/episode ID를 train, validation, test로 고정한다. Window를 생성하기 전에 split을
고정해 episode leakage를 막는다.

### `scripts/phase2d/temporal_holding.py`

- 입력: 한 episode의 frame snapshot sequence.
- 수행: contact, gripper closure, 최근 3-frame relative pose stability, object following을
  이용해 object별 holding evidence와 temporal state를 계산한다.
- 출력: `holding` relation, evidence, state transition 기록.

`HoldingPolicy`에 history 길이와 threshold가 모여 있다. 이 label은 heuristic weak
label이므로 human audit 대상이다.

### `scripts/phase2d/build_demo_dataset.py`

Phase 2D의 main release builder다.

1. Task별 HDF5 demo key를 읽는다.
2. Fixed split을 찾는다.
3. Demo state를 replay한다.
4. Frame별 graph와 holding label을 만든다.
5. 여러 horizon의 `graph_t/action_window/graph_target` sample을 만든다.
6. Demo별 atomic shard를 저장한다.
7. QA를 통과한 shard를 gzip JSONL release로 merge한다.

중단 후 resume할 수 있도록 demo shard 단위 persistence를 사용한다.

### `scripts/phase2d/input_clean.py`

미래 target을 암시할 수 있는 object-of-interest 등 금지 feature를 model input에서
제거하고 input-clean release를 만든다.

### `scripts/phase2d/event_windows.py`

Onset/release 등 event 주변 window를 계산하는 공통 함수다.

### `build_holding_event_dataset.py`

Holding event 중심 subset을 만든 초기 도구다. 현재 corrected 흐름에서는 natural과
target-aligned view의 구분을 함께 고려한다.

### `build_holding_target_dataset.py`

- 입력: natural Phase2D dataset.
- 수행: holding changed, future holding positive, hard negative, background category를
  판정하고 target-aligned sample을 선택한다.
- 출력: target-aligned gzip JSONL과 summary/manifest.

### `audit_relations.py`, `audit_holding_target_dataset.py`

Relation support, category count, split, sample consistency, QA status를 검사한다.

## 10. Phase 3 dataset, manifest, sampling 코드

### `scripts/phase3/dataset_io.py`

JSON, JSONL, gzip JSONL 형식을 streaming read하고 sample record를 표준화한다. 작은
smoke dataset도 만든다.

### `scripts/phase3/sampling.py`

Training cap 안에서 category와 episode diversity를 맞춘다. 현재 기준은
`category_aware_episode_round_robin_v2`다. 같은 episode에서 너무 많은 sample이 먼저
선택되는 것을 막고 category quota를 만족시키려 한다.

### `scripts/phase3/build_eval_manifest.py`

- 입력: natural/target root, split manifest, relation/category config.
- 수행: three task folds 생성, train sampling, natural validation/test와 stress view 고정,
  episode leakage와 overlap payload hash QA.
- 출력: corrected evaluation manifest JSON.

현재 구현의 `_read_task_samples`와 `load_samples`는 모든 graph payload를 list에 담는다.
압축 데이터가 큰 KCloudVPN 실행에서는 memory OOM으로 추정되는 `Killed`가 발생했다.
현재 실험은 기존 Colab `status=pass` manifest의 source root만 바꾼 portable copy를
사용한다. Streaming builder 개선은 아직 TODO다.

## 11. Phase 3 causal history 코드

### `scripts/phase3/pair_local_temporal.py`

미래 frame을 읽지 않고 같은 episode의 `[max(0,t-3), t]` graph만 불러와 각 directed
pair에 16차원 history feature를 붙인다.

Feature 구성:

- 3D relative-position delta + validity.
- 3D relative velocity + validity.
- Contact fraction, trailing contact fraction + validity.
- Gripper closure velocity + validity.
- Object-following residual mean/max + validity.

`attach_causal_pair_history`는 원 sample의 `graph_t` edge feature에 history vector를
추가하고 requested/available step과 `future_frame_reads=0` QA를 반환한다.

## 12. Phase 3 모델과 학습 코드

### `scripts/phase3/offline_probe.py`

Phase 3의 모델, batch collator, 학습, threshold 선택, 평가 metric이 모여 있는 핵심 파일이다.

#### `ProbeShape`

Dataset에서 max node/edge 수, node/edge/action/relation dimension, action step layout,
history dimension을 기록한다.

#### `ProbeCollator`

크기가 다른 graph를 batch tensor로 padding한다. Node mask, edge mask, source/target index,
geometry/history, action, relation label/validity, sample metadata를 묶는다.

#### `StructuredActionEncoder`

`6 steps × 7 dims` 같은 action chunk의 시간 구조를 보존한다. Arm channel과 gripper
channel을 따로 encoding하고 step position embedding/attention과 first, last, mean, sum,
delta, gripper min/max, coverage summary를 결합한다.

#### `RelationalDynamicsProbe`

`model_id`에 따라 여러 architecture를 한 class에서 구성한다.

- `b1_pair_feature_mlp_v2`: source/target node, action, pair geometry를 독립 MLP로 처리.
- `g1_sparse_holder_object_gnn`: robot–object sparse message passing 후 action을 late 입력.
- `s0_g1_no_action_holder_object_gnn_v2`: G1에서 action 입력 제거.
- H0: current pair state만 사용.
- H1: current pair state + causal history.
- H2: current pair state를 action FiLM으로 modulation.
- H3: pair state + causal history를 action FiLM으로 modulation.

H0–H3는 object–object global message passing 없이 pair별로 독립 처리한다. Action model은
pair token, action token, element-wise product를 이용해 gamma, beta, gate를 만들고 pair
representation을 조절한다.

Future head와 current auxiliary head가 따로 있다. Corrected protocol의
`current_head_contract=action_free_pair`에서는 current head가 action/history target shortcut을
사용하지 않는 공통 pair representation을 받는다.

#### `train_one`

Train-only normalization과 positive weight를 사용해 masked BCE를 최적화하고 natural
validation PR-AUC를 기준으로 checkpoint를 고른다. `training_action_mode`가
`episode_disjoint_matched`이면 다른 episode의 matched donor action을 사용한다.

#### `evaluate_model`

Correct/shuffled/no-history/edge perturbation mode별 probability를 수집하고 relation metric,
holding conditional event, end-to-end event, onset/release, hard-negative, calibration을
계산한다.

#### Threshold 함수

`_select_holding_threshold`는 natural validation prediction에서 threshold를 선택한다.
선택된 하나의 threshold를 test와 stress/control에 고정한다.

## 13. Corrected architecture runner

### `scripts/phase3/run_corrected_architecture_gate.py`

Config 하나를 받아 전체 학습과 artifact 저장을 수행하는 현재 main entry point다.

실행 순서:

```text
config 읽기와 ${ENV_VAR} 확장
  -> output이 persistent root 안인지 검사
  -> manifest status/split/hash QA 검사
  -> fold별 sample key로 natural/target payload 읽기
  -> 필요하면 t 이하 causal history 재구성
  -> model별 hidden dimension과 parameter count 맞추기
  -> fold × seed × model 학습
  -> natural validation에서 checkpoint와 threshold 선택
  -> natural test, stress, perturbation 평가
  -> checkpoint/prediction/partial result 저장
  -> runtime manifest와 final result 저장
```

중간 run이 끝날 때마다 partial result를 쓰므로 긴 실행 중에도 진행 상황을 확인할 수 있다.

생성 artifact:

- `phase3_*.json`: aggregate result와 summary.
- `runtime_manifest.json`: GPU, config/manifest SHA, path, causal history QA.
- `checkpoints/*.pt`: fold/seed/model checkpoint.
- `predictions/*.jsonl.gz`: per-sample probability/target/metadata.
- `code_snapshot/`: 실행 시 사용한 source/config/manifest copy와 SHA.

## 14. Phase 3 분석과 weak-label 코드

### `scripts/phase3/analyze_corrected_predictions.py`

두 모델의 per-sample prediction을 stable sample ID로 pair한다. Fold/seed difference를
계산하고 task fold → episode → event cluster 순서의 hierarchical bootstrap confidence
interval을 만든다. Seeds를 독립 test unit으로 세지 않는다.

현재 action-alignment runner는 shuffled H3만 담은 별도 result를 만든다. 기존 aligned
H3와 분석하려면 두 실행의 prediction을 같은 분석 입력 형식으로 모으는 adapter 또는
통합 result가 추가로 필요할 수 있다.

### `scripts/phase3/build_weak_label_audit.py`

Task 0/1/2 × onset/release/hard-negative별 candidate를 cluster하고 episode round-robin으로
각 10개를 선택한다. Graph digest, trajectory point, contact/gripper/follow evidence를 담은
90-item audit package를 생성한다.

### `scripts/phase3/weak_label_audit_viewer.py`

Audit item을 사람이 순서대로 보고 pass/fail/error type을 기록하는 viewer다. Review
decision은 원자적으로 저장하고 summary를 갱신한다.

### `scripts/phase3/analyze_ai_assisted_weak_labels.py`

자동 evidence consistency group을 사후 분석한다. Human ground truth를 대신하지 않으며
label 변경이나 checkpoint 선택에 사용하지 않는다.

## 15. Legacy와 이전 runner

다음 파일은 현재 결론의 근거가 된 과거 protocol을 재현하기 위해 남아 있다.

- `run_holder_action_smoke.py`: fixed-manifest holder/action smoke.
- `run_topology_action_followup.py`: sparse/complete topology와 action 구조 비교.
- `run_action_semantics_gate.py`: action semantics control.
- `run_global_action_shuffle_control.py`: evaluation-time shuffle.
- `run_reduced_crossfold_gate.py`: B1/G1/S0/train-shuffled의 3 folds × 3 seeds gate.
- `run_controlled_taskfamily.py`: 초기 controlled task-family experiment.

새 action-alignment 실험에는 `run_corrected_architecture_gate.py`를 사용한다.

## 16. Config를 읽는 법

JSON config는 코드 밖에 실험 조건을 고정한다.

주요 section:

- `protocol`: 결과를 구분하는 versioned ID.
- `manifest`: frozen sample/split 계약 경로.
- `folds`, `seeds`, `models`: 실행 조합.
- `preprocessing`: causal history, topology, 미래 입력 금지 조건.
- `training`: epoch, patience, batch, learning rate, loss/head/checkpoint 계약.
- `parameter_matching`: 비교 모델 parameter 수 맞춤.
- `evaluation`: primary/secondary metric, threshold source, control mode, claim limit.
- `artifacts`: output root와 파일명.
- `runtime`: CUDA와 persistent output 요구사항.

기존 결과와 다른 조건을 시도할 때는 config를 현장에서 수정하지 않고 새 version의 JSON을
만든다. Output root도 새 경로를 사용한다.

## 17. Manifest, result, prediction의 차이

### Manifest

학습 전에 생성된다. 어떤 sample을 어느 role에 쓸지와 QA 결과를 담는다. Model output은
없다.

### Aggregate result JSON

Model/fold/seed별 checkpoint, threshold, natural/stress/control metric과 summary를 담는다.
`status=completed`인지 확인해야 한다.

### Per-sample prediction

각 sample의 probability, target, episode/task ID, event type 등을 담는다. Paired difference,
bootstrap, calibration, failure analysis에 필요하다.

### Runtime manifest

실행 환경과 재현 정보를 담는다. GPU 이름, config/manifest SHA, output path, code snapshot,
causal-history QA를 확인한다.

## 18. Notebook의 역할

`notebooks/README.md`에 공식 순서가 있다. Notebook은 다음만 담당한다.

- 환경과 경로 확인.
- Config 선택.
- Python module 호출.
- 작은 결과 요약과 artifact 존재 검사.
- 다음 Phase 안내.

긴 model class나 data builder를 notebook 셀에 새로 복사하지 않는다. 재사용 코드는
`scripts/phase*/`에 추가하고 notebook에서 import한다.

## 19. Tests의 역할

주요 test 범주:

- `test_phase0_*`: CLaD core shape/loss/EMA.
- `test_phase1_*`: state extraction과 runtime inspection.
- `test_phase2_graph_extractor.py`: node/edge/validity graph 계약.
- `test_phase2d_*`: event window, holding target, recovered pipeline.
- `test_phase3_eval_manifest.py`: fold/quota/leakage/hash QA.
- `test_phase3_holder_object_features.py`: model input/parameter/metric 계약.
- `test_phase3_pair_local_temporal.py`: t 이하 history와 16-D feature.
- `test_phase3_corrected_*`: threshold, artifact, hierarchical analysis.
- `test_notebook_structure.py`: notebook header/input/output/next-phase 구조.
- `test_kcloudvpn_config.py`: portable environment path와 persistent output 계약.

Local environment에 PyTorch/LIBERO가 없으면 pure-Python test만 먼저 실행하고, 실행하지
못한 test를 성공으로 보고하지 않는다.

## 20. 향후 Stage 1/Stage 2 코드는 아직 어디에 있는가

현재 `baseline_code/`에는 Stage 1 core만 있고 완전한 trainer는 없다. Phase 3C, Phase 4,
Stage 2 Diffusion Policy와 rollout evaluator는 아직 구현되지 않았다.

구현할 때의 권장 ownership:

```text
scripts/phase3c/   no-future-action foresight bridge와 frozen probe
scripts/phase4/    semantic/pair-local/graph Stage 1 adapters와 trainer
scripts/phase5/    canonical DDPM Diffusion Policy와 policy-only/semantic baseline
scripts/phase6/    selected Graph-CLaD conditioning adapter
scripts/phase7/    LIBERO rollout와 success evaluator
```

실제 디렉터리는 gate 통과 후 plan/config와 함께 만든다. 아직 존재하지 않는 module을
문서에서 구현 완료로 표현하지 않는다.

## 21. 새 model을 추가할 때

1. 연구 가설과 대조군을 plan 문서에 쓴다.
2. `model_id`와 parameter matching 기준을 새 config version에 추가한다.
3. 필요 최소 변경을 `scripts/phase3/offline_probe.py` 또는 새 Phase module에 구현한다.
4. Synthetic shape/parameter/action perturbation test를 추가한다.
5. 기존 manifest와 동일 sample ID/split/threshold protocol을 사용한다.
6. 가장 작은 fold/seed smoke를 먼저 실행한다.
7. Result, checkpoint, prediction, runtime manifest, code snapshot을 함께 저장한다.
8. 불리한 결과도 `research_log.md`에 기록한다.

## 22. 현재 알려진 코드 주의사항

- `build_eval_manifest.py`는 큰 dataset에서 memory-safe하지 않다.
- Root compatibility wrapper와 phase source를 동시에 수정하면 두 구현이 갈라질 수 있다.
- Historical Colab config에는 `/content/drive` 절대경로가 있다.
- `requirements-phase0.txt`는 전체 LIBERO/Stage 2 환경을 설치하지 않는다.
- Conditional holding event metric은 oracle current state를 사용한다.
- Action-alignment result와 aligned H3 result는 자동으로 한 파일에 합쳐지지 않는다.
- Official Stage 2 network/noise schedule/rollout 세부사항은 확인되지 않았다.

## 23. 처음 코드를 읽는 권장 순서

1. `scripts/research_paths.py` — 경로와 실행 환경.
2. `scripts/phase2a/graph_extractor.py` — graph schema.
3. `scripts/phase2d/temporal_holding.py` — weak label의 의미.
4. `scripts/phase3/pair_local_temporal.py` — causal history feature.
5. `scripts/phase3/offline_probe.py`의 `ProbeShape`, `StructuredActionEncoder`,
   `RelationalDynamicsProbe`, `train_one`, `evaluate_model`.
6. `scripts/phase3/run_corrected_architecture_gate.py` — 전체 orchestration.
7. `scripts/phase3/analyze_corrected_predictions.py` — 통계 분석.
8. `baseline_code/LatentDynamics.py` — 향후 Stage 1 통합 대상.

