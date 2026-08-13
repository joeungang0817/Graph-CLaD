# Graph-CLaD 수정 연구 로드맵 v3

기준일: 2026-08-13  
현재 공식 단계: **Phase 3B-R1 — corrected architecture/action gate**  
원 영문 혼합본: `archive/pre_korean_translation_20260813.zip`

## 1. 개정 핵심

원래 질문은 CLaD의 semantic transition을 object–relation graph transition으로 구조화하면
robot–object interaction과 object spatial transition을 더 명시적으로 보존할 수 있는가다.
초기 scripted dataset과 loss 감소만으로는 이 질문을 검증할 수 없었다. v3는 다음을
고정한다.

1. Main data는 LIBERO official demonstration의 exact state replay에서 만든다.
2. Episode split은 graph/event window 생성 전에 고정한다.
3. Future 정보로 선택한 stress view는 독립 test로 부르지 않는다.
4. Holding onset/release는 최종 task가 아니라 representation/architecture probe다.
5. Natural event PR-AUC, onset/release F1, hard-negative FPR, calibration을 함께 본다.
6. Graph 모델은 near/exact capacity-matched non-graph pair baseline과 비교한다.
7. Action/edge shuffle은 donor provenance와 distance를 검사하는 강한 control이어야 한다.
8. Phase 4는 Phase 3B와 Phase 3C gate를 통과할 때까지 차단한다.

Phase 0, Phase 1A, Phase 2A의 기반은 유지한다. Phase 2R scripted dataset과 초기 Phase 3
결과는 diagnostic history로 보존하지만 최종 주장에는 사용하지 않는다.

## 2. 연구 질문과 주장 범위

### 주 연구 질문

같은 data, split, action, capacity, evaluation protocol에서 다음 representation 중 어느
것이 relation transition을 더 잘 보존하는가?

- 기존 CLaD semantic-transition representation.
- Robot–object pair-local temporal representation.
- Object–relation graph-transition representation.

### 허용되는 주장

- 같은 controlled environment 안에서 paired improvement를 보고한다.
- Natural held-out task fold가 primary다.
- Task가 세 개뿐이므로 넓은 task-family generalization을 과장하지 않는다.
- 동일 task의 seeds는 같은 test episode를 공유하므로 독립 표본으로 세지 않는다.
- Challenge/stress는 robustness analysis일 뿐 두 번째 독립 검증이 아니다.
- Conditional/oracle-current event metric과 end-to-end event metric을 구분한다.
- Weak-label audit 전에는 작은 성능 차이를 label-valid superiority로 해석하지 않는다.

## 3. 고정 통제 원칙

1. Sample ID, episode split, horizon, label mask를 비교 모델 간 공유한다.
2. Normalization은 train split에서만 fit한다.
3. Checkpoint는 natural validation event PR-AUC로 고른다.
4. Threshold는 natural validation에서 한 번 선택해 test/stress/control에 고정한다.
5. Primary metric은 threshold-free natural event PR-AUC다.
6. Thresholded F1, onset/release F1, hard-negative FPR, Brier/ECE는 secondary다.
7. Current auxiliary head는 action-free여야 하며 모델 간 동일해야 한다.
8. Per-sample probability/target/prediction과 task/episode/event metadata를 저장한다.
9. Config, manifest, checkpoint, code snapshot, runtime manifest, result를 versioned root에
   함께 저장한다.
10. Legacy output을 새 protocol 결과로 덮어쓰지 않는다.

## 4. 공식 단계 상태

| 공식 단계 | 상태 | 역할 |
|---|---|---|
| Phase 0 | 완료 | 제공 CLaD baseline 실행 경계와 unknown 보존 |
| Phase 1A | 완료 | LIBERO state/API와 runtime 계약 |
| Phase 2A | 완료 | deterministic static GraphSpec/extractor baseline |
| Phase 2R | diagnostic 완료 | scripted contact/handler/frame regression; main training 제외 |
| Phase 2D | 완료 | official-demo exact replay, temporal graph, holding event dataset |
| Phase 3A | 1차 완료, human QA 대기 | manifest/leakage/quota/hash와 weak-label audit |
| **Phase 3B-R1** | **진행 중** | corrected architecture, action/history control |
| Phase 3C | 미시작 | CLaD-aligned foresight bridge |
| Phase 4 | 차단 | Graph-CLaD Stage 1 representation 통합 |
| Phase 5–8 | 차단 | Stage 2, 최종 평가, 보고 |

## 5. Phase 2D — Official demonstration dataset

### 5.1 목적

Official HDF5 state/action sequence를 exact replay해 canonical graph timeline과 temporal
relation event를 만든다. Scripted Phase 2R의 분포를 주 학습에 사용하지 않는다.

### 5.2 D0 — Source와 provenance

- Task 0/1/2 HDF5 경로, size, SHA256, suite/task mapping을 기록한다.
- Demonstration 150개의 episode ID를 고정한다.
- Split을 sample 생성 전에 고정하고 episode leakage를 금지한다.
- Persistent Drive source를 authoritative input으로 사용한다.

### 5.3 D1 — Exact state replay

- HDF5 state를 environment에 restore하고 simulator forward 후 state를 검사한다.
- `states[t]`, `actions[t]`, next state timing을 명시한다.
- Qpos, EEF pose, object pose replay tolerance를 QA에 저장한다.
- Runtime compatibility patch는 research code/data 수정과 분리한다.

### 5.4 D2 — Canonical graph timeline

- Robot/object/fixture/site는 logical identity로 align한다.
- Node/edge feature와 validity mask를 저장한다.
- `is_object_of_interest`, future success, reward, terminal, future event ID는 primary model
  input에서 제외한다.
- Natural graph와 target-aligned view가 같은 source graph payload를 참조하게 한다.

### 5.5 D3 — Relation과 temporal event

- 기본 geometry relation: left/right/front/behind/above/below/near 등 valid support가 있는 것.
- Interaction relation: contact, holding, on/support 등 capability-aware predicate.
- Unknown은 false로 바꾸지 않고 mask한다.
- Holding은 contact, gripper closure, three-frame stability, object following 기반 weak label다.
- Onset는 not-holding→holding, release는 holding→not-holding이다.
- `inside`는 valid support가 생길 때까지 deferred한다.

### 5.6 D4 — Multi-horizon sample manifest

- Current graph, target graph, action window, task/episode/sample ID, timestep을 기록한다.
- Natural view와 event-enriched stress view를 분리한다.
- Overlap sample은 graph/action/label payload SHA256가 같아야 한다.
- Sample/episode leakage와 task-local category quota를 검사한다.

### 5.7 완료 산출물

- Task 0/1/2 official-demo graph gzip JSONL.
- Fixed episode split manifest.
- Holding-positive/onset/release/hard-negative index.
- Input-clean artifact와 QA manifest.
- Persistent checksum과 runtime provenance.

## 6. Phase 3A — Dataset와 label QA

### 필수 QA

- Task별 natural sample/event prevalence.
- Holding onset/release/hard-negative support.
- Label mask 및 unsupported relation 비율.
- Train/validation/test episode leakage 0.
- Category-aware sampler의 task-local quota.
- Natural/stress overlap payload hash 일치.
- Event-window frame availability와 conflict.

### Human weak-label audit

- Task 0/1/2 × onset/release/hard-negative × 10개 = 90개 최소 screen.
- 각 item에 t~t+6 trajectory와 action evidence를 제공한다.
- 판정은 `pass`, `label_error`, `ambiguous` 중 사람이 입력한다.
- Error/ambiguity가 있는 cell은 최소 30개로 확대한다.
- 자동 내부 일관성 검사는 human ground truth를 대체하지 않는다.

현재 trajectory evidence 592/592를 확보했지만 human review는 0/90이다.

## 7. Phase 3B — Action-conditioned offline relational dynamics

### 7.1 초기 모델 비교

- P0/B0: flat scene MLP.
- B1: robot–object pair feature MLP.
- G00: complete graph + late/global action.
- G10/G1: sparse holder–object graph + late/global action.
- G01: complete graph + action-conditioned update.
- G11: sparse holder–object graph + action-conditioned update.
- S-0: G1 구조에서 action 제거.
- Train-shuffled: 같은 model을 mismatched action donor로 학습.

### 7.2 Target

- Future relation state.
- Conditional/oracle-current change event.
- Predicted-current end-to-end change event.
- Holding onset/release.
- Hard-negative false positive.

### 7.3 필수 control

- No action/constant action/global episode-disjoint action shuffle.
- Train-time episode-disjoint matched action shuffle.
- Sender/receiver 또는 topology edge shuffle.
- Same sample/split/loss/checkpoint/threshold/budget/capacity.
- Donor episode, action distance, state match QA 저장.

### 7.4 초기 GNN gate 결과

Near-parameter-matched 3 folds × 3 seeds reduced gate 36/36 runs를 완료했다.

| Model | Natural PR-AUC | Natural event F1 | Stress event F1 | Natural hard-neg FPR |
|---|---:|---:|---:|---:|
| B1-v2 | **0.4047** | 0.3285 | **0.7670** | **0.1075** |
| G1 | 0.3872 | 0.3400 | 0.7465 | 0.2146 |
| S-0 | 0.3691 | **0.3604** | 0.5927 | 0.2836 |
| G1 train-shuffled | 0.2793 | 0.3470 | 0.4746 | 0.3277 |

G1은 B1을 task 전반에서 일관되게 이기지 못했고 B1이 가장 방어 가능한 baseline이었다.
Natural release가 특히 약했다. Action shuffle 하락은 action signal 사용 가능성을 보였지만
legacy current-head/validation confound 때문에 causal action effect는 아니었다.

### 7.5 Corrected GNN gate

Current head action confound, natural validation checkpoint/threshold, oracle-current naming,
task-local quota/hash QA를 corrected v2에서 고쳤다. Three-fold seed-0에서 G1−B1 PR-AUC
task-macro +0.0626의 CI가 [−0.0892, +0.1905]로 0을 포함했다. Release/FPR은 좋아졌지만
train-shuffled G1이 conditional PR-AUC에서 더 높아 action-alignment gate가 실패했다.

결정: GNN three-seed 확대를 중단하고 pair-local temporal encoder로 전환한다.

### 7.6 Pair-local temporal H0–H3

| Model | Causal history | Action |
|---|---:|---:|
| H0 | 없음 | 없음 |
| H1 | 있음 | 없음 |
| H2 | 없음 | 있음 |
| H3 | 있음 | 있음 |

History feature는 t 이하에서만 계산한다.

- Gripper–object relative-position delta와 relative velocity.
- Contact persistence.
- Gripper closure velocity.
- Object-following stability.
- Feature별 validity mask.

Pair를 독립 encode하고 scene context는 DeepSets/pair-set attention으로 제한한다.
Unrestricted object–object message passing은 이 gate에 넣지 않는다.

Three-fold seed-0 task-macro PR-AUC는 H0 0.3626, H1 0.4348, H2 0.3941,
H3 0.4824였다. H3−H1은 +0.0476이며 3/3 task에서 양수였지만 task 0 hard-negative가
악화됐다. H3만 candidate로 유지하고 matched train-action-shuffled H3 control을 먼저
실행한다.

### 7.7 현재 확대 gate

Aligned H3가 shuffled H3보다 natural PR-AUC에서 최소 2개 task 우세하고 release와
hard-negative가 안전할 때만 H3/H1/H3-shuffled의 seeds 1/2를 실행한다. 실패하면
full seed 확대를 중단하고 pair-local 결과를 architecture finding으로 보고한다.

## 8. Phase 3C — CLaD-aligned foresight bridge

### 목적

Future action 없이 past/current graph transition만으로 CLaD foresight adapter에 필요한
representation 신호가 있는지 본다. Phase 2D의 과거/current view만 사용한다.

### 입력 경계

- 모든 state/history는 현재 t 이하.
- Future action, future graph, future event identity는 encoder input에서 제외.
- Semantic CLaD, pair-local temporal, 선택된 graph encoder를 같은 data로 학습한다.

### Gate

- Frozen encoder에 동일 capacity linear/small probe를 사용한다.
- Holding onset/release, displacement, source→destination, valid spatial transition을 본다.
- Natural held-out task와 label efficiency를 primary로 본다.
- 통과하지 못하면 “offline relation predictor에는 유효하나 CLaD foresight adapter 근거가
  부족함”으로 결론 내리고 Phase 4를 차단한다.

현재 Phase 3C는 미시작이다.

## 9. Phase 4 — Graph-CLaD Stage 1 통합

### 선행 조건

- Phase 3B architecture/action gate 통과.
- Phase 3C no-future-action foresight bridge 통과.
- Weak-label QA와 valid spatial relation support 확보.

### 권장 비교

1. 기존 CLaD semantic-transition representation.
2. Pair-local temporal representation.
3. Object–relation graph-transition representation.

Encoder를 freeze하고 같은 data와 same-capacity probe로 비교한다. Metric은 downstream
event PR-AUC/F1, sample efficiency, hard-negative robustness, perturbation sensitivity,
task 간 paired improvement다.

현재 Phase 4는 차단 상태다.

## 10. Phase 5–8

- Phase 5: baseline CLaD Stage 2 policy 연결.
- Phase 6: 선택된 Graph-CLaD representation을 Stage 2에 연결.
- Phase 7: 동일 rollout/evaluation budget으로 최종 비교.
- Phase 8: statistical report, ablation, claim-limit, reproducibility package 작성.

Phase 4 통합 전에는 위 단계로 진입하지 않는다.

## 11. 현재 바로 다음 작업

1. 실행 중인 H3 action-alignment control 완료 상태와 artifact 무결성을 확인한다.
2. Aligned vs train-shuffled를 same fold/seed로 비교한다.
3. Natural PR-AUC 최소 2 task, release, hard-negative gate를 적용한다.
4. 90-item human weak-label review를 완료한다.
5. 통과 시에만 H3/H1/H3-shuffled seeds 1/2 확대를 검토한다.
6. 그 뒤 Phase 3C를 설계한다. Phase 4/5 notebook을 미리 만들지 않는다.

## 12. 실행 시 참조 우선순위

1. `docs/research_log.md`: 최신 실행과 결정.
2. 이 roadmap v3: 공식 단계와 gate.
3. `docs/phase3_corrected_protocol_v2.md`: corrected evaluation 계약.
4. `docs/01-plan/features/phase3_pair_local_temporal_encoder.plan.md`: 현재 architecture gate.
5. `docs/phase3_weak_label_audit_v2.md`: human label QA.
6. `RESEARCH_GUIDE.md`: source, folder, path, 실행 방법.

## 13. 현재 주장할 수 없는 것

- GNN이 pair MLP보다 일반적으로 우월하다는 주장.
- 현재 late-action G1이 action-conditioned temporal edge model이라는 주장.
- Action shuffle 하락만으로 causal action effect가 입증됐다는 주장.
- Stress view를 독립 challenge generalization으로 부르는 것.
- 3 tasks × 3 seeds를 9 independent samples로 간주하는 것.
- Human review 전 weak label이 ground truth라고 주장하는 것.
- Phase 4 Graph-CLaD representation superiority.
