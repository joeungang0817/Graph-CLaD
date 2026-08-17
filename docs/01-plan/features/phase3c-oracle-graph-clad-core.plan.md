# Phase 3C Oracle Graph-CLaD Core 계획서

> **Summary**: 원 CLaD의 causal past-action 계약을 유지하면서 semantic, set, pair, geometry graph, relation-pooling, relation-message-passing 표현을 공정하게 비교하고, 실제 Graph-CLaD 통합으로 넘길 구조를 선택한다.
>
> **Project**: Graph-CLaD
> **Version**: 1.0
> **Author**: Graph-CLaD 연구팀
> **Date**: 2026-08-17
> **Status**: Implementation Ready

---

## 1. 결정 요약

이제 다음 단계는 **코드 생성이 맞다.** 다만 한 번에 Stage 2 policy까지 구현하지 않는다.
먼저 Phase 3C의 데이터 경계, controlled CLaD baseline, 여섯 core model, metric, artifact
계약을 구현하고 one-task technical smoke를 통과시킨다. 그 뒤에만 SSH의 RTX 3090에서
three-fold seed-0 본 실행을 한다.

이번 구현의 core model은 다음 여섯 개다.

| ID | 추가 입력/구조 | 연구상 역할 |
|---|---|---|
| `C3-Sem-PastAct` | DecisionNCE visual-language + proprioception + causal past action | controlled semantic CLaD 기준선 |
| `C3-SceneSet-PastAct` | full-scene oracle node set, edge 없음 | graph와 같은 scene state를 받는 non-graph control |
| `C3-Pair-PastAct` | robot–entity pair별 temporal encoder + set pooling | pair-local 강한 baseline |
| `C3-GeomMPNN-PastAct` | geometry/contact temporal edge + message passing | relation token 없는 graph control |
| `C3-RelPool-PastAct` | RelMPNN과 동일한 relation edge token + 독립 edge encoding/pooling | relation 정보와 message passing을 분리하는 exact-token control |
| `C3-RelMPNN-PastAct` | geometry/contact/relation temporal edge + message passing | 사전 지정 primary graph candidate |

`RelPool`은 선택 실험이 아니라 필수다. `SceneSet`만으로는 relation token까지 받은
`RelMPNN`과 입력 정보량이 정확히 같지 않기 때문이다. Transformer 계열은 core 결과가
유망한 뒤에만 secondary backbone comparison으로 실행한다.

---

## 2. 연구 목적과 주장 경계

### 2.1 이번 단계가 답하는 질문

`tau=6`에서 이미 실행된 action `a[t-6:t]`과 `<=t` 상태만 사용할 때, explicit
object–relation 구조가 controlled semantic CLaD보다 미래 scene relation change를 더 잘
예측하는가?

### 2.2 Primary contrast

- 모델: `C3-RelMPNN-PastAct − C3-Sem-PastAct`
- 타깃: sample-level valid spatial-relation **any-change**
- metric: relation별 test PR-AUC의 fold macro와 3개 task-fold macro
- seed-0 gate: RelMPNN이 3개 task-fold 중 최소 2개에서 높고 task-macro 차이도 양수
- 확인용 공정성 contrast:
  - `RelMPNN − RelPool`: 같은 relation edge token에서 message passing의 추가 가치
  - `RelMPNN − SceneSet`: full-scene oracle state 대비 graph 구조의 추가 가치
  - `RelMPNN − GeomMPNN`: explicit relation token의 추가 가치
  - `RelMPNN − Pair`: scene graph context의 추가 가치

### 2.3 이번 단계가 증명하지 않는 것

- Phase 3C offline 결과만으로 policy success 개선을 주장하지 않는다.
- Simulator state graph를 쓰므로 결과 명칭은 `Oracle Graph-CLaD`로 제한한다.
- RGB에서 graph를 추출하는 deployable perception을 구현했다고 주장하지 않는다.
- Holding audit가 0/90인 동안 holding 성능을 model selection이나 논문의 성능 근거로 쓰지
  않는다.
- 논문의 공식 Stage 2 코드가 없으므로 향후 policy 구현은 `controlled reimplementation`로
  표기한다.

최종 Graph-CLaD 개선 주장은 같은 Stage 2 policy, demonstration, update budget, rollout
budget에서 controlled semantic CLaD보다 paired success rate가 높을 때만 가능하다.

---

## 3. 고정 데이터 계약

### 3.1 Causal sample join

Phase 2D의 `tau=6` sample 두 개를 다음처럼 join한다.

```text
left:  graph[t-6] -- action[t-6:t] --> graph[t]
right: graph[t]   ------------------> graph[t+6]
```

새 sample은 다음 값만 보유한다.

- 입력: `graph_prev=graph[t-6]`, `graph_t=graph[t]`, `past_action_window=a[t-6:t]`
- semantic 입력 key: `(task_id, demo_key, t-6)`, `(task_id, demo_key, t)`
- target key: `(task_id, demo_key, t+6)`
- target: `graph_target=graph[t+6]`
- metadata: fold, split, task, episode, demo, 세 timestep, source hash

다음 조건을 모두 만족해야 join한다.

1. episode, task, demo, split, `tau=6`이 같다.
2. `left.target_step == right.start_step`이다.
3. `left.graph_target`과 `right.graph_t`의 canonical JSON SHA-256이 같다.
4. past action shape는 정확히 `[6, 7]`이고 모두 finite다.
5. output schema에는 right sample의 `action[t:t+6]` field 자체가 없다.

Join QA에는 source sample 수, join 수, episode boundary에서 제외된 수, hash mismatch,
duplicate sample ID, split leakage, action shape 오류를 기록한다. Hash mismatch, duplicate,
split leakage, future-action payload가 하나라도 있으면 build를 실패시킨다.

### 3.2 Action timing QA

기존 Phase 2D label은 HDF5 state를 직접 복원했으며 action을 step해 label을 만들지 않았다.
따라서 action index가 state transition과 맞는지는 별도로 검사한다.

- 각 task의 train/validation/test demo에서 사전 고정한 frame을 뽑는다.
- `state[t]`를 복원하고 `action[t]`을 한 번 step한 simulator state와 저장된
  `state[t+1]`을 비교한다.
- max absolute state error, observation key, controller 설정을 저장한다.
- tolerance는 train/validation smoke의 simulator precision만 보고 **본 학습 전에**
  config에 고정한다. Test frame은 그 frozen tolerance로 pass/fail만 기록하고 tolerance를
  바꾸지 않는다.

### 3.3 Semantic feature store

현재 Phase 2D artifact에는 RGB/semantic embedding이 없으므로 controlled CLaD를 위해 같은
official HDF5 state를 render하는 별도 read-only feature 추출 단계가 필요하다.

- 모델: `DecisionNCE-P`를 primary controlled assumption으로 사용한다.
- 시점: joined manifest에 등장하는 unique `(task, demo, step)`만 render한다.
- view: runtime preflight에서 확인한 두 camera key를 config에 명시적으로 고정한다.
- render size: 224×224, 공식 DecisionNCE preprocessing 적용.
- 저장: view별 image embedding과 task별 language embedding, float32.
- 원본 RGB는 기본적으로 저장하지 않고 orientation QA용 소수 contact sheet만 저장한다.
- manifest에는 DecisionNCE repository commit, checkpoint SHA-256, preprocessing repr,
  camera key/order, image orientation, feature dimension, source HDF5 SHA-256을 남긴다.

CLaD 논문은 DecisionNCE 사용은 밝히지만 P/T variant를 특정하지 않는다. 따라서 P 선택은
논문 공식값이 아니라 controlled reimplementation assumption으로 명시하며,
`DecisionNCE-T`는 필요할 때만 sensitivity experiment로 둔다.

### 3.4 Proprioception과 action

- proprioception `p_t`: joint position 7 + joint velocity 7 + gripper qpos 2 = 16차원.
- missing/invalid proprioception은 0으로 조용히 대체하지 않고 sample build를 실패시킨다.
- train task에서만 mean/std를 적합하고 validation/test에 그대로 적용한다.
- action은 HDF5의 7차원 action을 시간순으로 flatten한 42차원 입력으로 사용한다.
- future action, reward, success, BDDL target flag, `is_object_of_interest`, task ID embedding은
  모든 model input에서 금지한다.

### 3.5 Node와 edge 입력

Phase 2D가 `include_all_sites=False`로 만들어졌으므로 기존 artifact의 node scope를 그대로
사용한다. 즉 robot, object, fixture를 사용하고 제외된 site를 몰래 재도입하지 않는다.

`phase3c_node_v1`은 다음만 포함한다.

- node type one-hot: robot/object/fixture/site 4차원
- robot-base position 3차원
- position valid 1차원

Stable logical ID는 prev/current node 정렬과 prediction provenance에만 사용하고 embedding
feature로 넣지 않는다. Joint/gripper 상태는 공통 CLaD proprio branch에서만 제공하여 graph
variant에 중복 privileged path를 만들지 않는다.

Graph topology는 현재 artifact에 있는 complete directed spatial edges를 사용한다.

- geometry token: prev/current relative xyz, distance, validity, temporal delta
- contact token: prev/current value와 validity
- relation token: prev/current relation value와 validity
- action token: 동일한 `a[t-6:t]` encoder output

`GeomMPNN`은 geometry/contact까지만, `RelPool`과 `RelMPNN`은 여기에 relation token을
추가한다. `RelPool`과 `RelMPNN`의 raw edge token은 byte-level schema 기준으로 동일해야
한다.

---

## 4. 타깃과 label support

### 4.1 Primary relation 집합

현재 handler에 실제로 구현된 관계 중 다음 여덟 개만 Phase 3C v1 후보로 사용한다.

```text
left, right, front, behind, above, below, contact, on
```

기존 계획에 적혔던 `near`와 `support`는 현재 relation handler에 없으므로 사용하지 않는다.
`inside`는 support 부족, `holding`은 human audit 미완료, `open/close`는 unary state이므로
primary target에서 제외한다.

### 4.2 Candidate target edge

- source: `node_type == object`
- target: `node_type in {object, fixture}`
- source와 target은 달라야 한다.
- current와 future relation record 모두 `valid=1`일 때만 해당 edge/relation이 valid다.
- robot edge는 primary spatial metric에서 제외하고 별도 diagnostic에만 둘 수 있다.

### 4.3 Sample-level any-change

Semantic CLaD처럼 global representation을 내는 모델과 graph model을 같은 head로 비교하기
위해 primary target을 고정 길이 scene vector로 만든다.

```text
y[sample, relation] = OR over eligible directed edges(
    valid_current AND valid_future AND value_current != value_future
)
```

해당 sample에서 valid candidate edge가 하나도 없으면 그 relation의 loss/metric mask는 0이다.
모델은 future-state 두 개의 thresholded XOR이 아니라 **direct change logit**을 출력한다.

### 4.4 Fold별 eligibility

Test를 보지 않고 train+validation support로 relation eligibility를 고정한다.

- train positive-change sample ≥ 20
- train no-change sample ≥ 20
- validation positive-change sample ≥ 5
- validation no-change sample ≥ 5

Test가 single-class이면 그 fold/relation PR-AUC는 `null`로 저장하고 억지로 0이나 0.5로
채우지 않는다. Evaluable relation이 fold당 2개 미만이면 model 학습 전에 protocol을
`unsupported`로 중단하고 relation 정의나 task 범위를 새 version으로 재설계한다.

### 4.5 Trivial baseline

- Primary direct-change target: `NO-CHANGE`, 모든 change probability를 train prevalence 또는
  0으로 예측하는 두 값을 모두 보고한다.
- Future-state diagnostic을 구현하는 경우: `COPY-CURRENT`를 보고한다.

Direct-change metric에서 `COPY-CURRENT`는 사실상 no-change와 같으므로 서로 다른 두
baseline처럼 과장하지 않는다.

---

## 5. Controlled CLaD와 core model 학습 계약

### 5.1 Fold별 base CLaD

각 outer fold에서 test task를 완전히 배제하고 base CLaD Stage 1을 한 번 학습한다.

- input tensor:
  - `v_history [B, 2, 2, D_v]`: prev/current × 2 views
  - `p_history [B, 2, 16]`
  - `past_action [B, 42]`
  - `language [B, D_v]`
  - target `v_next [B, 2, D_v]`, `p_next [B, 16]`
- paper-aligned fixed values: `H=1024`, `N_p=N_s=4`, `tau=6`, batch 128,
  25,000 updates, EMA 0.995, reconstruction weight 0.1.
- 제공된 `LatentDynamics`의 training action/token mask ratio는 0.3, evaluation은 0으로
  고정하고 resolved config에 기록한다.
- `D_v == H`를 runtime assert한다. 맞지 않으면 projection을 임의로 추가하지 않고 protocol을
  중단한다.
- total loss:
  `loss_p + loss_s + 0.1 * (loss_p_recon + loss_v_recon)`.
- optimizer는 논문에 명시되지 않았으므로 controlled assumption으로 AdamW, lr `1e-4`,
  weight decay `1e-4`, gradient clip `1.0`, scheduler 없음으로 사전 고정한다.
- optimizer step 직후 정확히 한 번 `update_ema()`를 호출한다.
- checkpoint는 validation total Stage 1 loss 최저값으로 선택하며 test relation metric을
  보지 않는다.

동일 `(fold, seed)`의 base CLaD checkpoint는 여섯 core model이 공유한다.

### 5.2 Phase 3C screen 방식

Base CLaD를 freeze하고 `[pred_p_emb; pred_s_emb]`에서 공통 base latent를 만든다. 각 candidate는
같은 크기의 adapter latent를 만들고 공통 256차원 residual fusion과 같은 prediction head를
사용한다.

```text
h_base = Project(LayerNorm(concat(pred_p, pred_s)))
h = LayerNorm(h_base + gate * h_adapter)
change_logits = Linear(h, R)
motion = Linear(h, 1)
```

- `C3-Sem`의 adapter는 `h_base + past-action embedding`만 받는 MLP다.
- 나머지 adapter는 각각 SceneSet, Pair, GeomMPNN, RelPool, RelMPNN 구조를 쓴다.
- 공통 base projector, candidate adapter, prediction heads는 학습하고 CLaD backbone은 freeze한다.
- adapter+head trainable parameter는 사전 지정 target의 ±5% 안으로 맞추고 실제 수를 모두
  공개한다.
- relation loss는 train support로 계산한 capped positive weight(`max=20`)를 쓰는 masked BCE다.
- secondary scene motion target은 object의 `t→t+6` 최대 displacement(m)이며 Smooth-L1을
  사용한다.
- total screen loss는 `relation_bce + 0.1 * motion_smooth_l1`이다.
- checkpoint criterion은 validation relation macro PR-AUC 하나다.

이 screen은 relation supervision을 사용한 **matched architecture screen**이다. 이를 순수한
self-supervised frozen-representation 결과라고 부르지 않는다. Phase 4에서는 선택 구조를
CLaD Stage 1 foresight residual로 다시 통합하고 원 latent prediction objective로 학습한다.

### 5.3 Core screen budget

- seed: 먼저 0만 사용
- outer folds: `test_task0`, `test_task1`, `test_task2`
- adapter updates: 10,000, batch 64
- optimizer: AdamW, lr `3e-4`, weight decay `1e-4`, gradient clip `1.0`
- validation: 500 updates마다
- early stopping: validation 5회 연속 무개선, 단 최소 3,000 updates 수행
- mixed precision: CUDA에서 bf16 가능 여부를 검사하고, 불가능하면 fp16 + GradScaler;
  metric과 prediction 저장은 float32
- seed별 deterministic flags와 실제 CUDA deterministic 상태를 runtime manifest에 기록

---

## 6. 평가와 gate

### 6.1 Primary metric

1. relation별 natural test PR-AUC
2. fold macro PR-AUC
3. 3개 held-out task-fold macro
4. same fold/seed `RelMPNN − comparator` paired difference
5. inverse relation 중복을 줄인 relation-family macro를 함께 보고한다:
   `left/right`, `front/behind`, `above/below`, `contact`, `on`. Family 내부를
   먼저 평균한 뒤 family 간 macro를 계산하며, 기존 relation macro를 대체하지 않고 병기한다.

### 6.2 Secondary metric

- validation에서 고정한 relation별 threshold의 F1
- Brier score, 10-bin ECE
- no-change false-positive rate
- scene max-displacement MAE/RMSE
- moving-scene(`max displacement > 0.01 m`) subset MAE
- graph model에 한해 edge-level relation localization과 per-object displacement diagnostic
- node/edge/action permutation sensitivity

### 6.3 확대 gate

Seed 0의 18개 core run이 모두 완료된 뒤 다음을 확인한다.

1. Primary `RelMPNN−Sem`이 최소 2/3 fold에서 양수이고 task-macro도 양수다.
2. `RelMPNN−RelPool`이 양수여야 message passing의 추가 가치를 주장할 수 있다.
3. `RelMPNN−SceneSet`이 양수여야 graph structure의 추가 가치를 주장할 수 있다.
4. hard-negative FPR과 calibration이 comparator보다 심하게 악화되지 않는다.
5. 결과가 한 relation 하나에만 의존하지 않는다.

Primary gate를 통과하면 seeds 1/2는 우선 `Sem`, `RelPool`, `RelMPNN` 세 모델만 확장한다.
Transformer, no-action, shuffled-past-action은 core 결과 해석에 필요할 때만 추가한다.

---

## 7. 구현 범위

### 7.1 새 코드

- `scripts/phase3c/contracts.py`: config/schema/blacklist/hash 검증
- `scripts/phase3c/build_joined_manifest.py`: causal sample join과 support report
- `scripts/phase3c/validate_action_timing.py`: HDF5 action/state indexing QA
- `scripts/phase3c/build_semantic_feature_store.py`: render + DecisionNCE embedding cache
- `scripts/phase3c/dataset.py`: tensorization, train-only normalization, collate/masks
- `scripts/phase3c/models/semantic_clad.py`: 제공된 LatentDynamics의 안전한 wrapper
- `scripts/phase3c/models/structured.py`: SceneSet/Pair/GeomMPNN/RelPool/RelMPNN
- `scripts/phase3c/models/adapters.py`: 공통 projection, residual fusion, heads
- `scripts/phase3c/losses.py`: masked BCE와 motion loss
- `scripts/phase3c/metrics.py`: PR-AUC/F1/calibration/motion metric
- `scripts/phase3c/train_base_clad.py`: fold별 Stage 1 trainer
- `scripts/phase3c/train_core.py`: 한 model/fold/seed trainer
- `scripts/phase3c/run_core.py`: resume 가능한 multi-run orchestrator
- `scripts/phase3c/analyze_core.py`: paired comparison과 hierarchical bootstrap

기존의 큰 `scripts/phase3/offline_probe.py`에 model ID를 계속 추가하지 않고 Phase 3C를 별도
package로 분리한다. `baseline_code/`는 원형 보존을 위해 직접 수정하지 않는다.

### 7.2 새 config

- `configs/phase3c_contract_v1.json`
- `configs/phase3c_kcloudvpn_data_smoke_v1.json`
- `configs/phase3c_kcloudvpn_core_smoke_seed0_v1.json`
- `configs/phase3c_kcloudvpn_core_threefold_seed0_v1.json`
- `requirements-phase3c.txt`

### 7.3 새 test

- join/hash/split/future-action 차단
- relation target/validity/eligibility
- semantic feature key/shape/determinism
- CLaD loss weight와 EMA 호출 순서
- model shape, finite gradient, parameter budget
- SceneSet/Pair/RelPool permutation invariance
- MPNN node permutation equivariance와 pooled-output invariance
- adapter-off equivalence
- metric의 single-class `null` 처리
- atomic artifact/resume/run-key 중복 방지

---

## 8. 실행 순서와 중단 조건

### Gate 0 — 로컬 CPU test

- synthetic fixture로 모든 unit test 실행
- 32-sample tiny overfit에서 loss가 감소하는지 확인
- 실패 시 SSH로 코드를 보내지 않는다.

### Gate 1 — SSH data contract smoke

- task 0의 demo 소수로 join dry-run
- action timing QA
- relation support report
- 미래 action/future graph input leakage 0 확인

### Gate 2 — semantic feature smoke

- 한 episode의 prev/current/future frame 몇 개만 render
- camera key/order와 image orientation contact sheet 확인
- embedding shape/finite/checkpoint hash 확인
- 같은 frame 재추출 cosine similarity ≥ 0.9999

### Gate 3 — base CLaD smoke

- synthetic `H=64` shape/EMA test
- 실제 feature 32–128 sample tiny overfit
- 그 뒤 task-fold 하나의 `H=1024` 100-update GPU smoke

### Gate 4 — six-model technical smoke

- 한 fold, seed 0, capped data에서 각 model 100 updates
- 목적은 shape, OOM, leakage, metric, artifact 검증이며 성능 선택에 쓰지 않는다.

### Gate 5 — seed-0 본 실행

1. fold별 base CLaD 25K updates 3개
2. 여섯 core model × 세 folds = 18 runs
3. paired analysis와 artifact completeness 검사

### Gate 6 — 조건부 확대

- gate 통과 시 Sem/RelPool/RelMPNN seeds 1/2
- 필요 시 Transformer 2×2와 action controls
- semantic baseline, 최종 winner, winner의 가장 가까운 fairness control을 Phase 4 CLaD
  foresight adapter로 이동. RelMPNN이 winner면 RelPool도 함께 이동

단일 RTX 3090에서는 GPU training을 병렬 실행하지 않는다. 정확한 시간·VRAM은 100-update
smoke에서 측정한 뒤 full-run manifest에 예상치와 실제치를 기록한다.

### 초기 실행 비용 예산

다음은 scheduling용 범위이지 보장 시간이 아니다.

| 단계 | RTX 3090 초기 예산 | 확정 방법 |
|---|---:|---|
| join/support/action QA | 10–30분 | 실제 record/simulator frame 수 |
| unique-frame semantic extraction | 1–6시간 | 100-frame throughput에서 외삽 |
| base CLaD 25K 한 fold | 3–6시간 | 100-update smoke에서 외삽 |
| base CLaD seed-0 세 folds | 9–18시간 | fold별 순차 실행 |
| core adapter 한 run | 15–45분 | 100-update smoke에서 외삽 |
| core 18 runs | 4.5–13.5시간 | 단일 GPU 순차 실행 |
| paired analysis | 30분 이내 | prediction row 수 |

Paper의 Stage 1 25K 약 2시간은 RTX 4090 기준이므로 3090 시간으로 그대로 약속하지 않는다.
Smoke가 위 범위를 크게 벗어나면 batch/gradient accumulation을 새 config version에 고정하고
전체 실행 전에 계획 시간을 갱신한다.

---

## 9. Artifact와 재현성

Git에는 code/config/작은 report만 저장하고 embedding/checkpoint/prediction은 다음 root 아래에
둔다.

```text
${GRAPH_CLAD_ARTIFACT_ROOT}/phase3c_oracle_graph_clad_v1/
  data_contract/
  semantic_store/
  base_clad/<protocol>/<fold>/seed<seed>/
  core_screen/<protocol>/<model>/<fold>/seed<seed>/
  analysis/<protocol>/
```

각 run은 다음을 필수 저장한다.

- resolved config와 SHA-256
- git commit, dirty diff hash, code snapshot
- source manifest/HDF5/DecisionNCE checkpoint SHA-256
- Python/PyTorch/CUDA/GPU/package versions
- seed와 deterministic flags
- normalization/eligibility/positive-weight statistics
- best/last checkpoint
- gzip per-sample prediction
- metric JSON, stdout/stderr, runtime manifest

Run key는 `(protocol, model_id, fold, seed)`이며 completed artifact가 있으면 SHA를 검증한 뒤
skip한다. Partial run은 last checkpoint에서만 resume하고 기존 completed result를 덮어쓰지
않는다. JSON은 temporary file에 쓴 뒤 atomic rename한다.

---

## 10. 완료 조건

- [ ] 모든 입력 schema와 금지 field가 versioned config에 고정됨
- [ ] Causal join/action timing/semantic render QA가 pass
- [ ] Controlled base CLaD가 실제 DecisionNCE feature로 train/eval 가능
- [ ] 여섯 core model의 parameter/probe/training 계약이 일치
- [ ] unit/integration/smoke test가 pass
- [ ] one-fold technical smoke artifact가 완전함
- [ ] three-fold seed-0 18/18 run이 완료됨
- [ ] primary/fairness contrasts와 claim limit이 연구기록에 남음
- [ ] full run 전후 config와 code SHA가 고정됨

---

## 11. 주요 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| Phase 2D에 RGB가 없음 | semantic CLaD 실행 불가 | 같은 HDF5 state에서 unique frame만 render해 embedding cache 생성 |
| CLaD 논문의 미공개 optimizer/VLM variant | 공식 재현 주장 불가 | controlled assumption으로 사전 고정하고 source/checkpoint SHA 공개 |
| SceneSet이 relation-token 정보량과 불일치 | graph 구조 효과 과대평가 | RelPool을 필수 exact-token control로 추가 |
| 존재하지 않는 near/support target | 잘못된 label/코드 | 현재 handler의 8개 relation만 사용 |
| future action 누수 | causal claim 무효 | output schema에서 field 제거 + poison test + recursive blacklist |
| relation rare positive | PR-AUC 불안정 | train/validation eligibility gate, single-class null, task별 결과 공개 |
| graph input이 simulator oracle | deployability 과장 | `Oracle Graph-CLaD`로 명명하고 RGB graph extraction 주장 금지 |
| seed-0 과해석 | 불안정한 결론 | gate 통과 후보만 seeds 1/2 확대 |
| 3090 runtime/VRAM 초과 | 전체 실험 중단 | 100-update smoke에서 batch/AMP 결정, gradient accumulation은 config에 기록 |

---

## 12. 다음 단계

1. 이 계획과 기술 설계를 기준으로 Phase 3C package와 test부터 구현한다.
2. 로컬 test가 모두 통과하면 commit/push한다.
3. SSH에서 pull 후 Gate 1부터 순서대로 실행한다.
4. 각 gate 결과와 모든 수정은 즉시 `docs/research_log.md`에 기록한다.

---

## Version History

| Version | Date | Changes | Author |
|---|---|---|---|
| 1.0 | 2026-08-17 | Phase 3C six-model core, causal join, semantic replay, support rule, training/evaluation/artifact gate 확정 | Graph-CLaD 연구팀 |
