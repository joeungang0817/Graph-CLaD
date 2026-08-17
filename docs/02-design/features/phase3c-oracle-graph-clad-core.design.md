# Phase 3C Oracle Graph-CLaD Core 기술 설계서

> **Summary**: causal joined sample, DecisionNCE feature cache, controlled CLaD backbone, 여섯 adapter, 공통 relation-change head와 재현 가능한 runner의 파일·자료형·검증 계약을 정의한다.
>
> **Project**: Graph-CLaD
> **Version**: 1.0
> **Author**: Graph-CLaD 연구팀
> **Date**: 2026-08-17
> **Status**: Implementation Ready
> **Planning Doc**: [phase3c-oracle-graph-clad-core.plan.md](../../01-plan/features/phase3c-oracle-graph-clad-core.plan.md)

---

## 1. 설계 원칙

1. **Causal boundary first**: model code보다 먼저 sample schema에서 future action과 future graph
   input을 제거한다.
2. **One source of truth**: relation list, feature list, blacklist, run key는 contract config 한 곳에서
   읽는다.
3. **Unknown is not false**: `valid=0` relation은 negative label로 바꾸지 않는다.
4. **Common backbone and head**: fold/seed별 frozen CLaD backbone과 output head 계약을 모든
   candidate가 공유한다.
5. **Information controls**: SceneSet은 scene-state control, RelPool은 exact relation-token
   no-message control이다.
6. **No silent fallback**: camera, proprioception, embedding dimension, relation support가 맞지
   않으면 임의 보정 없이 실패한다.
7. **Resume without overwrite**: completed run은 hash 검증 후 skip하고 partial run만 resume한다.
8. **Claims follow evidence**: Phase 3C는 architecture screen이고 최종 policy 개선 증명이 아니다.

---

## 2. 전체 구성

```text
Phase 2D graph shards + official HDF5 states
          |
          +--> JoinedManifestBuilder --> joined_manifest_full_demo_fixed.jsonl.gz
          |          |                    support_report.json
          |          +--> ActionTimingValidator
          |
          +--> SemanticFeatureBuilder --> feature shards + manifest.json
                                             |
joined manifest + semantic store -------------+
                    |
                    +--> Phase3CDataset / Collator
                               |
                               +--> Base CLaD trainer (per fold/seed)
                               |          |
                               |          +--> frozen base checkpoint
                               |
                               +--> Core candidate adapter + common heads
                                          |
                                          +--> predictions/checkpoints/metrics
                                                        |
                                                        +--> paired analyzer
```

Phase 2D 생성 코드는 수정하지 않는다. Joined manifest와 semantic store는 immutable derived
artifact이며, source hash가 바뀌면 새 version을 만든다.

---

## 3. Package와 책임

```text
scripts/phase3c/
  __init__.py
  contracts.py
  io.py
  build_joined_manifest.py
  validate_action_timing.py
  build_semantic_feature_store.py
  dataset.py
  parameter_match.py
  losses.py
  metrics.py
  train_base_clad.py
  train_core.py
  run_core.py
  analyze_core.py
  models/
    __init__.py
    common.py
    semantic_clad.py
    structured.py
    adapters.py
```

### `contracts.py`

- versioned constants를 config에서 읽고 schema를 검증한다.
- `${GRAPH_CLAD_ARTIFACT_ROOT}`만 명시적으로 확장한다.
- unknown config key를 경고로 넘기지 않고 오류로 처리한다.
- recursive forbidden-field scan을 제공한다.

```python
PRIMARY_RELATIONS = (
    "left", "right", "front", "behind",
    "above", "below", "contact", "on",
)

FORBIDDEN_INPUT_KEYS = {
    "graph_target", "future_graph", "future_action", "action_future",
    "reward", "success", "is_object_of_interest", "task_semantics",
    "bddl", "goal", "goal_state", "relation_changes",
}
```

`task_id`, `episode_id`, `demo_key`는 split/join/provenance용 metadata로 허용하지만 tensor
feature로 변환하는 함수에는 전달하지 않는다.

### `io.py`

- gzip JSONL streaming read/write
- canonical JSON SHA-256
- atomic JSON/checkpoint write
- completed/partial run 검사
- source/code/runtime manifest 작성

모든 큰 artifact를 list로 한꺼번에 메모리에 올리지 않고 record 단위로 처리한다.

---

## 4. Joined manifest schema

한 줄은 하나의 `tau=6` sample이다.

```json
{
  "schema": "phase3c-joined-sample.v1",
  "sample_id": "task0_demo_0_prev40_cur46_next52_tau6",
  "task_id": 0,
  "episode_id": "task0_demo_0",
  "demo_key": "demo_0",
  "split": "train",
  "tau": 6,
  "prev_step": 40,
  "current_step": 46,
  "target_step": 52,
  "past_action_window": [[0, 0, 0, 0, 0, 0, 0]],
  "graph_prev": {},
  "graph_t": {},
  "target": {
    "graph": {},
    "relation_any_change": {},
    "relation_valid": {},
    "scene_max_displacement_m": 0.0
  },
  "hashes": {
    "left_graph_t": "...",
    "right_graph_t": "...",
    "source_left": "...",
    "source_right": "..."
  }
}
```

실제 `past_action_window`은 정확히 6행×7열이다. 위 JSON은 shape 예시일 뿐이다.

Builder는 `(episode_id, start_step, tau)`를 right-sample lookup key로 사용한다. 각 left
sample의 `target_step`과 같은 start step의 right sample을 찾는다. 결과 생성 순서는
`task_id, demo_key, current_step` 정렬로 고정한다.

### Join pseudocode

```python
for episode_record in stream_phase2d_records():
    tau6 = index_samples(episode_record, tau=6)
    for left in tau6:
        right = tau6.get(start_step=left.target_step)
        if right is None:
            count_boundary_drop()
            continue
        assert_same_episode_split_tau(left, right)
        assert sha256(left.graph_target) == sha256(right.graph_t)
        emit(make_causal_record(
            graph_prev=left.graph_t,
            graph_t=left.graph_target,
            past_action=left.action_window,
            graph_target=right.graph_target,
        ))
```

Output record를 만든 직후 `model_input_view(record)`에만 recursive blacklist 검사를
실행한다. Target subtree는 정답을 보존하되 tensorizer가 model forward input으로 넘기지
않는다. Test에서는 right action을 극단값으로 poison해도 model-input tensor hash가
동일한지 확인한다.

---

## 5. Relation target builder

### 5.1 Edge 선택

`graph_t`와 `graph_target`의 edge를 `(source_id, target_id)`로 join한다. 두 graph에 공통으로
존재하고 source node가 object, target node가 object/fixture인 directed edge만 후보로 둔다.

### 5.2 Relation mask와 label

```python
valid_e_r = int(current.valid == 1 and future.valid == 1)
changed_e_r = int(valid_e_r and current.value != future.value)

sample_valid_r = int(any(valid_e_r for e in candidate_edges))
sample_changed_r = int(any(changed_e_r for e in candidate_edges))
```

Edge별 current/future value, valid, change는 prediction artifact의 error analysis용으로 함께
보존한다. 그러나 global semantic model과 공정한 primary head는 sample-level 8-vector만
학습한다.

### 5.3 Eligibility artifact

Fold마다 다음 schema를 저장한다.

```json
{
  "fold": "test_task0",
  "selection_source": ["train", "validation"],
  "thresholds": {
    "train_positive": 20,
    "train_negative": 20,
    "validation_positive": 5,
    "validation_negative": 5
  },
  "relations": {
    "left": {
      "train_positive": 0,
      "train_negative": 0,
      "validation_positive": 0,
      "validation_negative": 0,
      "eligible": false
    }
  },
  "test_counts_inspected_for_selection": false
}
```

이 파일의 SHA를 모든 run config에 기록한다. Eligibility가 만들어진 뒤 model runner가
test count를 보고 channel을 추가하거나 제거할 수 없게 한다.

---

## 6. Semantic feature store

### 6.1 Key와 shard

Feature key는 `task_id/demo_key/step/view_key`다. Demo별 `.npz` 또는 HDF5 shard를 사용하고
manifest index가 `(task, demo, step)`에서 shard/row를 찾는다.

```text
semantic_store/
  manifest.json
  task0/demo_0.npz
  task0/demo_1.npz
  ...
  qa/camera_inventory.json
  qa/orientation_contact_sheet.png
  qa/determinism.json
```

한 shard에는 다음 array를 둔다.

- `steps [F] int32`
- `view0 [F, D_v] float32`
- `view1 [F, D_v] float32`
- `state_restore_max_abs [F] float64`

Language embedding은 task별로 manifest에 별도 array/file로 저장한다. Feature는 encoder가
반환한 값을 그대로 저장하고 normalization 여부를 manifest에 명시한다. 임의 L2
normalization을 추가하지 않는다.

### 6.2 Camera discovery gate

1. render-enabled environment에서 observation key, dtype, shape를 수집한다.
2. RGB 후보 key를 report에만 출력한다.
3. config의 exact two-key allowlist와 모두 일치해야 추출한다.
4. camera order는 항상 `[external, wrist]`처럼 config에서 의미와 함께 고정한다.
5. 8개 representative frame contact sheet로 상하 반전과 channel order를 확인한다.

Builder가 임의로 첫 두 image key를 선택하지 않는다.

### 6.3 DecisionNCE wrapper

공식 loader API를 thin wrapper로 감싼다.

```python
class DecisionNCEEncoder:
    def encode_images(self, images: Tensor) -> Tensor: ...
    def encode_texts(self, texts: list[str]) -> Tensor: ...
    @property
    def feature_dim(self) -> int: ...
```

Wrapper는 `eval()`, `requires_grad_(False)`, `torch.inference_mode()`를 강제한다. Model id,
repository commit과 local checkpoint SHA가 없으면 feature store를 completed로 표시하지 않는다.

---

## 7. Dataset tensor contract

`Phase3CBatch`는 metadata와 tensor를 분리한다.

```python
@dataclass(frozen=True)
class Phase3CBatch:
    sample_ids: tuple[str, ...]
    task_ids: Tensor                 # metadata only; model forward에 전달 금지
    episode_ids: tuple[str, ...]
    v_history: Tensor                # [B, 2, 2, Dv]
    p_history: Tensor                # [B, 2, 16]
    past_action: Tensor              # [B, 6, 7]
    language: Tensor                 # [B, Dv]
    graph_prev: GraphBatch
    graph_current: GraphBatch
    target_v: Tensor                 # [B, 2, Dv], loss/eval only
    target_p: Tensor                 # [B, 16], loss/eval only
    target_relation_change: Tensor   # [B, 8]
    target_relation_mask: Tensor     # [B, 8]
    target_scene_motion: Tensor      # [B, 1]
```

`GraphBatch`:

```python
@dataclass(frozen=True)
class GraphBatch:
    node_features: Tensor            # [B, Nmax, 8]
    node_mask: Tensor                # [B, Nmax]
    edge_geometry: Tensor            # [B, Nmax, Nmax, G]
    edge_contact: Tensor             # [B, Nmax, Nmax, 2]
    edge_relations: Tensor           # [B, Nmax, Nmax, 7*2]
    edge_mask: Tensor                # [B, Nmax, Nmax]
```

Relation edge tensor의 7은 contact를 제외한 `left/right/front/behind/above/below/on`이다.
각 `GraphBatch`가 한 시점만 나타내므로 relation은 시점당
`7 relations × (value, valid) = 14`차원이고 contact는 `(value, valid) = 2`차원이다.
Prev/current/delta 시간 결합은 structured model에서 정확히 한 번만 계산한다.

Node order는 sample 안에서 `(node_type_rank, logical_id)`로 deterministic하게 정렬하되, model
test에서는 random permutation 후 pooled output 불변성을 확인한다.

### Normalization

- proprio mean/std: fold train samples만 사용
- position/relative position/distance mean/std: fold train graph만 사용
- std가 `<1e-8`이면 1로 고정하고 constant feature 목록을 기록
- binary value/valid/type mask는 normalize하지 않음
- validation/test로 fit하거나 test distribution에 맞춰 clip하지 않음

---

## 8. Controlled CLaD wrapper

`baseline_code.LatentDynamics`를 직접 고치지 않고 adapter wrapper로 다음을 보장한다.

```python
class ControlledCLaD(nn.Module):
    def training_loss(self, batch: Phase3CBatch) -> dict[str, Tensor]: ...
    @torch.no_grad()
    def encode_foresight(self, batch: Phase3CBatch) -> Tensor: ...  # [B, 2048]
    @torch.no_grad()
    def update_ema_after_optimizer_step(self) -> None: ...
```

### Required assertions

- exactly two time points and two views
- `Dv == hidden_dim == 1024`
- `p_dim == 16`
- flattened action dim `6*7 == 42`
- target tensors are present only in `training_loss`, never `encode_foresight` input path
- every tensor finite
- `training_loss`는 `action_mask_ratio=0.3`, validation/eval은 0으로 호출

### Training step order

```python
optimizer.zero_grad(set_to_none=True)
losses = model.training_loss(batch)
total = losses["loss_p"] + losses["loss_s"] \
      + 0.1 * (losses["loss_p_recon"] + losses["loss_v_recon"])
scaler.scale(total).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(online_parameters, 1.0)
scaler.step(optimizer)
scaler.update()
model.update_ema_after_optimizer_step()
```

EMA target network은 optimizer parameter group에 들어가면 안 된다. Unit test는 optimizer
step 전후 online parameter와 EMA parameter 변화를 모두 검사한다.

---

## 9. Candidate model 설계

공통 dimension:

- frozen CLaD output: 2048
- base/adapter latent: 256
- action-adapter hidden width: parameter-matched per candidate
- relation output channels: 8
- dropout: 0.1
- activation: GELU
- 모든 pooling은 mask-aware

### 9.1 `C3-Sem-PastAct`

```text
CLaD foresight 2048 -> base projector 256
past action 42 -> parameter-matched action MLP -> 256
LayerNorm(base + sigmoid(gate) * action_adapter) -> shared heads
```

Graph tensor를 forward signature에 받지 않는다. Shared trainer가 실수로 graph를 넘겨도
semantic module에서는 사용할 수 없도록 typed interface를 분리한다.

### 9.2 `C3-SceneSet-PastAct`

Prev/current 각 node의 feature를 shared MLP로 encode하고 temporal difference를 만든 뒤
action-FiLM을 적용한다. Node 사이 message passing 없이 gated attention pooling한다.

```text
node(prev), node(cur), delta -> shared temporal node MLP
                              -> action FiLM
                              -> masked attention pool -> 256
```

### 9.3 `C3-Pair-PastAct`

Robot과 각 non-robot entity의 prev/current relative state를 독립적으로 encode한다. Pair간
상호작용 없이 마지막에 masked attention pooling한다. Object-object edge는 보지 않는다.

### 9.4 `C3-GeomMPNN-PastAct`

- complete directed edge
- input: node temporal token + geometry + contact + validity + action
- 2 residual message-passing layers
- aggregation: masked mean
- graph readout: node attention pooling

```text
m_ij = EdgeMLP([h_i, h_j, geom_ij, contact_ij, action])
h_i' = LayerNorm(h_i + NodeMLP([h_i, mean_j(m_ji)]))
```

### 9.5 `C3-RelPool-PastAct`

RelMPNN과 완전히 같은 edge input token을 사용하지만 edge끼리 정보를 교환하지 않는다.

```text
e_ij = EdgeMLP([node_i, node_j, geom, contact, relation, action])
h = masked_attention_pool({e_ij})
```

Edge set permutation에 불변이어야 한다. 이 모델과 RelMPNN 사이 차이가 message passing의
추가 효과다.

### 9.6 `C3-RelMPNN-PastAct`

GeomMPNN의 동일 topology/layer 수에 relation value/valid temporal token만 추가한다.
RelPool과 raw edge encoder input schema를 공유한다.

### 9.7 Parameter matching

`parameter_match.py`는 adapter hidden width 후보를 생성하고
`common projector + adapter + heads`의 trainable parameter 수가 target budget에 가장 가까운
값을 선택한다.

- target budget: first implementation에서 RelMPNN hidden 128의 실제 count
- 허용 오차: ±5%
- 허용 width 후보: 64부터 256까지 8 단위
- 오차를 만족하지 못하면 dummy parameter를 추가하지 않고 architecture/config를 조정한다.
- frozen CLaD와 frozen DecisionNCE parameter는 total count에는 보고하지만 matching count에서는
  제외한다.

---

## 10. Loss와 metric

### 10.1 Masked relation BCE

```python
loss_r = BCEWithLogits(logit_r, target_r, pos_weight_r)
loss_relation = sum(mask_r * loss_r) / max(sum(mask_r), 1)
```

`pos_weight_r = min(train_negative_r / max(train_positive_r, 1), 20)`이며 fold train에서만
계산한다. Batch에 valid label이 하나도 없으면 해당 batch를 무시하지 말고 motion loss만
계산하되 count를 기록한다. 이런 batch 비율이 높으면 support gate에서 중단한다.

### 10.2 Motion loss

Train fold의 scene motion scale로 target을 나누고 Smooth-L1을 계산한다. 보고 metric은 다시
meter 단위로 역변환한다.

```text
total_loss = relation_loss + 0.1 * motion_loss
```

Weight는 seed-0 full run 전 고정하고 model별로 바꾸지 않는다.

### 10.3 PR-AUC

- sklearn `average_precision_score`와 동일한 정의를 사용한다.
- relation별 valid sample만 포함한다.
- test target이 single-class면 `null`, reason=`single_class_test`.
- fold macro는 non-null eligible relation의 산술평균이다.
- prediction probability는 float32 sigmoid 값으로 전부 저장한다.

### 10.4 Threshold metric

Relation별 threshold는 validation에서 `[0.05, 0.10, ..., 0.95]` grid의 F1 최대값으로
고정한다. Tie는 더 높은 threshold를 택한다. Test와 모든 perturbation에 같은 threshold를
적용한다.

### 10.5 Bootstrap

Same sample ID의 두 model prediction을 inner join한 뒤 task fold를 outer unit, episode를 inner
unit으로 2,000회 resample한다. 세 task뿐이므로 CI는 descriptive uncertainty로만 쓰며
population-level 확정적 유의성으로 표현하지 않는다.

---

## 11. Runner 상태 기계

```text
PENDING -> PREFLIGHT_PASSED -> RUNNING -> COMPLETED
                         \-> FAILED
RUNNING + last checkpoint -> RESUMED -> COMPLETED/FAILED
```

Run directory에 `RUNNING.json`, `COMPLETED.json`, `FAILED.json` 중 최종적으로 하나만 남긴다.
완료 marker는 checkpoint, prediction, metric, runtime manifest의 존재와 SHA를 확인한 뒤
atomic write한다.

### Run loop

```python
for fold in configured_folds:
    validate_base_checkpoint(fold, seed)
    for model in configured_models:
        key = (protocol, model, fold, seed)
        if completed_and_hash_valid(key):
            continue
        train_one_run(key)
        evaluate_validation_and_test(key)
        validate_artifact_contract(key)
        mark_completed(key)
```

Config의 model 순서는 `Sem, SceneSet, Pair, GeomMPNN, RelPool, RelMPNN`으로 고정해 초기
문제를 단순 모델부터 발견한다. 성능이 중간에 나쁘다는 이유로 뒤 model을 건너뛰지 않는다.

---

## 12. Prediction artifact

한 sample/relation당 한 row를 저장한다.

```json
{
  "protocol": "phase3c-oracle-graph-clad-core-seed0-v1",
  "model_id": "C3-RelMPNN-PastAct",
  "fold": "test_task0",
  "seed": 0,
  "sample_id": "...",
  "task_id": 0,
  "episode_id": "task0_demo_0",
  "current_step": 46,
  "target_step": 52,
  "relation": "left",
  "eligible": true,
  "valid": 1,
  "target": 0,
  "probability": 0.12,
  "threshold": 0.65,
  "prediction": 0,
  "scene_motion_target_m": 0.034,
  "scene_motion_prediction_m": 0.030
}
```

Edge-level diagnostic은 별도 file로 두어 global primary row와 섞지 않는다.

---

## 13. Error handling

| 오류 | 처리 |
|---|---|
| shared graph hash mismatch | data build 즉시 실패, 양쪽 sample ID/hash 기록 |
| future field 발견 | input contract 실패 |
| missing proprio/camera feature | sample drop 금지, build 실패와 source key 기록 |
| DecisionNCE dim ≠ 1024 | projection 자동 추가 금지, semantic gate 실패 |
| relation support 부족 | full training 시작 전 protocol unsupported |
| single-class test relation | metric null, 다른 값으로 대체 금지 |
| NaN/Inf loss | offending sample IDs와 tensor stats 저장 후 run 실패 |
| CUDA OOM | 자동 batch 변경 금지; smoke config를 새 version으로 수정 |
| completed artifact hash 불일치 | overwrite 금지, 새 output version 요구 |
| code/config SHA 변경 후 resume | resume 금지, 새 run으로 시작 |

---

## 14. Test 계획

### Unit

1. `test_phase3c_join_contract.py`
   - adjacent join, episode boundary, hash mismatch, duplicate, poison future action
2. `test_phase3c_targets.py`
   - value/valid/change, OR aggregation, unsupported relation, eligibility
3. `test_phase3c_feature_store.py`
   - key lookup, view order, shape, nonfinite, checkpoint manifest
4. `test_phase3c_dataset.py`
   - normalization train-only, padding/mask, forbidden metadata not in tensor
5. `test_phase3c_clad.py`
   - shape, loss weight, target no-grad, EMA order, eval determinism
6. `test_phase3c_models.py`
   - six model shape, gradients, invariance/equivariance, raw edge token parity
7. `test_phase3c_metrics.py`
   - masked AP, single-class null, validation threshold freeze, calibration
8. `test_phase3c_artifacts.py`
   - atomic write, resume, hash mismatch, duplicate run key

### Integration

- synthetic 3-task/3-fold manifest를 만들어 base smoke와 six-model 2-update run을 끝까지 수행
- 각 prediction row 수가 expected valid sample/relation 수와 같은지 검사
- analyzer가 six model과 trivial baseline을 모두 찾는지 검사

### GPU smoke

- actual DecisionNCE feature를 사용한 `H=1024` base CLaD 100 updates
- 각 core model 100 updates
- peak VRAM, samples/sec, checkpoint size, prediction size 기록
- GPU smoke 결과는 성능 표에 넣지 않는다.

---

## 15. 구현 순서

### Milestone 1 — Contract와 data join

`contracts.py`, `io.py`, joined builder, target builder, support report와 unit test를 먼저 만든다.
이 단계에서 model import는 없어야 한다.

### Milestone 2 — Semantic replay/cache

Camera inventory와 orientation QA를 먼저 구현하고, 그 뒤 DecisionNCE wrapper와 resume 가능한
demo shard extraction을 구현한다.

### Milestone 3 — Controlled CLaD

Dataset semantic/proprio path, LatentDynamics wrapper, loss/EMA trainer, tiny overfit와 실제-feature
GPU smoke를 구현한다.

### Milestone 4 — Structured adapters

Common tensorizer와 SceneSet/Pair부터 만든 뒤 GeomMPNN, 공통 relation edge encoder, RelPool,
RelMPNN 순으로 추가한다. RelPool/RelMPNN raw token parity test를 같은 commit에서 넣는다.

### Milestone 5 — Common trainer/runner/analyzer

Loss, metric, checkpoint, prediction artifact, resume, paired analysis를 연결한다.

### Milestone 6 — SSH runbook와 연구기록

실제 config SHA, camera key, DecisionNCE commit/checkpoint SHA, smoke runtime을 문서에 기록하고
three-fold 실행을 승인한다.

---

## 16. 예정 실행 명령

코드 구현 후 module interface는 다음 형태로 고정한다.

```bash
python -m unittest discover -s tests -p "test_phase3c_*.py"

python -m scripts.phase3c.build_joined_manifest \
  --config configs/phase3c_kcloudvpn_data_smoke_v1.json

python -m scripts.phase3c.validate_action_timing \
  --config configs/phase3c_kcloudvpn_data_smoke_v1.json

python -m scripts.phase3c.build_semantic_feature_store \
  --config configs/phase3c_kcloudvpn_data_smoke_v1.json

python -m scripts.phase3c.train_base_clad \
  --config configs/phase3c_kcloudvpn_core_smoke_seed0_v1.json

python -m scripts.phase3c.run_core \
  --config configs/phase3c_kcloudvpn_core_smoke_seed0_v1.json

python -m scripts.phase3c.run_core \
  --config configs/phase3c_kcloudvpn_core_threefold_seed0_v1.json

python -m scripts.phase3c.analyze_core \
  --config configs/phase3c_kcloudvpn_core_threefold_seed0_v1.json
```

### 16.1 Local → Git → SSH 순서

1. Local workspace에서 Phase 3C unit/integration test를 통과시킨다.
2. Phase 3C 관련 file만 diff로 검토하고 연구기록을 갱신한다.
3. 사용자가 승인한 branch/commit을 push한다.
4. SSH repository `/home/ubuntu/Graph-CLaD`에서 `git pull --ff-only`로 같은 commit을 받는다.
5. `/home/ubuntu/Graph-CLaD/.venv`를 활성화하고 pinned
   `requirements-phase3c.txt`를 설치한다.
6. `GRAPH_CLAD_ARTIFACT_ROOT=/home/ubuntu/graphclad-artifacts`를 설정한다.
7. data/action/camera smoke는 foreground에서 확인한다.
8. semantic full extraction, base 25K, core 18-run은 각각 별도 tmux session에서 순차 실행한다.
9. 각 단계의 `COMPLETED.json`과 runtime manifest를 확인한 뒤 다음 gate로 간다.

Full GPU job을 시작하기 전에 `git rev-parse HEAD`, `git status --short`, `nvidia-smi`, PyTorch
CUDA/device 출력과 resolved config SHA를 runtime manifest에 쓴다. 실행 중 local code를
추가 수정하면 기존 tmux job에는 반영되지 않으므로 그 run의 code snapshot을 기준으로
해석한다.

Notebook cell에서 실행할 때도 shell command는 `!python ...` 또는 `%cd ...`로 실행하고 Python
cell에 그대로 `ls`, `export` 같은 shell 문법을 쓰지 않는다.

---

## 17. Phase 4 연결 계약

Phase 3C winner architecture는 weight가 아니라 **구조와 input schema**를 Phase 4로 넘긴다.
Semantic baseline과 winner의 가장 가까운 fairness control도 함께 넘긴다. RelMPNN이
winner이면 RelPool을 유지해야 policy 수준에서도 relation token과 message passing 효과를
분리할 수 있다. Phase 4에서는 relation classifier head를 버리고 structured output을 원
CLaD의 두 foresight branch에 residual로 연결한다.

```text
pred_p_graph = normalize(pred_p_sem + alpha_p * delta_p_graph)
pred_s_graph = normalize(pred_s_sem + alpha_s * delta_s_graph)
```

- adapter-off 또는 `alpha=0`에서 semantic baseline과 atol/rtol `1e-6` 이내 일치
- Stage 1 original latent/reconstruction objective 사용
- relation auxiliary loss를 추가하려면 `with/without auxiliary` 별도 ablation 필요
- Stage 2가 받는 shape와 normalization은 semantic/graph variant에서 동일
- Phase 3C relation-supervised weight를 가져오는 경우와 random initialization을 분리 보고

이 equivalence test를 통과하기 전에는 Stage 2 policy 코드를 구현하지 않는다.

---

## Version History

| Version | Date | Changes | Author |
|---|---|---|---|
| 1.0 | 2026-08-17 | 파일, schema, tensor, model, loss, runner, test, SSH 실행 interface 확정 | Graph-CLaD 연구팀 |
