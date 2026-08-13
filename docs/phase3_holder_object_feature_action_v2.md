# Phase 3 holder-object feature/action v2 설계

작성일: 2026-08-11  
상태: 로컬 구현 완료, Colab smoke 검증 전

## 결론

현재의 sparse holder-object 방향은 유지한다. 다만 기존 구현을 그대로 확장하지 않고,
입력 feature와 action routing을 `holder_object_v2` 계약으로 보완한다. 핵심 원칙은 다음과
같다.

1. 모든 `robot0-object_i` candidate pair를 사용하며 미래에 실제로 잡히는 object를 미리
   선택하지 않는다.
2. node에는 holder dynamics에 필요한 compact current state만 넣고, pair geometry는
   edge에 둔다.
3. edge에는 현재 상대위치·거리뿐 아니라 raw contact와 directed pair role을 넣는다.
4. 현재 `holding` weak label은 입력하지 않는다. contact, gripper state, action으로 미래
   holding을 예측하게 한다.
5. 6×7 action window를 평탄한 42차원 벡터로만 처리하지 않고 시간축과 gripper channel을
   명시적으로 보존한다.
6. action encoder 개선과 edge-level action conditioning을 서로 다른 모델 비교로 분리한다.

기존 `legacy_v1` 경로는 과거 결과 재현을 위해 그대로 보존한다. v2는 명시적으로
`--feature-contract holder_object_v2`를 지정했을 때만 활성화된다.

## 기존 구현에서 확인된 문제

| 항목 | 기존 구현 | 문제 | v2 보완 |
|---|---|---|---|
| sparse edge | 상대위치 3, 거리, geometry validity만 사용 | 문서에 적힌 current contact가 실제 model input에 없었음 | contact value/validity와 edge direction 추가 |
| sparse node set | edge만 줄이고 fixture/site node는 고립된 채 유지 | normalization과 flat baseline에는 계속 영향을 줌 | v2에서는 robot과 movable object node만 유지 |
| node | 24차원 공통 vector와 type one-hot | 절대위치와 전체 joint position이 task 진행 순서 shortcut이 될 수 있음 | 절대위치·joint position을 뺀 17차원 compact state |
| B1 pair MLP | source/target node와 action만 사용 | GNN이 받는 pair geometry를 받지 않아 B1 대 GNN 비교가 불공정 | `b1_pair_feature_mlp_v2`에 동일 edge feature 제공 |
| action | 6×7을 42차원으로 flatten한 뒤 하나의 MLP | 시간 순서, 누적 command, gripper 변화가 약하게 표현됨 | arm/gripper 분리, temporal position, summary feature 사용 |
| G1 대 G3 | action routing 외에 gate, residual, LayerNorm도 동시에 변경 | action-conditioning 효과와 block 변경 효과가 섞임 | 같은 v2 block에서 한 요소씩 바꾸는 3단계 비교 |
| current auxiliary head | future action이 섞인 representation으로 current relation도 예측 | current loss가 action 정보를 지우도록 압박할 수 있음 | current head는 action-free pair state, future head만 action-conditioned state 사용 |

따라서 기존 B1의 낮은 성능만으로 pair-only baseline이 실패했다고 강하게 결론 내리면 안
된다. feature parity가 보장된 v2 B1을 다시 측정해야 한다.

## v2 sparse graph 계약

### Candidate topology

- node: `robot0`과 모든 movable `object_i`; fixture/site node는 v2 graph에서 제거
- directed edge: `robot0 -> object_i`, `object_i -> robot0`
- fixture/site 및 object-object complete edge는 첫 실험에서 제외
- message passing: 1 layer
- node update: residual + LayerNorm
- prediction: 각 directed pair의 future relation
- holding 평가는 validity가 있는 `robot0 -> object_i` output에 의해 결정

양방향 star를 쓰는 이유는 object가 robot state를 받고, robot도 현재 주변 object 상태를
제한적으로 모을 수 있게 하기 위해서다. complete graph처럼 무관한 fixture/site message를
섞지는 않는다.

### Compact node feature: 17차원

| Feature | 차원 | 이유 |
|---|---:|---|
| node type one-hot | 4 | robot/object 역할 구분 |
| position validity | 1 | pair geometry 신뢰 여부 |
| gripper qpos | 2 | 현재 finger opening 상태 |
| gripper validity | 1 | unknown과 실제 0 구분 |
| gripper aperture (`max(abs(qpos))`) | 1 | closure 정도를 작은 모델에 직접 제공 |
| robot joint velocity | 7 | 현재 motion state 보존 |
| joint velocity validity | 1 | unknown과 정지 구분 |

제외한 항목은 absolute position, joint position, object-of-interest flag이다. 상대 geometry는
edge에 이미 있으며, object-of-interest는 oracle target shortcut이 될 수 있다. joint
position은 action-to-motion 변환에 도움이 될 가능성이 있으므로 v2가 실패할 경우 별도
ablation으로만 복원한다.

### Holder-object edge feature: 9차원

| Feature | 차원 | 이유 |
|---|---:|---|
| target-minus-source relative position | 3 | pair 방향과 공간 관계 |
| Euclidean distance | 1 | 접근/이탈 정도 |
| geometry validity | 1 | missing geometry 구분 |
| current raw contact value | 1 | grasp 전제와 hard negative 구분 |
| current contact validity | 1 | false와 unknown 구분 |
| robot-to-object flag | 1 | directed holder edge 역할 |
| object-to-robot flag | 1 | reverse context edge 역할 |

현재 holding relation은 넣지 않는다. 이 label은 contact, closure, 과거 relative stability로
만든 weak label이므로 입력하면 persistence 복사와 label-rule 재현으로 성능이 부풀 수 있다.
필요하면 나중에 `with-current-holding`을 별도 persistence ablation으로만 측정한다.

## structured action embedding

Frozen manifest의 action은 `6 steps × 7 dims` OSC-Pose command이다. 마지막 channel은
gripper command이고 앞의 6개 channel은 arm translation/rotation command로 취급한다.

v2 encoder는 다음 순서로 계산한다.

1. train split의 모든 유효 timestep을 사용해 action channel별 normalization을 fit한다.
2. 각 timestep에서 arm 6차원과 gripper 1차원을 별도 MLP로 encoding한다.
3. learned temporal position embedding을 더해 같은 값이라도 window 안 위치를 구분한다.
4. masked attention으로 step representation을 모은다.
5. `first`, `last`, `mean`, `sum`, `last-first`, gripper min/max, valid-step coverage를
   별도 summary MLP로 encoding한다.
6. attention representation과 summary representation을 residual 형태로 합친다.

이 구조는 recurrent model보다 작으면서도 holding에 중요한 세 가지 신호를 직접 보존한다.

- gripper를 닫는가/여는가
- command가 한 순간인지 여러 step 동안 지속되는가
- window 시작과 끝 사이에 command가 바뀌는가

### Pair-conditioned FiLM message

v2 edge-conditioned model은 동일한 global action을 단순 broadcast한 뒤 scalar gate만 곱하지
않는다. 먼저 pair state를 만들고 action과의 element-wise interaction을 사용한다.

```text
pair_h = pair_encoder(robot_h, object_h, edge)
condition = concat(pair_h, action_h, pair_h * action_h)
gamma, beta, gate = action_film(condition)
message = sigmoid(gate) * ((1 + tanh(gamma)) * pair_h + beta)
```

FiLM layer는 `gamma=0`, `beta=0`, `gate bias=4`로 초기화한다. 따라서 학습 시작점은
기본 sparse pair message와 거의 같고, 이후에만 action에 따른 modulation을 학습한다.
이는 S-LS와 S-EF의 차이가 무작위 초기 modulation에서 비롯되는 것을 줄인다.

scalar gate는 message 전체 크기만 바꿀 수 있지만 FiLM은 action에 따라 message 각 차원의
내용과 방향을 바꿀 수 있다. `pair_h * action_h`는 같은 action도 가까이 접촉한 object와 먼
object에서 다르게 작동하도록 object-specific interaction을 노출한다.

현재 relation 보조 head는 초기 node pair와 current edge feature만 사용한다. future action이
들어간 message와 final action concat은 future head에만 연결한다. 따라서 current auxiliary
loss가 action embedding을 불필요한 noise로 보고 제거하도록 만드는 경로를 차단한다.
current/future label의 positive ratio도 서로 다를 수 있으므로 두 head의 positive weight는
각각 train split에서 따로 계산한다.

## 원인을 분리하는 v2 모델 비교

| 비교 ID | 코드 model ID | Action encoder | Message에 action 사용 | 확인 질문 |
|---|---|---|---|---|
| B1-v2 | `b1_pair_feature_mlp_v2` | flat | message 없음 | pair feature만으로 충분한가 |
| S-LF-v2 | `g2_flat_action_holder_object_gnn_v2` | flat | 아니오 | v2 sparse block 기준점 |
| S-LS-v2 | `g2_structured_action_holder_object_gnn` | structured | 아니오 | structured embedding 자체가 유효한가 |
| S-EF-v2 | `g3v2_action_film_holder_object_gnn` | structured | FiLM | edge-level action conditioning이 추가로 유효한가 |

핵심 paired comparison은 다음과 같다.

- `B1-v2 vs S-LF-v2`: 동일 pair feature에서 graph message passing 효과
- `S-LF-v2 vs S-LS-v2`: action embedding 방식의 효과
- `S-LS-v2 vs S-EF-v2`: action encoder를 고정한 edge conditioning 효과

기존 G1/G3도 같은 v2 input으로 함께 재측정하지만 legacy 참고선으로만 사용한다.

## 실행과 판정 순서

1. feature leakage, dimension, action mask unit test를 통과한다.
2. frozen `test_task0`, seed 0, 동일 sample ID에서 v2 smoke를 실행한다.
3. natural test와 holding challenge test를 따로 보고 위 세 paired comparison을 계산한다.
4. parameter count 차이를 기록한다. smoke 이후 정식 비교에서는 hidden width를 조정해
   parameter-matched 결과도 낸다.
5. correct, physical-zero action, whole-action shuffle, reverse-window,
   gripper-only shuffle, arm-only shuffle, shuffled-edge를 확인한다.
6. 유망한 action model에 adjacent-window temporal shift를 추가한다.
7. 그 뒤에 기존 계획의 C-L, C-E, S-0와 3 folds × 3 seeds로 확장한다.

### 구현된 holding 평가 및 checkpoint 계약

`changed_relation`은 과거 report 호환을 위해 유지하되, 이름 그대로 실제 change-event를
뜻하지 않는다. 실제 holding 평가는 별도 `holding` block으로 저장한다.

| Metric | 정의 |
|---|---|
| `future_state` | 모든 current/future-valid holder pair에서 future holding state |
| `change_event` | `current_holding != future_holding` 여부 |
| `onset` | current holding이 0인 pair에서 future holding 1 여부 |
| `release` | current holding이 1인 pair에서 future holding 0 여부 |
| `future_value_on_true_changed` | 기존 `changed_relation` holding 값의 명확한 이름 |
| `hard_negative.false_positive_rate` | endpoint contact가 있고 current/future holding이 모두 0인 pair에서 holding 예측 비율 |
| `pr_auc` | threshold-free average precision |

Holding threshold는 매 epoch의 validation correct-action output에서 0.05~0.95를 0.01
간격으로 탐색하여 actual holding change-event F1을 최대화한다. 선택된 checkpoint의
threshold는 natural test, challenge test, no-action/shuffle controls에 변경 없이 적용한다.
Validation에 holding change positive가 없을 때만 future holding F1, 그마저 지원되지 않을
때만 전체 future-relation macro-F1로 fallback한다.

Checkpoint primary score도 같은 actual holding change-event F1이다. 따라서 전체 spatial
relation macro-F1이 높지만 holding이 낮은 모델이 자동으로 선택되는 문제를 방지한다.

### Parameter matching 계약

P0는 raw flattened node/action을 직접 사용하므로, 사용하지 않는 node/action encoder를 더
이상 생성하거나 parameter count에 포함하지 않는다. Parameter count는
`requires_grad=True`인 실제 model parameter만 계산한다. Smoke는 shape/gradient 확인을 위해
동일 hidden width를 허용하지만, 최종 3 folds × 3 seeds 실행에서는 각 모델 width를 target
parameter budget에 가장 가깝게 선택하고 실제 count와 target 차이를 report에 저장한다.

v2 smoke 실행 예시는 다음과 같다.

```bash
python scripts/phase3/run_holder_action_smoke.py \
  --manifest /content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/phase3B_R1_eval_manifest.json \
  --output /content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/smoke_test_task0_seed0_feature_v2.json \
  --fold test_task0 \
  --seed 0 \
  --feature-contract holder_object_v2
```

## 아직 넣지 않는 feature

- relative velocity
- object-following stability
- past graph history
- object orientation
- fixture/support context
- current holding weak label

앞의 세 temporal feature는 현재 sample에 full past graph sequence가 없어서 정확히 계산할 수
없다. v2 smoke에서 structured action을 사용해도 onset/release timing이 부족할 때 Phase 2D
timeline을 확장하여 추가한다.

기존 report의 `no_action`은 normalized tensor를 0으로 만든 값이라 raw command 기준으로는
train-mean action이다. 결과 호환을 위해 이 이름은 남기되, v2 판정에는 train 통계로 변환한
실제 raw zero command인 `physical_zero_action`을 사용한다. 이것도 action 없이 재학습한
`S-0`을 대신하지는 않는다.

## S-0 및 complete-topology 후속 구현

후속 runner는 다음 대조군을 제공한다.

| ID | 코드 model ID | 의미 |
|---|---|---|
| S-0 | `s0_no_action_holder_object_gnn_v2` | v2 residual sparse block에서 action 제거 |
| S-0-G1 | `s0_g1_no_action_holder_object_gnn_v2` | 기존 G1 구조에서 action encoder와 final concat만 제거 |
| C-L | `c_l_complete_late_action_gnn_v2` | S-LS와 같은 구조, complete message topology |
| C-E | `c_e_complete_action_film_gnn_v2` | S-EF와 같은 구조, complete message topology |

Complete topology도 robot/object node만 사용한다. Object-object edge는 message-only이며
robot-object edge만 loss, validation threshold fitting, test evaluation에 사용한다. Sparse
train normalization을 complete view에도 재사용한다. 따라서 S-LS/C-L과 S-EF/C-E는 topology
이외의 조건과 parameter count가 동일하다.

`--only`로 특정 대조군만 실행할 수 있고, `--parameter-match`와
`--target-parameter-count`로 action-free control의 용량을 대응 모델에 맞출 수 있다.
현재 task-0/seed-0 결과와 해석은
`docs/phase3_topology_action_followup_result.md`에 기록되어 있다.

## 구현 위치

- feature/action/model: `scripts/phase3/offline_probe.py`
- fixed-manifest smoke entry point: `scripts/phase3/run_holder_action_smoke.py`
- S-0/C-L/C-E follow-up entry point: `scripts/phase3/run_topology_action_followup.py`
- regression tests: `tests/test_phase3_holder_object_features.py`
