# Phase 2A: GraphSpec과 deterministic graph extractor

> **2026-08-06 roadmap v3 참고:** 이 문서와 `phase2.v1`은 정적 extractor의
> regression baseline이다. 현재 main dataset은 official demonstration을 state replay해
> 전체 graph timeline과 temporal relation event를 만드는 Phase 2D 계약을 따른다.
> `is_object_of_interest`와 BDDL-derived task relevance는 primary model input에서 제외한다.

> **2026-08-05 문헌 재평가:** `phase2.v1`은 유효한 static extraction baseline이지만
> 최종 relational-dynamics dataset specification은 아니다. Phase 2A는 통과했고 action
> trajectory, relation target, episode split은 이후 Phase 2D에서 다시 정의했다.

## 목적

Phase 1의 raw LIBERO state 계약을 deterministic graph representation으로 변환한다.
이 단계는 GNN을 학습하지 않으며 RGB policy에서 simulator state를 쓸 수 있다고 주장하지
않는다. 첫 graph는 CLaD와 Graph-CLaD의 통제 비교를 위한 oracle graph다.

## Graph 경계

각 snapshot을 directed graph 하나로 변환한다. Node는 다음과 같다.

- `robot0`: end-effector position과 robot state.
- `object`: 이동 가능한 task object.
- `fixture`: 고정 scene element.
- `site`: valid position이 있는 task-local site.

Logical environment name을 node identity로 사용한다. `body_id`는 runtime audit metadata에만
남기고 node identity와 model feature에서는 제외한다.

## Phase 2 v1 node feature

공통 numeric vector는 24개 값으로 구성된다.

```text
position(3), position_valid(1), is_object_of_interest(1),
gripper_qpos(2), gripper_qpos_valid(1),
joint_pos(7), joint_pos_valid(1),
joint_vel(7), joint_vel_valid(1)
```

Node type은 별도의 4-way one-hot이다. Missing numeric value는 0과 validity mask를 함께
사용해 실제 0과 구분한다. Raw quaternion은 audit payload에 보존하지만 v1 model feature
vector에서는 제외한다. Ordering을 확인하기 전에 normalize하거나 learned feature로 쓰지
않는다.

## Phase 2 v1 edge 정책

Valid position node 사이에 self-loop가 없는 complete directed spatial graph를 만든다.
각 directed edge는 다음을 가진다.

```text
relative_position = target_position - source_position  (3)
distance                                              (1)
distance_valid                                        (1)
```

v1에서는 distance threshold를 쓰지 않는다. 이 단계의 임의 threshold는 long-range
dependency를 숨길 수 있다. 이후 complete, radius-limited, predicate-only topology를
통제 ablation으로 비교할 수 있다.

## Predicate와 오류 정책

Phase 1에서 `get_joint_state`, `is_open`, `is_close`가 object, fixture, site wrapper에
일관되게 구현되지 않았음을 확인했다. 따라서 다음 규칙을 쓴다.

- 지원하는 boolean은 `valid=1`로 저장한다.
- Missing value는 `value=null, valid=0`으로 저장한다.
- Catch한 exception도 error text와 함께 `value=null, valid=0`으로 저장한다.
- Unknown을 `false`로 조용히 바꾸지 않는다.
- Semantic predicate edge는 `phase2.v1`에서 audit-only다.

현재 graph를 geometric하게 유지하고 predicate coverage를 숨겨진 label noise가 아니라
가시적인 향후 ablation으로 남긴다.

## 시간 및 leakage 정책

Graph는 source snapshot `step` 순서를 보존하고 logical identity로 node를 align한다.
Future state, success, reward, terminal 정보는 node/edge feature에 넣지 않는다.
Normalization statistics는 training episode에서만 fit한다.

## 실행

```powershell
python -m scripts.phase2a.graph_extractor `
  --input data/phase1_libero_state_capture.json `
  --spec configs/phase2_graph_spec.json `
  --output data/phase2_graph_sequence.json
```

로컬 Phase 1 file은 compact manifest이므로 위 명령에는 Colab에서 생성한 full raw
capture가 필요하다. Extractor 자체는 dependency-free snapshot fixture로 로컬에서도
검사한다.

## Live 검증 결과

Full Phase 1 capture에 Colab에서 extractor를 실행했다. Graph 2개가 생성됐고 각각 node
24개(로봇 1, object 5, fixture 2, site 16)였다. Complete directed topology는 graph당
edge 552개였고 모든 node position이 valid였으며 동일한 24-dimensional feature vector를
사용했다.

Predicate audit은 graph당 error 55개, available value 12개, unsupported value 2개를
보고했다. 이는 Phase 1에서 확인한 object-specific API 경계 때문에 예상된 결과다.
오류를 false label로 만들지 않고 unknown으로 유지했다. Compact live 결과는
`data/phase2_live_graph_manifest.json`에 있다.

## 파일

- `configs/phase2_graph_spec.json`: 고정된 v1 계약.
- `scripts/phase2a/graph_extractor.py`: dependency-free extractor와 CLI.
- `tests/test_phase2_graph_extractor.py`: topology, mask, identity, predicate, temporal 검사.
