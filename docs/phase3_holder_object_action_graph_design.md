# Target-centric holder–object graph 설계 초안

> 상태: architecture 초안으로 보존한다. 평가 protocol, 2 x 2 ablation,
> 실행 gate의 현재 기준은
> `docs/01-plan/features/phase3_target_centric_action_conditioned_gnn.plan.md`다.

## 목적

현재 complete geometry graph는 모든 위치-valid node 사이에 message를 전달한다. Holding은 scene 전체의 일반 공간 관계보다 robot gripper와 특정 object 사이의 접촉·폐합·상대 운동·action에 의해 결정된다. 다음 GNN은 이 구조적 가정을 edge와 update에 직접 반영한다.

## 그래프 범위

- primary target edge: `robot0 -> object_i`
- optional reverse edge: `object_i -> robot0`
- context edge: target object와 직접 접촉하거나 support 관계를 가진 object/fixture만 포함
- 나머지 complete geometry edge는 제거하거나 별도 context channel로 분리
- future relation, event category, holding label은 input feature로 사용하지 않음

## Holder–object edge feature

현재 시점에서만 계산한다.

- gripper–object relative position과 distance
- gripper–object relative velocity 또는 직전 관측 대비 변화량
- raw contact value와 validity mask
- gripper qpos, closure 정도, closure velocity
- object-following stability evidence의 과거 관측값
- object/edge validity mask
- target-object indicator는 task instruction이나 fixed object-of-interest contract가 허용하는 경우에만 사용

## Action-conditioned temporal edge

Action window를 scene-level vector로 한 번 더 붙이는 대신 holder–object edge update에 직접 조건화한다.

```text
edge_t(robot, object)
  + robot_t
  + object_t
  + action_window(t:t+tau)
      -> temporal edge encoder
      -> gated holder-object message
      -> future holding/contact relation head
```

Action feature 후보는 end-effector translation/rotation command, gripper command, window 내 누적량, 마지막 command, command 변화량이다. Action은 train split에서만 normalization한다.

## 최소 ablation

| ID | 구조 | 확인 질문 |
|---|---|---|
| B0 | flat MLP | 현재 최강 baseline |
| B1 | target-object-only MLP | P0가 non-target context를 실제로 쓰는가 |
| G0 | complete geometry GNN | 기존 GNN 기준선 |
| G1 | holder–object edge GNN | target-centric topology 자체가 유효한가 |
| G2 | G1 + global action concat | topology와 기존 action 방식의 조합 |
| G3 | G1 + action-conditioned temporal edge | action을 edge update에 넣는 이점이 있는가 |
| G4 | G3 + restricted context edges | 주변 support/contact context가 추가로 필요한가 |

## 필수 controls

- correct action / no action / shuffled action
- train-time shuffled-action model
- correct holder–object edge / shuffled target edge
- node-order permutation
- non-target node masking
- 동일 parameter budget과 동일 optimizer/update 수
- 동일 fixed split과 동일 selected sample IDs

## 평가

- primary: holding changed F1, holding onset F1, holding release F1
- secondary: 전체 changed-relation macro F1
- hard-negative false-positive rate
- task/fold별 paired P0 대비 차이
- episode/event bootstrap confidence interval
- natural test와 holding challenge test를 별도 보고

## 구현 gate

1. 새 edge가 future state나 target category를 읽지 않는 unit test
2. holder–object edge coverage와 validity QA
3. action shuffle이 sample ID를 보존하는 paired-control test
4. node permutation에서 GNN 출력이 permutation-equivariant한지 확인
5. smoke run 후에만 3 folds × 3 seeds full run
