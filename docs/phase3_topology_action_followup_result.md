# Phase 3B topology/action 후속 결과

날짜: 2026-08-12  
범위: `test_task0`, seed 0 smoke  
Primary: actual holding change-event F1

## 재현성

Persistent root:
`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1`

- Manifest: `phase3B_R1_eval_manifest.json`
- S-0/C-L/C-E result: `smoke_test_task0_seed0_topology_action_followup_v2.json`
- Parameter-matched control: `smoke_test_task0_seed0_s0_parammatched_v2.json`,
  `smoke_test_task0_seed0_s0_g1_parammatched_v2.json`
- Same-width control: `smoke_test_task0_seed0_s0_g1_hidden48_v2.json`
- Decision: `phase3_topology_action_decision_task0_seed0_v3.json`
- Snapshot: `code_snapshot_v2_followup`
- QA: A100에서 feature/topology/action invariance/model/metric/threshold/checkpoint/parameter
  accounting 22 tests 통과.

## Topology 통제 계약

C-L/C-E는 sparse counterpart와 같은 robot/object node를 쓴다. Object–object edge는
message passing에만 추가하고 loss, threshold fitting, evaluation은 같은 directed
robot–object prediction edge에 제한한다. Sparse-train normalization을 재사용한다.
따라서 paired sparse/complete model은 parameter, target, sample ID, normalization이 같고
message topology만 다르다.

## 결과

| Model | Hidden | Parameters | Natural F1 | Stress F1 | Natural AP | Stress AP |
|---|---:|---:|---:|---:|---:|---:|
| G1 sparse + late action | 48 | 44,946 | **0.4257** | **0.8000** | **0.4426** | 0.5767 |
| S-0-G1 same width | 48 | 35,778 | 0.2835 | 0.6852 | 0.2719 | **0.6770** |
| S-0-G1 parameter matched | 54 | 44,784 | 0.3204 | 0.6667 | 0.4225 | 0.6632 |
| S-0 v2 block matched | 55 | 59,363 | 0.3022 | 0.6239 | 0.3037 | 0.4538 |
| S-LS sparse structured late | 48 | 60,235 | 0.3015 | 0.6452 | 0.3703 | 0.4941 |
| C-L complete structured late | 48 | 60,235 | 0.2589 | 0.4324 | 0.1320 | 0.4076 |
| S-EF sparse structured FiLM | 48 | 74,300 | 0.3321 | 0.7500 | 0.3498 | 0.5960 |
| C-E complete structured FiLM | 48 | 74,300 | 0.2636 | 0.4486 | 0.1230 | 0.5014 |

Parameter-identical topology 차이:

| 비교 | Natural | Stress |
|---|---:|---:|
| S-LS−C-L | +0.0426 | +0.2127 |
| S-EF−C-E | +0.0684 | +0.3014 |
| C-E−C-L | +0.0047 | +0.0162 |

Sparse topology가 late-action과 edge-FiLM 모두 두 view에서 우세했다. Complete graph 안의
edge conditioning 이득은 거의 없었다. Complete-model edge shuffle에서 F1이 오히려
상승해 추가 object–object context가 이 smoke에서는 유용한 relation evidence가 아니라
noise로 작용했다. C-L/C-E는 확대하지 않는다.

## Action 해석

G1 action pathway를 제거하면 다음만큼 F1이 낮아졌다.

| 비교 | Natural G1 gain | Stress G1 gain |
|---|---:|---:|
| G1−same-width S-0-G1 | +0.1422 | +0.1148 |
| G1−parameter-matched S-0-G1 | +0.1054 | +0.1333 |

이는 task-0/seed-0에서 action pathway의 존재가 도움이 됐다는 뜻이지 올바른
sample-specific action을 사용했다는 증거는 아니다. Correct vs shuffled, arm/gripper,
reversed-window control은 작거나 불일치했다. Extra optimization pathway, action
distribution bias, coarse nonzero-action cue 가능성이 남았다.

후속 three-seed action gate에서 aligned G1은 train-shuffled보다 두 view event F1이 3/3
seed에서 높았다. 하지만 첫 test-time shuffle donor의 95.8–98.4%가 같은 episode여서
약한 control임을 확인했다. Corrected global episode-disjoint shuffle에서는 event PR-AUC가
두 view 3/3 seed에서 하락했다. Task 0의 provisional action-use 근거는 있지만 task-family
generalization은 검증하지 않았다. Authoritative 결과는
`docs/phase3_action_semantics_gate_result.md`다.

## 현재 결정

Sparse holder–object topology를 유지하고 complete topology와 추가 FiLM complexity를
확대하지 않는다. G1은 thresholded F1 candidate, B1-v2는 pair-only/AP 및 hard-negative
baseline으로 유지한다. 이 문서 자체의 topology 결론은 one fold/seed 한계가 있다.
