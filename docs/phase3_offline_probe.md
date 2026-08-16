# Phase 3 historical offline relational-dynamics probe

> 이 문서는 과거 probe 기록이다. 현재 official-demo holding summary는
> `data/phase3_holding_target_v2_summary.json`,
> `data/phase3_holding_target_balanced_v3_summary.json`이다. 현재 claim limit은
> `docs/literature_alignment_and_phase_reassessment.md`와
> `docs/phase3_holder_object_action_graph_design.md`를 따른다.

> 아래 수치는 scripted/bounded Phase 2R dataset의 diagnostic 결과이며 Phase 4 진입
> 근거가 아니다. Historical `changed_relation`은 실제 change-event classifier F1이 아니라
> 실제로 변한 위치의 future relation value F1이었다. 현재 code는 이 field를 호환성 때문에
> 유지하면서 holding change-event/onset/release/future-state/hard-negative/PR-AUC를 별도
> block으로 보고한다.

## 목적

Original CLaD latent-dynamics에 통합하기 전에 relational structure가 action-conditioned
future relation prediction을 개선하는지 본 초기 probe다. Validated Phase 2R oracle graph를
입력으로 쓰며 RGB perception 실험이 아니다.

## Protocol

- Dataset: `phase2r.v2-robot-base`, `include_all_sites`, `tau=6`.
- Input: current node feature, geometry edge feature, six-action window.
- Target: `left/right/front/behind/above/below/contact/on/inside/holding` future edge label.
- Current relation auxiliary weight 0.25.
- Unknown label은 false가 아니라 mask 처리.
- Normalization은 train sample에서만 fit.
- Episode-disjoint split: train 532, validation 19, test 19.
- 모든 model에 Colab CUDA에서 seeds 3개 실행.

## 모델과 초기 결과

| ID | 구조 | Changed macro-F1 | Std | Parameters |
|---|---|---:|---:|---:|
| P0 | flat node-set MLP | 0.2640 | 0.0330 | 92,244 |
| P1 | independent node, no message | 0.7873 | 0.0811 | 43,988 |
| P2 | fully connected GNN, empty edge | **0.8576** | 0.0496 | 60,692 |
| P3 | geometry-edge GNN | 0.8506 | 0.0305 | 78,036 |
| P4 | geometry edge + soft attention | 0.7896 | 0.0699 | 86,677 |

Control은 retraining 없이 zero action, batch-shuffled action, shuffled sender/receiver를
평가했다. P2가 이 split에서 가장 높았고 P3/P4는 edge shuffle에 크게 하락해 edge
assignment 사용 신호가 있었다. 그러나 P3가 P2를 이기지 못해 geometry-edge superiority는
입증되지 않았다.

Summary는 `data/phase3_offline_probe_summary.json`, 당시 full report는
`/content/Graph-CLaD-phase2r-scaleup-r3/data/phase3_offline_probe_report.json`에 있었다.

## 범위 한계

Inherited test split은 `libero_spatial:task9:init1:seed0` 하나뿐이고 changed support는 주로
`above/below/on`이었다. `inside/holding` test support는 없었다. 따라서 controlled spatial
split의 relational signal일 뿐 모든 semantic relation에 대한 주장이 아니다. Parameter도
exact matched가 아니었다.

## Task-family-held-out near-matched 재검증

570 samples를 train 380, validation 76, test 114로 family-isolated split했다. Validation은
`libero_90:1`, `libero_spatial:8`; test는 `libero_90:2`, `libero_goal:0`,
`libero_object:0`이었다. 모든 model에 seeds 3개, 약 70k parameter width를 사용했다.

| Model | Future macro-F1 | Changed macro-F1 | Inside F1 |
|---|---:|---:|---:|
| P0 | 0.6585 ± 0.0124 | 0.1394 ± 0.0299 | 0.2309 ± 0.1075 |
| P1 | 0.6081 ± 0.0010 | 0.1289 ± 0.0169 | 0.1181 ± 0.0249 |
| P2 | 0.5819 ± 0.0083 | **0.2223 ± 0.0676** | 0.0623 ± 0.0881 |
| P3 | 0.6229 ± 0.0099 | **0.2222 ± 0.0361** | 0.2018 ± 0.0392 |
| P4 | 0.6070 ± 0.0154 | 0.1998 ± 0.0425 | 0.0791 ± 0.1119 |

P2/P3의 changed-relation advantage는 남았지만 correct action이 no/shuffled action보다
일관되게 높지 않았고 shuffled edge도 일관되게 낮지 않았다. Action-conditioned foresight나
causal edge use는 입증되지 않았다.

Holding label 0 문제는 `robot0*`만 인식하던 collector가 runtime의 `gripper0_*` body를
놓치고 closure threshold가 지나치게 엄격했던 것이 원인이었다. `gripper0_*` contact를
logical `robot0`에 mapping하고 `|qpos| <= 0.025`로 calibration했다. Pilot은 positive 10,
transition 10, six-episode scale-up은 transition 34를 만들었지만 task 0/1에만 positive가
있고 task 2에는 contact만 있었다. Task 2 96-step 추가 probe도 transition 0이었다.

결과는 `data/phase3_taskheldout_matched_summary.json`, holding follow-up은
`data/phase2r_holding_probe_summary.json`에 있다. 이후 main path를 official-demo Phase 2D로
전환했고 이 historical probe는 Phase 4 근거에서 제외했다.
