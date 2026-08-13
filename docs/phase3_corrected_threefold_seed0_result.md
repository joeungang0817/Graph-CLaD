# Phase 3B corrected three-fold seed-0 gate 결과

날짜: 2026-08-12  
결정: GNN three-seed 확대를 중단하고 pair-local temporal encoder로 전환한다.
Phase 4는 계속 차단한다.

## 범위와 protocol

Held-out task 0/1/2, seed 0에서 B1-v2, G1 late-action, S-0, matched
train-shuffled G1을 비교했다. Task 1은 initial smoke 결과를 재사용했고 후속 run은 task
0/2만 실행해 총 12 runs를 결합했다.

Checkpoint와 future/current threshold는 natural validation에서만 선택했다. Threshold는
natural test와 event-enriched stress subset에 고정했다. Primary는 natural-test
conditional/oracle-current holding-event PR-AUC다. Stress subset은 독립 test나 challenge
generalization으로 해석하지 않는다.

## Natural-test 결과

| Task | Model | Conditional PR-AUC | Event F1 | End-to-end PR-AUC | Release F1 | Hard-negative FPR |
|---:|---|---:|---:|---:|---:|---:|
| 0 | B1-v2 | 0.3566 | 0.3396 | 0.2199 | 0.0000 | 0.5524 |
| 0 | G1 | 0.2405 | 0.3309 | 0.1729 | 0.2537 | 0.2190 |
| 0 | S-0 | 0.4796 | 0.2970 | 0.2782 | 0.0513 | 0.4762 |
| 0 | G1 train-shuffled | 0.4719 | 0.3882 | 0.2034 | 0.2946 | 0.1143 |
| 1 | B1-v2 | 0.3710 | 0.4091 | 0.2595 | 0.0000 | 0.4800 |
| 1 | G1 | 0.5032 | 0.3276 | 0.2190 | 0.1047 | 0.2000 |
| 1 | S-0 | 0.4241 | 0.3897 | 0.1296 | 0.0000 | 0.4700 |
| 1 | G1 train-shuffled | 0.4994 | 0.4211 | 0.2351 | 0.2360 | 0.1500 |
| 2 | B1-v2 | 0.4100 | 0.4762 | 0.2576 | 0.0000 | 0.4615 |
| 2 | G1 | 0.5817 | 0.3229 | 0.3751 | 0.1293 | 0.0769 |
| 2 | S-0 | 0.5075 | 0.3039 | 0.3053 | 0.1197 | 0.2308 |
| 2 | G1 train-shuffled | 0.5244 | 0.4908 | 0.0627 | 0.0000 | 0.4519 |

Task-macro mean:

| Model | Conditional PR-AUC | Event F1 | End-to-end PR-AUC | End-to-end F1 | Release F1 | Hard-negative FPR |
|---|---:|---:|---:|---:|---:|---:|
| B1-v2 | 0.3792 | 0.4083 | 0.2456 | 0.0721 | 0.0000 | 0.4980 |
| G1 | 0.4418 | 0.3272 | 0.2557 | 0.2278 | 0.1626 | 0.1653 |
| S-0 | 0.4704 | 0.3302 | 0.2377 | 0.1515 | 0.0570 | 0.3923 |
| G1 train-shuffled | 0.4985 | 0.4333 | 0.1671 | 0.0994 | 0.1769 | 0.2387 |

Natural validation에서 선택한 threshold는 model/task별로 크게 달랐고 G1은 흔히
0.91–0.95를 선택했다. 따라서 PR-AUC를 primary로 하고 thresholded F1을 secondary로
유지한다. Brier/ECE는 aggregate와 각 run artifact에 저장했다.

## Paired hierarchical bootstrap

Task fold, episode, event cluster를 재표집했다. Task당 seed 하나뿐이므로 seed uncertainty나
넓은 task-family generalization이 아닌 held-out episode/event uncertainty다.

| 비교와 metric | 추정값 | 95% CI |
|---|---:|---:|
| G1−B1 conditional PR-AUC | +0.0626 | [−0.0892, +0.1905] |
| G1−B1 event F1 | −0.0811 | [−0.2017, +0.0294] |
| G1−B1 release F1 | +0.1626 | [+0.0701, +0.2967] |
| G1−B1 hard-negative FPR | −0.3326 | [−0.4201, −0.2429] |
| G1−shuffled conditional PR-AUC | −0.0567 | [−0.1943, +0.0572] |
| G1−shuffled event F1 | −0.1062 | [−0.2001, −0.0379] |
| G1−shuffled release F1 | −0.0143 | [−0.1476, +0.1320] |
| G1−shuffled hard-negative FPR | −0.0734 | [−0.3624, +0.1162] |

## Gate 판단

- G1이 B1 PR-AUC를 최소 2개 task에서 이김: 통과(task 1/2).
- Hard-negative FPR의 심각한 악화를 피함: 통과(3개 task 모두 개선).
- Release 개선: 통과(3개 task 모두 B1보다 개선).
- Train-time action shuffle에서 aligned G1이 일관되게 하락: 실패.

Primary gain은 task-dependent이고 CI가 0을 포함한다. Thresholded F1은 B1보다 낮고
train-shuffled G1이 task-macro conditional metric에서 더 강하다. 추가 GNN 9 runs를
정당화하지 못하므로 GNN depth/message passing 확대를 중단한다.

다음 architecture는 target-pair temporal이다. Robot–object pair를 독립 encode하고 t 이하
causal history만 추가하며 context는 DeepSets 또는 pair-set attention으로 제한한다.
H0–H3 factorial로 missing temporal state와 action conditioning을 분리한다.

## 재현성 artifact

Persistent root:
`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/corrected_protocol_v2`

Combined directory: `threefold_seed0_combined`

- `phase3_corrected_threefold_seed0_combined_v2.json`
- `phase3_corrected_threefold_seed0_gate_decision_v2.json`
- `bootstrap_g1_vs_b1.json`
- `bootstrap_g1_vs_train_shuffled.json`

각 run directory에는 config, code snapshot, checkpoint, runtime manifest, per-sample gzip
prediction이 있다. Legacy 36-run 결과는 변경하지 않았다.
