# Phase 3B pair-local temporal encoder smoke 결과

날짜: 2026-08-12  
Protocol: `phase3-pair-local-temporal-smoke-v1`  
Fold/seed: `test_task1`, seed 0  
Result: `phase3_pair_local_temporal_smoke_task1_seed0_v1.json`

## 범위

Corrected architecture/probe smoke이며 generalization 결과가 아니다. 모든 모델은 같은
sample ID, natural-validation checkpoint, frozen threshold, action-free current head,
causal history 계약을 사용한다. H0–H3는 robot–object pair를 독립 처리하고 unrestricted
object–object message passing을 쓰지 않는다. History는 t 이하 frame만 읽는다.
Conditional event metric은 oracle current holding을 사용하고 end-to-end metric은 별도로
보고한다.

## Natural test

| Model | History | Action | PR-AUC | F1 | End-to-end PR-AUC | Onset F1 | Release F1 | Hard-neg FPR | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 없음 | 없음 | 0.3810 | 0.3671 | 0.1939 | 0.4270 | 0.0000 | 0.5000 | 59,688 |
| H1 | 있음 | 없음 | 0.4960 | 0.5352 | 0.2207 | 0.6726 | 0.0000 | 0.3000 | 60,728 |
| H2 | 없음 | 있음 | **0.5869** | 0.4184 | 0.2439 | **0.8837** | **0.2143** | **0.0900** | 59,708 |
| H3 | 있음 | 있음 | 0.5731 | 0.3357 | **0.4227** | 0.7872 | 0.1075 | 0.1100 | 60,476 |

Natural validation threshold는 H0–H3 순서로 0.95, 0.95, 0.90, 0.85였다. One task/seed
smoke이며 validation-fitted threshold이므로 F1은 secondary다.

| Contrast | PR-AUC | Release F1 | Hard-neg FPR |
|---|---:|---:|---:|
| H1−H0 | +0.1149 | +0.0000 | −0.2000 |
| H2−H0 | **+0.2058** | **+0.2143** | **−0.4100** |
| H3−H1 | +0.0771 | +0.1075 | −0.1900 |
| H3−H2 | −0.0138 | −0.1068 | +0.0200 |
| H3−H0 | +0.1921 | +0.1075 | −0.3900 |

## 해석

이 smoke에서는 action이 conditional ranking, release, hard-negative에서 가장 강한 단일
factor였다. Causal history는 action이 없을 때 PR-AUC/onset에 도움을 줬지만 action 위에
추가한 increment는 primary metric에서 음수였다. H3 end-to-end PR-AUC가 높은 것은
current state도 추론할 때 history가 도움될 수 있음을 시사하지만 release는 약했다.

Graph-transition advantage, task 간 causal action 효과, B1/G1 대비 우월성을 입증하지
않는다. Stress view는 독립 test가 아니다.

Persistent root:
`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/corrected_protocol_v2/pair_local_temporal_smoke_task1_seed0_v1`

SHA256: `ae986247bc1d536269e94964d32a46e0c3167c1bd40b3015b623f2ced6148c502`

다음은 H2/H3 candidate와 H0/H1 ablation의 three-fold/one-seed screen이다. 최소 2개
task consistency, hard-negative 안전성, release 개선, weak-label review를 확인한 뒤에만
three seeds로 확대한다.
