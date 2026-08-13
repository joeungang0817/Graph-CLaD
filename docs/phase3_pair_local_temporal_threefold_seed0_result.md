# Phase 3B pair-local temporal three-fold seed-0 결과

날짜: 2026-08-13  
Protocol: `phase3-pair-local-temporal-threefold-seed0-v1`  
Runs: 12/12 (3 task folds × 1 seed × H0–H3)  
SHA256: `492d45521e6ccecbc4f0d89923f50d49642962d60c1c53daf093b5aec9b4d188`

## 범위

One-seed architecture screen이다. Outer evaluation unit은 12 model이 아니라 task fold다.
Natural test가 primary이고 stress는 future-event-selected subset이다. Event PR-AUC/F1은
oracle current holding 조건이며 end-to-end metric과 per-sample prediction도 artifact에
저장했다.

H0–H3는 pair-independent encoding, t 이하 causal history, 공통 action-free current head,
natural-validation PR-AUC checkpoint, frozen threshold를 사용한다.

## Task별 natural 결과

| Task | Model | PR-AUC | F1 | Release F1 | Hard-neg FPR | Threshold |
|---|---|---:|---:|---:|---:|---:|
| 0 | H0 state | 0.2373 | 0.2881 | 0.2016 | 0.1429 | 0.94 |
| 0 | H1 history | 0.3998 | 0.3846 | 0.0645 | 0.4286 | 0.93 |
| 0 | H2 action | 0.1826 | 0.2934 | 0.2286 | 0.4381 | 0.95 |
| 0 | H3 history+action | **0.4009** | **0.4074** | **0.2353** | 0.4762 | 0.87 |
| 1 | H0 state | 0.3810 | 0.3671 | 0.0000 | 0.5000 | 0.95 |
| 1 | H1 history | 0.4960 | **0.5352** | 0.0000 | 0.3000 | 0.95 |
| 1 | H2 action | **0.5869** | 0.4184 | **0.2143** | **0.0900** | 0.90 |
| 1 | H3 history+action | 0.5731 | 0.3357 | 0.1075 | 0.1100 | 0.85 |
| 2 | H0 state | 0.4695 | 0.2849 | **0.0777** | 0.3077 | 0.94 |
| 2 | H1 history | 0.4086 | **0.5563** | 0.0000 | 0.2788 | 0.95 |
| 2 | H2 action | 0.4129 | 0.2440 | 0.0759 | 0.1442 | 0.95 |
| 2 | H3 history+action | **0.4734** | 0.3419 | 0.0150 | **0.1154** | 0.94 |

## Task-macro mean

| Model | PR-AUC | F1 | Release F1 | Hard-neg FPR |
|---|---:|---:|---:|---:|
| H0 | 0.3626 | 0.3134 | 0.0931 | 0.3168 |
| H1 | 0.4348 | **0.4920** | 0.0215 | 0.3358 |
| H2 | 0.3941 | 0.3186 | **0.1729** | **0.2241** |
| H3 | **0.4824** | 0.3617 | 0.1193 | 0.2339 |

| Contrast | PR-AUC | Positive tasks | F1 | Release F1 | Hard-neg FPR |
|---|---:|---:|---:|---:|---:|
| H1−H0 | +0.0722 | 2/3 | +0.1786 | −0.0716 | +0.0190 |
| H2−H0 | +0.0315 | 1/3 | +0.0052 | +0.0798 | −0.0927 |
| H3−H1 | +0.0476 | **3/3** | −0.1304 | **+0.0978** | **−0.1019** |
| H3−H2 | +0.0883 | 2/3 | +0.0430 | −0.0536 | +0.0097 |
| H3−H0 | +0.1198 | **3/3** | +0.0483 | +0.0262 | −0.0830 |

H3−H1 release는 3 task 모두 개선됐다. Hard-negative는 task 1/2에서 좋아졌지만 task
0에서 0.0476 악화됐다. H3−H0는 macro가 좋아도 task 0 FPR이 크게 악화됐다.
Threshold 0.85–0.95가 많아 F1은 secondary다.

## Gate 결정

H3만 candidate로 유지한다. Action을 causal history model에 추가했을 때 primary PR-AUC와
release가 3 task 모두 개선되고 mean hard-negative도 좋아졌다. 그러나 바로 full
three-seed factorial을 실행하지 않는다. Episode-disjoint matched donor의 three-fold/seed-0
`H3-train-shuffled` control을 먼저 수행한다. Aligned H3가 natural PR-AUC에서 최소 2개
task 우세하지 않거나 release/hard-negative가 위험하면 확대를 중단한다. 통과 시 H3,
H1, H3-train-shuffled에만 seeds 1/2를 추가한다.

Persistent root:
`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/corrected_protocol_v2/pair_local_temporal_threefold_seed0_v1`
