# Phase 3B H3 action-alignment three-fold seed-0 결과

날짜: 2026-08-16  
Protocol: `phase3-pair-local-temporal-action-alignment-seed0-v1`  
Runs: 3/3 (`H3-train-shuffled` × 3 task folds × seed 0)  
Comparison: `H3-history-action_minus_H3-train-shuffled`

## 목적과 범위

Pair-local H0–H3 screen에서 관찰된 H3의 action gain이 올바른 action–state 의미 정렬에
의존하는지 검사했다. Aligned H3는 기존 three-fold seed-0 artifact를 사용하고,
train-shuffled H3는 task-local episode-disjoint matched donor action으로 새로 학습했다.
두 조건은 같은 fold, sample, split, model capacity, training budget과 validation-threshold
protocol을 사용했다.

이 결과는 세 held-out task와 seed 0 하나의 architecture/control gate다. 여러 seed의
일반화 결론이 아니다.

## Artifact

Shuffled output root:

`/home/ubuntu/graphclad-artifacts/phase3_holder_action_v1/corrected_protocol_v2/kcloudvpn_pair_local_temporal_action_alignment_seed0_v1`

Paired hierarchical bootstrap:

`/home/ubuntu/graphclad-artifacts/phase3_holder_action_v1/corrected_protocol_v2/kcloudvpn_pair_local_temporal_action_alignment_seed0_v1/aligned_vs_shuffled_bootstrap_v1.json`

분석은 task fold → episode → event cluster 계층으로 2,000회 재표집했으며 bootstrap seed는
`20260816`이다.

## Task별 natural 결과

Conditional/oracle-current holding event 결과다. Aligned 값은 기존 H0–H3 result의 표시
정밀도, shuffled 값은 KCloudVPN runner stdout을 따른다.

| Task | Aligned PR-AUC | Shuffled PR-AUC | Aligned−Shuffled | Aligned F1 | Shuffled F1 | Aligned release F1 | Shuffled release F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4009 | 0.4245 | −0.0236 | 0.4074 | 0.4524 | 0.2353 | 0.4000 |
| 1 | 0.5731 | 0.5059 | +0.0672 | 0.3357 | 0.7086 | 0.1075 | 0.5882 |
| 2 | 0.4734 | 0.4913 | −0.0179 | 0.3419 | 0.6034 | 0.0150 | 0.3733 |
| Task macro | 0.4824 | 0.4739 | 약 +0.0085 | 0.3617 | 0.5881 | 0.1193 | 0.4539 |

Aligned H3가 PR-AUC에서 우세한 task는 task 1의 1/3뿐이다. 사전 통과 기준인 최소
2/3-task 우세에 미달한다.

## Paired hierarchical bootstrap

모든 차이는 aligned minus train-shuffled다. Hard-negative FPR은 낮을수록 좋으므로 양의
차이는 aligned가 더 나쁜 방향이다.

| Metric | Estimate | 95% CI | 판정 |
|---|---:|---:|---|
| Event PR-AUC | +0.0085 | [−0.0506, +0.0772] | 0 포함; 정렬 이점 근거 없음 |
| Event F1 | −0.2264 | [−0.3955, −0.0459] | aligned가 낮음 |
| Release F1 | −0.3346 | [−0.5339, −0.1016] | aligned가 낮음 |
| Hard-negative FPR | +0.0057 | [−0.0796, +0.1007] | 0 포함; 뚜렷한 차이 없음 |

PR-AUC와 hard-negative FPR의 interval은 0을 포함한다. Event F1과 release F1 interval은
전부 음수이며, 각 모델/fold의 natural-validation threshold를 적용한 thresholded 성능에서
aligned H3가 더 낮다.

## Gate 판정

Phase 3B H3 action-alignment gate는 **실패**다.

- Primary 기준인 aligned PR-AUC task별 우세가 1/3이다.
- PR-AUC paired estimate는 작고 95% interval이 0을 포함한다.
- Release F1은 aligned가 3/3 tasks에서 낮고 paired interval도 0 아래다.
- Hard-negative FPR의 뚜렷한 개선은 없다.

따라서 H3의 성능을 올바르게 정렬된 action의 causal 또는 semantic 효과로 해석하지
않는다. H3/H1/H3-train-shuffled seeds 1/2 확대는 중단한다. H3는 one-seed
architecture-screen finding으로 보존하고, Phase 3C에서는 H1 또는 다른 action-free
pair-local representation을 우선 검토한다.

## Claim limit과 후속 작업

- Natural held-out test가 primary다. Challenge는 독립 generalization test가 아니다.
- Event metric은 oracle current holding에 조건부이므로 end-to-end metric을 별도로 본다.
- 세 task와 seed 0 하나의 결과이므로 작은 차이에 일반화 주장을 붙이지 않는다.
- Human 90-item weak-label audit은 아직 완료되지 않았다.
- 다음 실행은 대규모 seed 확대가 아니라 weak-label audit과 Phase 3C action-free
  technical smoke다.
