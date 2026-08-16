# Phase 3B pair-local temporal encoder 계획

날짜: 2026-08-12  
상태: three-fold/seed-0 H0–H3 screen과 H3 action-alignment control 완료; gate 실패.

## 연구 동기

Corrected three-fold/seed-0 screen에서 late-action G1은 pair MLP나 train-shuffled
control보다 안정적으로 우세하지 않았다. 다만 release detection과 hard-negative
rejection은 개선했다. 다음 실험은 holding transition에 complete scene message passing보다
causal pair-local temporal state가 더 중요한지 묻는다.

## 고정 2×2 factorial

| Model | Past history | Action |
|---|---:|---:|
| H0 | 없음 | 없음 |
| H1 | 있음 | 없음 |
| H2 | 없음 | 있음 |
| H3 | 있음 | 있음 |

네 모델 모두 같은 robot–object pair encoder, set-context capacity, loss,
natural-validation checkpoint rule, frozen threshold protocol, sample, split, epoch budget,
patience를 사용한다. Capacity 차이는 기록하고 width selection 또는 explicit adapter로
near-matched하게 유지한다.

## Causal feature 계약

모든 temporal feature는 현재 시점 t 이하 frame만 사용해 계산한다. Future frame, future
event identity, future holding heuristic을 encoder가 읽으면 안 된다.

- Gripper–object relative-position delta.
- Relative velocity.
- Contact persistence.
- Gripper closure velocity.
- Object-following stability.
- 각 feature family의 validity mask.

History window, timestamp gap, missing-frame 처리, normalization source, mask를
manifest/runtime artifact에 기록한다. Weak holding label에도 쓰이는 feature는 t에서
strictly causal하게 계산될 때만 input으로 허용한다.

## Architecture 경계

각 target robot–object pair를 먼저 독립적으로 encode한다. Scene context가 필요하면 pair
token에 대한 permutation-invariant DeepSets summary 또는 pair-set attention으로 제한한다.
이 gate에는 unrestricted object–object message passing을 도입하지 않는다. Action은 최종
global head에서만 concatenate하지 않고 pair temporal encoder 또는 transition block에서
조건으로 사용한다.

## 평가와 확대 기준

같은 task-1/seed-0 corrected smoke로 시작하고 technical/leakage QA를 통과한 architecture만
three folds × one seed를 실행한다. Natural-test conditional event PR-AUC가 primary다.
End-to-end event PR-AUC/F1, onset/release F1, hard-negative FPR, Brier/ECE,
action/history ablation, per-sample prediction을 반드시 저장한다.

History 또는 action gain이 최소 2개 task에서 일관되고 hard-negative가 실질적으로
악화되지 않으며 extreme threshold에만 의존하지 않고 release가 개선될 때만 three seeds로
확대한다. Task를 outer unit으로 한 hierarchical bootstrap을 사용한다. 작은 H0–H3 gain을
label-valid 결과로 해석하기 전에 90-item weak-label manual review를 완료해야 한다.

Trajectory-enriched audit viewer는 선택한 90개 item의 deduplicated graph frame 592/592를
확보했다. Model 구현은 병행할 수 있지만 human decision 완료 전에는 작은 gain을 label
validity 근거로 해석하지 않는다.

## Smoke 결과 — 2026-08-12

Task 1, seed 0의 natural conditional/oracle-current event PR-AUC는 H0 0.3810,
H1 0.4960, H2 0.5869, H3 0.5731이었다. Factorial contrast는 H1−H0 +0.1149,
H2−H0 +0.2058, H3−H1 +0.0771, H3−H2 −0.0138이다.

H2가 natural release F1 0.2143과 hard-negative FPR 0.0900으로 가장 좋았다. H1은
calibration과 onset을 개선했으나 release hit가 없었다. H3는 natural end-to-end event
PR-AUC 0.4227로 가장 높았지만 conditional PR-AUC와 release F1은 H2보다 낮았다.
이 smoke에서는 action이 가장 강한 단일 factor였고 history는 action이 없을 때 유용했다.
Primary conditional metric은 action 위에 history를 추가할 근거를 주지 못했다.

이는 one task/one seed 구현 gate이지 multi-task conclusion이 아니다.

## Three-fold seed-0 결과 — 2026-08-13

12-run screen의 task-macro natural conditional event PR-AUC는 H0 0.3626, H1 0.4348,
H2 0.3941, H3 0.4824였다. H3−H1은 +0.0476이고 3개 task 모두 양수였다. Release
F1도 모든 task에서 개선되고 mean hard-negative FPR은 0.1019 감소했다. H3−H0 PR-AUC는
+0.1198로 모든 task에서 양수였으나 task 0 hard-negative FPR은 크게 악화됐다.
H3−H2 history gain은 2개 task에서만 양수였고 mean release F1을 낮췄다.

H3만 candidate로 유지한다. Seed 추가 전 episode-disjoint matched donor를 사용하는
three-fold/seed-0 H3 train-action-shuffled control을 실행한다. Aligned H3가 natural PR-AUC
에서 최소 2개 task 우세하고 release/hard-negative가 안전할 때만 H3, H1,
H3-train-shuffled의 seeds 1/2를 실행한다. 상세 값은
`docs/phase3_pair_local_temporal_threefold_seed0_result.md`에 있다.

## Action-alignment 결과 — 2026-08-16

KCloudVPN에서 H3 train-shuffled 3/3 runs를 완료했다. Aligned H3의 natural PR-AUC
우세는 1/3 tasks였고 paired hierarchical bootstrap은 PR-AUC +0.0085
95% CI [−0.0506,+0.0772], event F1 −0.2264 [−0.3955,−0.0459], release F1 −0.3346
[−0.5339,−0.1016], hard-negative FPR +0.0057 [−0.0796,+0.1007]이었다.

사전 기준에 따라 gate는 실패다. H3의 causal action 주장을 중단하고 seeds 1/2 확대를
하지 않는다. Phase 3C 후보는 H1 또는 다른 action-free pair-local encoder로 좁힌다.
상세 값은 `docs/phase3_pair_local_temporal_action_alignment_seed0_result.md`에 있다.
