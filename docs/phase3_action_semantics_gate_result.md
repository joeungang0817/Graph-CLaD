# Phase 3B action-semantics gate 결과

날짜: 2026-08-12  
범위: held-out `test_task0`, seeds 0/1/2  
Primary: actual holding change-event F1

## 재현성

아래 경로는 모두
`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1` 아래에 있다.

- Frozen manifest: `phase3B_R1_eval_manifest.json`
- 4-condition/12-run report: `phase3_action_semantics_gate_task0_3seed_v1.json`
- Corrected global-shuffle report: `phase3_global_action_shuffle_control_task0_3seed_v1.json`
- Decision artifact: `phase3_action_semantics_gate_task0_3seed_v2_analysis.json`
- G1 checkpoint: `checkpoints_global_action_shuffle_v1`
- Code snapshot: `code_snapshot_action_gate_v1`
- Colab QA: Python compile 후 25 tests 통과.

Natural/stress sample ID, train-only normalization, validation threshold, sparse topology,
loss, optimizer, checkpoint rule을 조건 간 공유했다. Run마다 checkpoint와 partial JSON을
Drive에 저장했다.

## 비교 control

| ID | 조건 | Hidden | Parameters | 분리한 질문 |
|---|---|---:|---:|---|
| G1-correct | aligned sample action | 48 | 44,946 | 현재 full candidate |
| S-0-G1 | action 없는 동일 G1-style block | 54 | 44,784 | action pathway가 필요한가? |
| G1-constant | action branch에 fixed template 입력 | 48 | 44,946 | branch capacity만으로 충분한가? |
| G1-train-shuffled | batch 내 mismatched action으로 학습 | 48 | 44,946 | training의 scene-action alignment가 필요한가? |

`G1-constant`는 G1과 architecture/parameter count가 같지만 sample-specific action을
제거한다. `G1-train-shuffled`는 batch action marginal을 보존하면서 scene-action pair를
cyclic shift한다.

## Three-seed 평균

| Model | Natural event F1 | Stress event F1 | Natural PR-AUC | Stress PR-AUC | Natural hard-neg FPR | Stress hard-neg FPR |
|---|---:|---:|---:|---:|---:|---:|
| G1-correct | **0.3965** | **0.7162** | 0.3720 | 0.5144 | 0.2476 | 0.1858 |
| S-0-G1 | 0.3192 | 0.6565 | **0.3761** | **0.5805** | **0.1714** | **0.1585** |
| G1-constant | 0.3452 | 0.6261 | 0.3413 | 0.5444 | 0.2603 | 0.1967 |
| G1-train-shuffled | 0.3207 | 0.5164 | 0.2767 | 0.4959 | 0.2762 | 0.1967 |

G1은 thresholded event F1이 가장 높다. S-0는 PR-AUC와 hard-negative FPR이 더 좋아
action pathway가 모든 holding metric에서 일관되게 우세하지는 않다.

`G1-correct − control` event F1:

| Control | Natural mean | Natural seed 차이 | Stress mean | Stress seed 차이 |
|---|---:|---|---:|---|
| S-0-G1 | +0.0773 | +0.1054, −0.0027, +0.1291 | +0.0597 | +0.1333, −0.1436, +0.1895 |
| G1-constant | +0.0513 | +0.1120, +0.0629, −0.0209 | +0.0901 | +0.1962, +0.0434, +0.0308 |
| G1-train-shuffled | **+0.0758** | +0.0474, +0.0596, +0.1205 | **+0.1998** | +0.2717, +0.0385, +0.2894 |

Aligned-action training은 두 view 모두 3/3 seed에서 train-shuffled보다 높았다. G1은
exact parameter-matched constant branch보다 stress 3/3, natural 2/3에서 높아 capacity만으로
전체 차이를 설명하기 어렵다. S-0 비교는 seed 1에서 뒤집혀 덜 안정적이다.

## Legacy test shuffle 교정

Legacy `shuffled_action`은 evaluation batch 안에서 한 row만 roll했다. Row가 episode-local로
정렬돼 donor 대부분이 같은 demonstration의 인접 action이었다.

| Split | Legacy action L2 | Legacy same-episode | Global action L2 | Global same-episode |
|---|---:|---:|---:|---:|
| Natural | 0.6835 | 98.35% | 4.0506 | 0% |
| Stress | 0.9284 | 95.81% | 2.6823 | 0% |

Corrected `global_shuffled_action`은 split 전체의 deterministic bijection이며 action
distribution을 보존하고 다른 episode donor만 사용한다. Legacy 결과는 호환성 때문에
남기지만 strong semantic control로 해석하지 않는다.

Correct action − global episode-disjoint shuffle:

| Metric | Natural mean | Natural support | Stress mean | Stress support |
|---|---:|---:|---:|---:|
| Event F1 | +0.0729 | 2/3 | +0.1206 | 3/3 |
| Event PR-AUC | **+0.0718** | **3/3** | **+0.0831** | **3/3** |
| Onset F1 | +0.1590 | 2/3 | +0.0724 | 2/3 |
| Release F1 | −0.0053 | 1/3 | +0.1713 | 3/3 |

Threshold-free PR-AUC가 두 view 모든 seed에서 낮아진 점이 가장 강한 action-semantic
근거였다. No-action과 edge-shuffle control도 useful pathway와 일치했지만 natural release와
hard-negative 안정성은 충분하지 않았다.

## 결정과 주장 한계

- Sparse holder–object topology를 유지한다.
- Complete topology와 추가 FiLM complexity는 확대하지 않는다.
- Task 0의 aligned action training 근거는 유지한다.
- Sample-specific inference action 사용은 provisional evidence로만 본다.
- Task-family generalization은 주장하지 않는다. Three-seed 근거는 task 0뿐이다.
- Phase 4는 차단한다.

다음은 B1-v2, G1, S-0, G1-train-shuffled의 near-parameter-matched 3 folds × 3 seeds다.
Global episode-disjoint action control, edge shuffle, natural/stress 분리, frozen threshold,
per-run checkpoint를 유지한다.
