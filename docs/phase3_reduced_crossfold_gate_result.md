# Phase 3B-R1 reduced cross-fold gate 결과

날짜: 2026-08-12  
범위: 고정 Phase 3B-R1 manifest, held-out `test_task0/1/2`, seeds 0/1/2  
Primary: holding change-event F1, event PR-AUC, hard-negative FPR

Holding stress row는 natural held-out episode에서 future event를 사용해 선택한 subset이다.
독립 test가 아니며 두 번째 generalization 결과로 해석하지 않는다. Legacy event metric은
ground-truth current holding 조건이고 corrected v2는 이를 oracle-current로 명시하며
predicted-current end-to-end event metric을 추가했다.

## 재현성

Root: `/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1`

- Manifest: `phase3B_R1_eval_manifest.json`
- Result: `phase3_reduced_crossfold_gate_v1.json`
- Checkpoint: `checkpoints_reduced_crossfold_v1` 36개.
- Snapshot: `code_snapshot_reduced_crossfold_v1`.
- Completed: 36/36 runs.
- Natural/stress를 별도로 평가하고 validation-fitted threshold를 고정했다.
- Paired run은 같은 sample ID를 사용했다.

## 모델

| ID | 조건 | Hidden | Parameters | G1과 차이 |
|---|---|---:|---:|---:|
| B1-v2 | target-pair feature MLP | 55 | 45,668 | +722 |
| G1-correct | sparse holder–object GNN + aligned action | 48 | 44,946 | 0 |
| S-0-G1 | action 없는 sparse GNN | 54 | 44,784 | −162 |
| G1-train-shuffled | batch-shuffled action으로 학습한 G1 | 48 | 44,946 | 0 |

B1-v2와 S-0는 exact가 아니라 near-parameter-matched다. 차이는 model size에 비해 작지만
gate의 한계로 명시한다.

## 9-run 평균

| Model | Natural F1 | Stress F1 | Natural PR-AUC | Stress PR-AUC | Natural hard-neg FPR | Stress hard-neg FPR |
|---|---:|---:|---:|---:|---:|---:|
| B1-v2 | 0.3285 | **0.7670** | **0.4047** | 0.4898 | **0.1075** | **0.1017** |
| G1-correct | 0.3400 | 0.7465 | 0.3872 | 0.4972 | 0.2146 | 0.1884 |
| S-0-G1 | **0.3604** | 0.5927 | 0.3691 | **0.5159** | 0.2836 | 0.2613 |
| G1-train-shuffled | 0.3470 | 0.4746 | 0.2793 | 0.4093 | 0.3277 | 0.2931 |

모든 metric을 지배하는 모델은 없다. B1-v2는 stress thresholded F1과 hard-negative FPR,
S-0는 natural F1과 stress PR-AUC가 가장 좋다. G1은 competitive하지만 B1보다 일관된
graph advantage를 입증하지 못했다.

Same-fold/same-seed `G1−control` event F1:

| 비교 | Natural mean (양수 pair) | Stress mean (양수 pair) |
|---|---:|---:|
| G1−B1-v2 | +0.0115 (2/9) | −0.0206 (2/9) |
| G1−S-0-G1 | −0.0204 (4/9) | +0.1538 (8/9) |
| G1−G1-train-shuffled | −0.0070 (4/9) | +0.2719 (9/9) |

Graph가 target-pair MLP를 일관되게 이기지 못했다. Action-enabled G1은 stress에서
action-free/train-shuffled보다 강하지만 natural에서는 이점이 없거나 약간 음수다.
Uniform improvement가 아니라 split-dependent benefit이다.

## Action control 해석

Corrected global episode-disjoint action shuffle에서 G1 event PR-AUC는 natural 9/9 runs에서
평균 0.1303 하락했고 stress 7/9에서 평균 0.0341 하락했다. S-0는 예상대로 action
invariant였다. Train-shuffled control의 stress F1은 aligned G1보다 9/9에서 평균 0.2719
낮았지만 natural 차이는 −0.0070이고 4/9만 aligned가 높았다. 가장 강한 action-alignment
근거는 stress에서 나왔으며 natural에서 일관되지 않았다.

## 결정

1. Sparse holder–object topology를 현재 preferred graph topology로 유지한다.
2. GNN message passing이 target-pair MLP보다 일관되게 우세하다고 주장하지 않는다.
3. Action pathway는 stress-oriented hypothesis로 유지하되 natural generality를 주장하지 않는다.
4. Phase 4를 계속 차단한다.
5. Final capacity claim에서는 exact matching을 사용하거나 near-matched 한계를 명시한다.

다음은 calibration/hard-negative control을 보완한 corrected protocol이다. Complete C-L/C-E와
추가 FiLM complexity는 이전 topology follow-up에 따라 확대하지 않는다.
