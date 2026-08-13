# Phase 3B holder–object feature/action v2 smoke 결과

날짜: 2026-08-11  
범위: `test_task0`, seed 0, width unmatched smoke  
Primary: actual holding change-event F1. Validation correct-action output에서 threshold를
선택해 natural/stress와 모든 control에 고정했다.

## 재현성

- Manifest: `phase3_holder_action_v1/phase3B_R1_eval_manifest.json`
- Result: `phase3_holder_action_v1/smoke_test_task0_seed0_feature_v2.json`
- Analysis: `smoke_test_task0_seed0_feature_v2_analysis.json`
- Snapshot: `code_snapshot_v2`
- QA: A100, fixed manifest, feature/model/metric regression 16 tests 통과.
- Split: train 1,200; validation 474; natural 486; stress 167.

## 결과

| Model | Parameters | Threshold | Natural F1 | Stress F1 | Natural AP | Stress AP |
|---|---:|---:|---:|---:|---:|---:|
| P0 flat MLP | 20,514 | 0.86 | 0.3662 | 0.7400 | 0.3783 | 0.4984 |
| B1-v2 pair-feature MLP | 35,490 | 0.59 | 0.4022 | 0.7872 | 0.4327 | **0.7153** |
| G1 sparse holder-object GNN | 44,946 | 0.59 | **0.4257** | **0.8000** | **0.4426** | 0.5767 |
| Legacy G3 gated action-edge | 54,787 | 0.94 | 0.3980 | 0.7959 | 0.2539 | 0.5226 |
| S-LF-v2 flat late-action | 54,930 | 0.95 | 0.3544 | 0.6957 | 0.4210 | 0.5949 |
| S-LS-v2 structured late | 60,235 | 0.65 | 0.3015 | 0.6452 | 0.3703 | 0.4941 |
| S-EF-v2 structured FiLM edge | 74,300 | 0.91 | 0.3321 | 0.7500 | 0.3498 | 0.5960 |

G1은 두 view event F1 1위였지만 B1-v2 대비 +0.0236/+0.0128로 작았다. B1-v2는 stress
AP가 가장 높고 hard-negative FPR이 natural 0.0381, stress 0.0000으로 가장 낮았다.
G1 FPR은 0.1048/0.0984였다. Sparse message passing의 thresholded F1 신호는 약하게
있지만 결정적인 graph advantage는 아니다.

Action-shuffle delta는 작고 view 간 불일치하거나 음수였다. Gripper-only shuffle도 거의
0이었다. Structured action encoding과 FiLM routing은 action-use gate를 통과하지 못했다.
Edge shuffle은 특히 stress에서 더 큰 drop을 만들었지만 B1-v2 edge-feature control도
하락했으므로 GNN message passing 고유 근거가 아니다.

## 결정

One fold/seed이고 parameter matching이 꺼져 있어 바로 3×3으로 확대하지 않는다. 이후
S-0와 complete-topology follow-up을 수행했으며 authoritative 결과는
`docs/phase3_topology_action_followup_result.md`다.
