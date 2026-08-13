# Phase 3B corrected architecture gate v2

날짜: 2026-08-12  
상태: task-1/seed-0 smoke, three-fold/seed-0 gate, prediction analysis,
weak-label audit sampling 완료. 사전 등록 gate에 따라 three-seed GNN 확대는 중단했고
manual weak-label review는 대기 중이다.

## Corrected version이 필요한 이유

Legacy reduced cross-fold 결과는 당시 기록한 evaluation contract 안에서만 유효하다.
덮어쓰거나 현재 protocol 결과처럼 재해석하면 안 된다. 다음 confound를 확인했다.

1. Legacy G1 current-relation auxiliary head가 action-blended edge representation을 받아
   action shuffle이 current와 future objective 모두를 훼손할 수 있었다.
2. 의도한 natural protocol과 달리 checkpoint와 threshold를 event-enriched validation에서
   선택했다.
3. Reported event metric이 ground-truth current holding을 사용해 conditional/oracle-current
   future-change metric이었다.
4. Manifest QA가 task-local category quota나 natural/stress overlap sample payload hash를
   검사하지 않았다.

## Corrected 계약

- 모든 비교 모델의 current auxiliary head는 같은 action-free pair encoder를 사용한다.
- Checkpoint는 natural-validation conditional holding-event PR-AUC로 선택한다.
- Future/current threshold는 natural validation에서 한 번 선택해 natural test, stress,
  모든 perturbation control에 고정한다.
- Natural-test conditional event PR-AUC가 primary다. Thresholded F1, onset/release F1,
  hard-negative FPR, Brier, 10-bin ECE는 secondary다.
- Conditional/oracle-current 이름을 명시하고 predicted current/future holding을 사용하는
  end-to-end event metric도 저장한다.
- Validation/natural/stress output에 pair probability, target, prediction, task, episode,
  sample ID, timestep, edge identity, event cluster metadata를 gzip JSONL로 저장한다.
- Stress view는 natural held-out episode의 future-event-enriched subset이며 독립 test가 아니다.
- Manifest QA는 sample/episode leakage, duplicate ID, task-local quota, subset membership,
  overlap graph/action/label payload SHA256를 검사한다.
- Train-shuffled G1은 task-local, episode-disjoint donor를 action magnitude와 coarse state로
  matching하고 donor QA와 distance를 저장한다.

## Smoke와 확대 gate

첫 run은 held-out task 1, seed 0에서 B1-v2, G1 late-action, S-0-G1,
matched train-shuffled G1을 비교한다. Current head를 포함한 parameter count로 width를
near-matched하게 선택한다.

바로 3×3으로 확대하지 않는다. 먼저 three folds × one seed를 실행한다. G1이 B1-v2보다
natural PR-AUC에서 최소 2개 task 우세하고 hard-negative가 심하게 악화되지 않으며
release가 개선되고 action shuffle에서 일관되게 하락할 때만 확대한다. 실패하면 GNN
depth 개발을 중단하고 limited set context의 pair-local temporal encoder로 전환한다.

## Corrected gate 결과

Task-1/seed-0 smoke 후 누락된 task 0/2 seed-0만 실행했다. Task 1은 반복하지 않았다.
12 model/fold runs가 같은 manifest와 protocol을 사용했다.

- G1−B1 natural conditional PR-AUC: task 0 −0.1161, task 1 +0.1322,
  task 2 +0.1717.
- Task-macro +0.0626, hierarchical bootstrap 95% CI [−0.0892, +0.1905].
- Thresholded event F1: G1 0.3272, B1 0.4083.
- Release F1: G1−B1 +0.1626 [0.0701, 0.2967].
- Hard-negative FPR: G1−B1 −0.3326 [−0.4201, −0.2429].
- G1−train-shuffled G1 PR-AUC: −0.0567 [−0.1943, +0.0572].
- Train-shuffled G1이 task-macro conditional PR-AUC 0.4985와 event F1 0.4333으로 가장 높았다.

필수 action-alignment criterion이 실패했다. 결정은
`stop_gnn_three_seed_expansion_and_pivot_pair_local_temporal_encoder`다. 이는 graph
structure가 일반적으로 쓸모없다는 결론이 아니라 현재 late/global action G1이
architecture/action gate를 통과하지 못했다는 결론이다.

## 통계와 weak-label audit

Task를 outer unit으로 두고 task 내 seed 평균 후 task fold, episode, event cluster를
hierarchical bootstrap한다. One-fold smoke CI는 task generalization CI가 아니라 diagnostic다.
Historical artifact에 event ID가 없으면 sample ID를 보수적인 proxy로 썼음을 기록한다.

Deterministic audit builder가 task 0/1/2 × onset/release/hard-negative × 10개, 총 90개를
선택했다. Trajectory-enriched v2에서 seven-frame trajectory 592/592를 확보했고 conflict는
없었다. Viewer는 human decision을 원자적으로 기록한다. Review는 아직 0/90이므로 pass
rate를 주장하지 않는다.

## Versioned artifact

Local entry point:

- `configs/phase3_holder_action_eval_v2_corrected.json`
- `configs/phase3_corrected_smoke_v2.json`
- `scripts/phase3/run_corrected_architecture_gate.py`
- `scripts/phase3/analyze_corrected_predictions.py`
- `scripts/phase3/build_weak_label_audit.py`
- `scripts/phase3/weak_label_audit_viewer.py`

Persistent root:

`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/corrected_protocol_v2`

Legacy `phase3_reduced_crossfold_gate_v1.json`, checkpoint 36개, code snapshot은 변경하지 않았다.
