# Graph-CLaD 연구 기록

이 문서는 연구 진행, 실험 결과, gate 판단을 날짜순으로 기록한다. 모델명, metric 이름,
config ID, 경로, SHA256은 재현성을 위해 영문 표기를 유지한다. 2026-08-13 이전 영문
원문은 `archive/pre_korean_translation_20260813.zip`에 보존했다.

## 연구 질문

CLaD의 두 시점 semantic transition을 object–relation graph transition으로 구조화했을 때
robot–object interaction과 object spatial transition이 더 명시적으로 보존되는지 검증한다.
Holding onset/release는 최종 목적이 아니라 representation과 architecture를 검사하는
중간 probe다. Loss만으로 가설을 판단하지 않고 event PR-AUC/F1, release/onset,
hard-negative FPR, calibration, action/edge control, non-graph baseline을 함께 본다.

## 2026-08-05 — Phase 0: 제공 CLaD baseline 실행 경계

- `baseline_code/`를 수정하지 않고 synthetic train/eval/EMA smoke 경로를 만들었다.
- Training forward의 loss 네 개, finite gradient, evaluation embedding shape `[B,H]`,
  EMA copy/update 계약을 검사하도록 했다.
- 실제 VLM preprocessing, action packing, Stage 2 diffusion policy는 확인되지 않았으므로
  `docs/unknowns.md`에 남겼다.
- 로컬 bundled Python에는 PyTorch가 없어 model smoke 자체는 project runtime에서만
  실행할 수 있었다. Phase 0 code/config 계약은 완료했다.

## 2026-08-05 — Phase 1A: LIBERO state와 API 조사

- LIBERO task 0의 live environment에서 observation, `env.sim`, robot/object pose,
  logical object identity, predicate interface를 조사했다.
- `logical_id`를 episode 간 graph identity로 사용하고 MuJoCo `body_id`는 runtime audit
  metadata로만 쓰기로 고정했다.
- Live capture는 snapshot 2개, snapshot당 object/fixture/site entry 23개,
  92-dimensional simulator state를 포함했다.
- 일부 `get_joint_state`, `is_open`, `is_close`가 object type에 따라 오류를 내므로
  unavailable predicate를 false로 채우지 않고 `valid=0`으로 유지하기로 했다.
- Colab runtime에서는 robosuite/MuJoCo mass-matrix compatibility와 신뢰한 init-state
  loading 옵션이 필요했다. Baseline model source에는 runtime patch를 적용하지 않았다.

## 2026-08-05 — Phase 2A: 정적 GraphSpec과 extractor

- Snapshot 하나를 deterministic directed graph로 바꾸는 `phase2.v1` 계약을 고정했다.
- Node는 robot/object/fixture/site, identity는 logical name이다.
- 공통 node numeric feature는 position, gripper/joint state와 validity mask를 포함한
  24차원이고 node type은 별도 one-hot이다.
- Valid position node 사이의 self-loop 없는 complete directed spatial edge를 사용했다.
- Colab live capture에서 graph 2개, graph당 node 24개와 edge 552개가 생성됐다.
- Predicate audit의 error는 unknown으로 유지했다. 이 graph는 controlled oracle graph이며
  RGB policy에서 simulator state 사용 가능성을 주장하지 않는다.

## 2026-08-05 — 문헌 재검토와 data 경로 개정

- 정적 graph snapshot만으로는 action-conditioned relational dynamics를 검증할 수 없음을
  확인했다.
- Episode 단위 split, action-bearing trajectory, temporal target, capability-aware label이
  필요하다고 판단했다.
- Phase 2A는 extractor regression baseline으로 유지하고 scripted Phase 2R과 official-demo
  Phase 2D를 분리했다.

## 2026-08-05 — Phase 2R scripted diagnostic

- Bounded/holding scripted probe, robot-base frame, semantic handler, transition dataset,
  scale-up validation을 구현했다.
- Robot-base coordinate와 containment semantic 처리에서 오류를 발견해 수정했다.
- Phase 2R은 extractor, contact/handler/frame, hard-negative 진단에는 유효하지만 scripted
  distribution이므로 main training source에서 제외하기로 했다.
- 관련 compact 결과는 `data/phase2r_*_summary.json`에 보존했다.

## 2026-08-05 — 초기 Phase 3 offline probe

- Graph pair와 action history로 future relation을 예측하는 offline probe 골격을 만들었다.
- Flat/non-graph baseline, graph encoder, action control, relation change metric을 비교했다.
- 초기 단일 split 결과는 낮은 label support와 dataset shortcut 가능성 때문에 architecture
  우월성 근거로 사용하지 않기로 했다.
- Task-family-held-out matched 재검증에서도 holding-positive와 changed-event support가
  부족한 문제가 드러났다.

## 2026-08-06 — Holding-positive 후속과 연구 roadmap v3

- Holding-positive 수집을 보완했지만 scripted source의 분포 한계를 해결하지 못했다.
- 연구 전체를 다시 조사해 main data source를 LIBERO official demonstration HDF5의 exact
  state replay로 변경했다.
- Phase 2D에서 full graph timeline, action alignment, holding state machine, event-centered
  sample을 만들고 Phase 3A QA 후 Phase 3B gate를 열도록 결정했다.
- `is_object_of_interest`와 BDDL-derived target relevance는 primary model input에서 제외했다.
- Phase 4는 Phase 3B와 Phase 3C gate 통과 전까지 차단했다.

## 2026-08-06 — Phase 2D official-demo replay

- `libero_spatial` task 0/1/2의 official demonstration 150개를 persistent Drive에서
  exact replay했다.
- Per-demo shard, merged task dataset, graph/action timing, provenance, fixed episode split을
  저장했다.
- 일시적으로 `holding=0`이 된 원인은 gripper contact mapping 문제였으며 runtime geom/body
  mapping을 바로잡아 전체 replay와 QA를 다시 수행했다.
- Sample/episode leakage 없이 task 0/1/2 natural graph dataset을 완성했다.
- Persistent root는 `/content/drive/MyDrive/Graph-CLaD/artifacts`로 고정했다.

## 2026-08-06 — Relation 및 event audit

- Contact, holding, on, inside와 temporal event support를 검사했다.
- Holding은 contact, gripper closure, 3-frame relative-pose stability, object following을
  조합한 heuristic weak label로 정의했다.
- `inside`는 valid label support가 없어 deferred했다.
- Holding-positive, onset, release, hard-negative index를 만들었다.
- Future event 정보를 이용해 선택한 challenge는 독립 test가 아니라 natural held-out
  episode의 event-enriched stress view로만 해석하기로 했다.

## 2026-08-07 — Phase 2D input-clean과 구조 정리

- Train input에서 target-derived relevance와 future-dependent field를 제거한 input-clean
  artifact를 만들고 structural QA를 통과했다.
- Colab 임시 `/content` source와 persistent Drive source의 역할을 분리했다.
- Phase별 source를 `scripts/phase*/`로 정리하고 과거 명령 호환을 위해 root wrapper를
  유지했다.
- Code snapshot, config, manifest, checkpoint, prediction, aggregate result를 함께
  보존하는 artifact 정책을 고정했다.

## 2026-08-07 — Legacy Phase 3 controlled experiment

- A100에서 smoke 후 task-family-held-out controlled experiment를 수행했다.
- 총 45 runs를 완료했다.
- 자연 prevalence 성능과 target-conditioned holding stress 성능이 크게 달라 하나의
  숫자로 합치지 않기로 했다.
- Holding을 primary relation으로 포함하고 target-aligned event dataset을 추가했다.
- Category-aware episode round-robin v2 sampler와 balanced-cap 비교를 수행했다.
- 이 결과는 모델/trainer 골격과 failure mode를 제공하지만 corrected protocol 이전
  결과이므로 최종 architecture 주장에는 사용하지 않는다.

## 2026-08-11 — Phase 3B-R1 계획 고정

- Complete graph보다 target-centric holder–object topology가 holding에 더 적합할 수 있다는
  가설을 세웠다.
- 비교 모델을 flat/pair MLP, complete/sparse GNN, late/global action,
  action-conditioned update의 통제 조합으로 정의했다.
- 평가 protocol을 먼저 고정하는 Gate E0, shortcut 진단 E1, architecture pilot E2,
  full comparison E3, action/edge control E4, history/context E5 순서를 정했다.
- Natural held-out test가 primary이고 challenge는 stress analysis임을 명시했다.

## 2026-08-11 — Holder–object smoke와 action 진단

- Sparse holder–object GNN G1 smoke에서 complete topology보다 유망한 task 0 신호를 봤다.
- G3 action-conditioned edge smoke와 action-signal diagnostic을 수행했지만 단일 task/seed
  근거에 불과했다.
- 이후 확인 결과 G1의 action은 message update가 아니라 prediction head에 들어가는
  late/global action이었다. 따라서 action-conditioned temporal edge 가설을 검증했다고
  주장하지 않기로 했다.
- B1-v2, G1-correct, S-0-G1, G1-train-shuffled의 near-parameter-matched 비교를 준비했다.

## 2026-08-11 — Holding metric과 parameter-count 교정

- 기존 display 일부가 holding metric의 nested path를 잘못 읽는 문제를 고쳤다.
- Near-matched parameter count는 다음과 같다.
  - B1-v2: hidden 55, 45,668 parameters.
  - G1-correct: hidden 48, 44,946 parameters.
  - S-0-G1: hidden 54, 44,784 parameters.
  - G1-train-shuffled: hidden 48, 44,946 parameters.
- Exact matching이 아니라 near matching임을 한계로 명시했다.

## 2026-08-12 — Reduced cross-fold gate 완료

- 3 task folds × 3 seeds × 4 conditions, 총 36/36 runs와 checkpoint 36개를 저장했다.
- Persistent artifact:
  - `phase3_reduced_crossfold_gate_v1.json`
  - `phase3B_R1_eval_manifest.json`
  - `checkpoints_reduced_crossfold_v1`
  - `code_snapshot_reduced_crossfold_v1`

9-run 평균은 다음과 같다.

| Model | Natural event F1 | Stress event F1 | Natural PR-AUC | Stress PR-AUC | Natural/Stress hard-neg FPR |
|---|---:|---:|---:|---:|---:|
| B1-v2 | 0.3285 | 0.7670 | 0.4047 | 0.4898 | 0.1075 / 0.1017 |
| G1 | 0.3400 | 0.7465 | 0.3872 | 0.4972 | 0.2146 / 0.1884 |
| S-0 | 0.3604 | 0.5927 | 0.3691 | 0.5159 | 0.2836 / 0.2613 |
| G1 train-shuffled | 0.3470 | 0.4746 | 0.2793 | 0.4093 | 0.3277 / 0.2931 |

- G1−B1 natural event F1은 +0.0115였지만 2/9 runs에서만 G1이 우세했다.
- Stress event F1 차이는 −0.0206이고 역시 2/9만 G1이 우세했다.
- B1 pair MLP가 가장 방어 가능한 baseline이며 graph의 일관된 추가 이점은 입증되지
  않았다.
- Natural release F1 약 0.21이 주요 병목이었다.
- 동일 task의 seed는 test episode를 공유하므로 9개 독립 표본으로 해석하지 않는다.

## 2026-08-12 — Corrected protocol v2 구현

Legacy 결과를 덮어쓰지 않고 별도 config/protocol/output root를 만들었다.

- Current auxiliary head의 action confound를 제거했다. 모든 비교 모델에 action-free
  current head를 사용하고 smoke에서는 동일한 loss 계약을 유지했다.
- Checkpoint criterion은 natural-validation conditional holding-event PR-AUC다.
- Future/current threshold는 natural validation에서 한 번 선택해 natural test, stress,
  perturbation control에 고정한다.
- Natural conditional/oracle-current event PR-AUC를 primary로 하고 thresholded F1,
  onset/release F1, hard-negative FPR, Brier, 10-bin ECE를 secondary로 저장한다.
- Predicted-current와 predicted-future를 쓰는 end-to-end event metric을 함께 저장한다.
- Sample probability, target, prediction, task, episode, sample ID, timestep, edge identity,
  event cluster를 gzip JSONL로 저장한다.
- Manifest QA에 task-local quota와 natural/stress overlap payload SHA256 검사를 추가했다.
- Train-shuffled donor는 task-local, episode-disjoint이며 action magnitude와 coarse state로
  matching하고 donor QA를 저장한다.

## 2026-08-12 — Corrected three-fold seed-0 gate

Task 1 seed 0 smoke 후 누락된 task 0/2만 실행해 12 runs를 결합했다. Task 1은 반복하지
않았다.

- G1−B1 conditional PR-AUC: task 0 −0.1161, task 1 +0.1322, task 2 +0.1717.
- Task-macro 차이 +0.0626, hierarchical bootstrap 95% CI [−0.0892, +0.1905].
- G1 event F1 0.3272, B1 0.4083.
- G1 release F1은 B1보다 +0.1626 [0.0701, 0.2967].
- G1 hard-negative FPR은 −0.3326 [−0.4201, −0.2429]로 개선됐다.
- 그러나 G1−train-shuffled G1 PR-AUC는 −0.0567 [−0.1943, +0.0572]이고 task 0에서
  부호가 뒤집혔다. Train-shuffled G1이 task-macro PR-AUC 0.4985와 event F1 0.4333으로
  가장 높았다.

결정은 `stop_gnn_three_seed_expansion_and_pivot_pair_local_temporal_encoder`다. 이는
graph가 쓸모없다는 뜻이 아니라 현재 late-action G1이 architecture/action gate를
통과하지 못했다는 뜻이다. Phase 4는 계속 차단했다.

## 2026-08-12 — Weak-label audit 준비

- Task 0/1/2 × onset/release/hard-negative × 10개, 총 90개 audit manifest를 만들었다.
- Trajectory-enriched v2는 각 item의 t~t+6 graph frame을 재구성했다.
- 요청 frame 592/592를 회수했고 missing/conflict가 없었다.
- Interactive viewer가 distance, relative xyz motion, object-following residual,
  contact/holding, arm/gripper action을 표시한다.
- 판정은 `pass`, `label_error`, `ambiguous` 중 사람이 명시적으로 입력해야 한다.
- 자동 판정을 human ground truth로 대체하지 않는다. 현재 review 상태는 0/90이다.

## 2026-08-12 — Pair-local temporal H0–H3 smoke

Task 1, seed 0에서 robot–object pair를 독립 처리하고 causal history/action의 2×2
factorial을 실행했다.

- H0: 현재 state, action 없음.
- H1: causal history, action 없음.
- H2: 현재 state + action.
- H3: causal history + action.

Natural conditional PR-AUC는 H0 0.3810, H1 0.4960, H2 0.5869, H3 0.5731이었다.
H2가 release F1 0.2143과 hard-negative FPR 0.0900으로 가장 좋았다. H3는 natural
end-to-end event PR-AUC 0.4227이 가장 높았지만 conditional PR-AUC와 release는 H2보다
낮았다. 한 task/seed 결과이므로 일반화 결론은 내리지 않았다.

## 2026-08-13 — Pair-local temporal three-fold seed-0 screen

H0–H3를 held-out task 0/1/2, seed 0에서 12/12 runs 완료했다.

- Result: `phase3_pair_local_temporal_threefold_seed0_v1.json`
- SHA256: `492d45521e6ccecbc4f0d89923f50d49642962d60c1c53daf093b5aec9b4d188`

Task-macro natural conditional PR-AUC:

| Model | PR-AUC | Event F1 | Release F1 | Hard-negative FPR |
|---|---:|---:|---:|---:|
| H0-state | 0.3626 | 0.3134 | 0.0931 | 0.3168 |
| H1-history | 0.4348 | 0.4920 | 0.0215 | 0.3358 |
| H2-action | 0.3941 | 0.3186 | 0.1729 | 0.2241 |
| H3-history-action | 0.4824 | 0.3617 | 0.1193 | 0.2339 |

- H3−H1 action increment는 PR-AUC +0.0476으로 3/3 task에서 양수였다.
- Release F1 +0.0978, hard-negative FPR −0.1019였지만 task 0 FPR은 악화됐다.
- H3−H0 PR-AUC +0.1198로 3/3 task에서 양수였으나 task 0 hard-negative FPR은
  0.1429에서 0.4762로 크게 악화됐다.
- Full three-seed factorial은 즉시 실행하지 않고 H3 aligned vs matched
  episode-disjoint train-action-shuffled control을 먼저 수행하기로 했다.

## 2026-08-13 — Action-alignment control 실행

- Protocol ID: `phase3-pair-local-temporal-action-alignment-seed0-v1`.
- Output:
  `/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/corrected_protocol_v2/pair_local_temporal_action_alignment_seed0_v1`
- Episode-disjoint matched action donor를 사용하며 aligned H3와 train-shuffled H3를
  same fold/seed로 비교한다.
- 이 기록을 정리하던 시점에 Colab PID 12616이 실행 중이었다. 완료 결과를 확인하기
  전에는 성능이나 gate 통과 여부를 기록하지 않는다.

## 2026-08-13 — 저장소와 Colab source 비파괴 정리

- 실행 중 process, checkpoint, Drive output을 읽기 전용으로 유지했다.
- Colab live stage와 local source를 SHA256으로 비교했다. Manifest builder, probe,
  pair-local model, helper가 일치했고 action-alignment config를 로컬에 반영했다.
- `scripts/phase*/`를 source of truth로 유지하고 위험한 `src/` migration은 하지 않았다.
- `scripts/research_paths.py`로 local/Colab artifact path를 공통화했다.
- 대용량 dataset/checkpoint/prediction은 복사하지 않고 경로와 checksum만 문서화했다.
- 임시 bundle/stage/cache는 삭제하지 않고 `docs/archive_candidates_20260813.md`에
  보관 후보로만 분류했다.

## 2026-08-13 — 공식 Phase 이름 교정과 문서 한국어화

- 처음 추가한 순번형 notebook 이름 `phase_01`~`phase_05`가 공식 roadmap Phase와
  충돌함을 확인했다. 특히 architecture training/evaluation은 Phase 4/5가 아니라 모두
  Phase 3B였다.
- Notebook을 공식 단계에 맞게 다음처럼 교정했다.
  - 환경 준비
  - Phase 0 baseline
  - Phase 1A state/API audit
  - Phase 2A static graph contract
  - Phase 2R scripted diagnostic
  - Phase 2D official-demo dataset
  - Phase 3A dataset/label QA
  - Phase 3B architecture gate와 evaluation/control
- Phase 3C와 Phase 4 이후 notebook은 gate 통과 전까지 만들지 않았다.
- 기존 영문 Markdown 원본은 `archive/pre_korean_translation_20260813.zip`에 보존하고
  활성 설명서와 연구 기록을 한국어로 정리했다. Code identifier, metric, path, SHA는
  재현성을 위해 원문 표기를 유지했다.

## 현재 gate와 다음 단계

1. 실행 중 action-alignment control의 완료 여부, stderr, expected run count, checkpoint,
   prediction, runtime manifest, code snapshot을 확인한다.
2. Aligned H3가 shuffled control보다 natural PR-AUC에서 최소 2개 task 우세한지 본다.
3. Release와 hard-negative 안전성을 함께 판단한다.
4. 90-item weak-label 수동 review를 완료한다.
5. Gate를 통과할 때만 H3/H1/H3-shuffled의 seeds 1/2 확대를 검토한다.
6. 이후 Phase 3C CLaD-aligned foresight bridge를 검토한다.
7. Phase 4에서는 semantic CLaD, pair-local temporal, graph-transition representation을
   같은 데이터와 동일 capacity probe로 비교한다.

현재까지 graph의 일반적 우월성, causal action 효과, 최종 Graph-CLaD representation의
우월성은 입증되지 않았다.
